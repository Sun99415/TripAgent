"""多智能体旅行规划系统"""

import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Iterator, Optional

from loguru import logger
from hello_agents import SimpleAgent
from hello_agents.tools import MCPTool

from ..services.llm_service import get_llm
from ..services.cache_service import get_cache
from ..models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo, Location, Hotel
from ..config import get_settings

# ============ Agent提示词 ============

ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。你的任务是根据城市和用户偏好搜索合适的景点。

**重要提示:**
你必须使用工具来搜索景点!不要自己编造景点信息!

**工具调用格式:**
使用maps_text_search工具时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_text_search:keywords=景点关键词,city=城市名]`

**示例:**
用户: "搜索北京的历史文化景点"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=历史文化,city=北京]

用户: "搜索上海的公园"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=公园,city=上海]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
3. 参数用逗号分隔
"""

WEATHER_AGENT_PROMPT = """你是天气查询专家。你的任务是查询指定城市的天气信息。

**重要提示:**
你必须使用工具来查询天气!不要自己编造天气信息!

**工具调用格式:**
使用maps_weather工具时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_weather:city=城市名]`

**示例:**
用户: "查询北京天气"
你的回复: [TOOL_CALL:amap_maps_weather:city=北京]

用户: "上海的天气怎么样"
你的回复: [TOOL_CALL:amap_maps_weather:city=上海]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
"""

HOTEL_AGENT_PROMPT = """你是酒店推荐专家。你的任务是根据城市和景点位置推荐合适的酒店。

**重要提示:**
你必须使用工具来搜索酒店!不要自己编造酒店信息!

**工具调用格式:**
使用maps_text_search工具搜索酒店时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_text_search:keywords=酒店,city=城市名]`

**示例:**
用户: "搜索北京的酒店"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=酒店,city=北京]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
3. 关键词使用"酒店"或"宾馆"
"""

PLANNER_AGENT_PROMPT = """你是行程规划专家。你的任务是根据景点信息和天气信息,生成详细的旅行计划。

请严格按照以下JSON格式返回旅行计划:
```json
{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.397128, "latitude": 39.916527},
        "price_range": "300-500元",
        "rating": "4.5",
        "distance": "距离景点2公里",
        "type": "经济型酒店",
        "estimated_cost": 400
      },
      "attractions": [
        {
          "name": "景点名称",
          "address": "详细地址",
          "location": {"longitude": 116.397128, "latitude": 39.916527},
          "visit_duration": 120,
          "description": "景点详细描述",
          "category": "景点类别",
          "ticket_price": 60
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐推荐", "description": "早餐描述", "estimated_cost": 30},
        {"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50},
        {"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "南风",
      "wind_power": "1-3级"
    }
  ],
  "overall_suggestions": "总体建议",
  "budget": {
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }
}
```

**重要提示:**
1. weather_info数组必须包含每一天的天气信息
2. 温度必须是纯数字(不要带°C等单位)
3. 每天安排2-3个景点
4. 考虑景点之间的距离和游览时间
5. 每天必须包含早中晚三餐
6. 提供实用的旅行建议
7. **必须包含预算信息**:
   - 景点门票价格(ticket_price)
   - 餐饮预估费用(estimated_cost)
   - 酒店预估费用(estimated_cost)
   - 预算汇总(budget)包含各项总费用
"""


class MultiAgentTripPlanner:
    """多智能体旅行规划系统"""

    def __init__(self):
        """初始化多智能体系统"""
        print("🔄 开始初始化多智能体旅行规划系统...")

        try:
            settings = get_settings()
            self.llm = get_llm()

            # 前置检查: amap-mcp-server 通过 uvx 启动,缺少 uvx 时快速失败,
            # 避免工具静默发现失败后每个Agent空跑多轮LLM调用
            if not shutil.which("uvx"):
                raise RuntimeError(
                    "未找到 uvx 命令,无法启动高德地图MCP服务(amap-mcp-server)。"
                    "请先安装 uv: pip install uv (安装后重新打开终端再启动服务)"
                )

            # 创建共享的MCP工具(只创建一次)
            print("  - 创建共享MCP工具...")
            self.amap_tool = MCPTool(
                name="amap",
                description="高德地图服务",
                server_command=["uvx", "amap-mcp-server"],
                env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
                auto_expand=True
            )
            self.amap_tool.expandable=True

            # 创建景点搜索Agent
            print("  - 创建景点搜索Agent...")
            self.attraction_agent = SimpleAgent(
                name="景点搜索专家",
                llm=self.llm,
                system_prompt=ATTRACTION_AGENT_PROMPT
            )
            self.attraction_agent.add_tool(self.amap_tool)

            # 创建天气查询Agent
            print("  - 创建天气查询Agent...")
            self.weather_agent = SimpleAgent(
                name="天气查询专家",
                llm=self.llm,
                system_prompt=WEATHER_AGENT_PROMPT
            )
            self.weather_agent.add_tool(self.amap_tool)

            # 创建酒店推荐Agent
            print("  - 创建酒店推荐Agent...")
            self.hotel_agent = SimpleAgent(
                name="酒店推荐专家",
                llm=self.llm,
                system_prompt=HOTEL_AGENT_PROMPT
            )
            self.hotel_agent.add_tool(self.amap_tool)

            # 创建行程规划Agent(不需要工具)
            print("  - 创建行程规划Agent...")
            self.planner_agent = SimpleAgent(
                name="行程规划专家",
                llm=self.llm,
                system_prompt=PLANNER_AGENT_PROMPT
            )

            # 校验MCP工具是否真正发现成功
            # 注意: MCPTool._discover_tools() 发现失败时会静默置空工具列表,
            # 此时只会注册一个名为 amap 的"壳工具",后续所有 amap_maps_xxx 调用都会落空,
            # 每个Agent将空跑3轮工具迭代,请求时间被放大数倍并最终前端超时
            agents_to_check = [
                ("景点搜索", self.attraction_agent),
                ("天气查询", self.weather_agent),
                ("酒店推荐", self.hotel_agent),
            ]
            tool_counts = {label: len(agent.list_tools()) for label, agent in agents_to_check}

            print(f"✅ 多智能体系统初始化成功")
            for label, count in tool_counts.items():
                print(f"   {label}Agent: {count} 个工具")

            failed_agents = [label for label, count in tool_counts.items() if count <= 1]
            if failed_agents:
                raise RuntimeError(
                    f"高德地图MCP工具加载失败({','.join(failed_agents)}Agent工具数<=1)。"
                    "请依次排查: 1) 命令行执行 `uvx amap-mcp-server` 确认能正常启动; "
                    "2) 检查 .env 中 AMAP_API_KEY 是否正确有效; "
                    "3) 检查网络能否访问 PyPI(下载amap-mcp-server)和高德服务"
                )

        except Exception as e:
            print(f"❌ 多智能体系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def plan_trip(self, request: TripRequest) -> TripPlan:
        """
        使用多智能体协作生成旅行计划(同步版)。

        遍历流式事件,返回最终计划;出错时回退到占位计划。
        """
        for event in self.plan_trip_events(request):
            if event.get("type") == "done":
                return TripPlan(**event["plan"])
            if event.get("type") == "error":
                return self._create_fallback_plan(request)
        return self._create_fallback_plan(request)

    def plan_trip_events(self, request: TripRequest) -> Iterator[Dict[str, Any]]:
        """
        流式生成旅行计划,逐步 yield 事件(dict)。

        事件类型:
        - {"type": "status", "stage": str, "message": str}   进度状态
        - {"type": "plan_delta", "text": str}                 行程JSON增量文本
        - {"type": "done", "plan": dict, "from_cache": bool} 完成
        - {"type": "error", "message": str}                   失败
        """
        cache = get_cache()
        cache_key = self._build_cache_key(request)

        # 1. 命中缓存直接返回
        cached = cache.get_json(cache_key)
        if cached is not None:
            logger.info("命中结果缓存: {}", cache_key)
            yield {"type": "status", "stage": "cache_hit", "message": "命中缓存,直接返回"}
            yield {"type": "done", "plan": cached, "from_cache": True}
            return

        try:
            logger.info("开始规划 {} {} 天行程", request.city, request.travel_days)
            yield {"type": "status", "stage": "start", "message": f"开始规划 {request.city} {request.travel_days} 天行程"}

            # 2. 并行搜索景点/天气/酒店(三者无依赖)
            yield {"type": "status", "stage": "search", "message": "并发搜索景点、天气、酒店..."}
            attraction_response, weather_response, hotel_response = self._search_parallel(request)
            logger.info(
                "搜索完成: 景点{}字符 / 天气{}字符 / 酒店{}字符",
                len(attraction_response), len(weather_response), len(hotel_response),
            )
            yield {"type": "status", "stage": "search_done", "message": "搜索完成,开始生成行程计划"}

            # 3. 流式生成行程(逐步推送增量文本)
            planner_query = self._build_planner_query(request, attraction_response, weather_response, hotel_response)
            planner_messages = [
                {"role": "system", "content": PLANNER_AGENT_PROMPT},
                {"role": "user", "content": planner_query},
            ]

            parts: List[str] = []
            for delta in self._stream_completion_iter(planner_messages):
                parts.append(delta)
                yield {"type": "plan_delta", "text": delta}

            planner_response = "".join(parts)
            if not planner_response.strip():
                raise RuntimeError("行程规划模型返回内容为空,请检查LLM配置或稍后重试")

            # 4. 解析并写缓存(仅成功结果入缓存,避免污染)
            logger.info("LLM原始响应(len={}): {}", len(planner_response), planner_response[:800])
            trip_plan = self._parse_response(planner_response, request)
            if trip_plan is None:
                trip_plan = self._create_fallback_plan(request)
            else:
                cache.set_json(cache_key, trip_plan.model_dump())

            yield {"type": "done", "plan": trip_plan.model_dump(), "from_cache": False}

        except Exception as e:
            logger.exception("生成旅行计划失败: {}", e)
            fallback = self._create_fallback_plan(request)
            yield {"type": "error", "message": str(e)}
            yield {"type": "done", "plan": fallback.model_dump(), "from_cache": False}

    def _search_parallel(self, request: TripRequest) -> tuple[str, str, str]:
        """并行执行景点/天气/酒店三路搜索,单路失败不影响整体(置空兜底)。"""
        tasks = {
            "attraction": (self.attraction_agent.run, self._build_attraction_query(request)),
            "weather": (self.weather_agent.run, f"请查询{request.city}的天气信息"),
            "hotel": (self.hotel_agent.run, f"请搜索{request.city}的{request.accommodation}酒店"),
        }

        results: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_map = {executor.submit(fn, arg): name for name, (fn, arg) in tasks.items()}
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    results[name] = future.result() or ""
                except Exception as e:
                    logger.warning("{} 搜索失败: {}", name, e)
                    results[name] = ""

        return results["attraction"], results["weather"], results["hotel"]

    def _build_cache_key(self, request: TripRequest) -> str:
        """基于请求参数构建缓存key,相同请求命中同一缓存。"""
        raw = json.dumps(
            {
                "city": request.city,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "travel_days": request.travel_days,
                "transportation": request.transportation,
                "accommodation": request.accommodation,
                "preferences": sorted(request.preferences or []),
                "free_text_input": request.free_text_input or "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"trip:plan:{digest}"
    
    def _stream_completion_iter(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """
        流式消费底层响应,逐步 yield 正文增量文本(兼容推理型模型)。

        库的 think() 只读取 delta.content,而推理模型可能把输出全部放在
        reasoning_content 中,导致正文为空。这里同时收集两个字段,并在
        思考阶段每5秒记录一次心跳,便于区分“仍在生成”和“真的卡住”。
        """
        client = getattr(self.llm, "_client", None)
        if client is None:
            # 无底层流式 client 时退化为一次性调用
            yield self.llm.invoke(messages)
            return

        stream = client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            stream=True,
        )

        parts: List[str] = []
        reasoning_parts: List[str] = []
        start = time.time()
        last_beat = start
        answering = False
        chunk_count = 0

        for chunk in stream:
            chunk_count += 1
            if chunk_count <= 3:
                logger.info("原始chunk#{}: choices={}", chunk_count, len(chunk.choices))
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            text = getattr(delta, "content", None)
            if text:
                if not answering:
                    answering = True
                    r_len = sum(len(p) for p in reasoning_parts)
                    logger.info("思考结束(思维链{}字符,用时{:.0f}s),开始输出行程JSON", r_len, time.time() - start)
                parts.append(text)
                yield text

            now = time.time()
            if not answering and now - last_beat >= 5:
                last_beat = now
                r_len = sum(len(p) for p in reasoning_parts)
                logger.info("模型思考中(思维链{}字符),已等待{:.0f}s...", r_len, now - start)

        # 若正文为空但存在思维链,改用思维链兜底(推理模型可能把结果全放入 reasoning_content)
        if not "".join(parts).strip() and reasoning_parts:
            logger.warning("模型未返回正文,仅返回思维链,临时改用思维链内容进行解析")
            yield "".join(reasoning_parts)

        logger.info(
            "流式结束: chunk数{}, 正文{}字符, 思维链{}字符, 总耗时{:.0f}s",
            chunk_count, sum(len(p) for p in parts), sum(len(p) for p in reasoning_parts), time.time() - start,
        )

    def _build_attraction_query(self, request: TripRequest) -> str:
        """构建景点搜索查询 - 直接包含工具调用"""
        keywords = []
        if request.preferences:
            # 只取第一个偏好作为关键词
            keywords = request.preferences[0]
        else:
            keywords = "景点"

        # 直接返回工具调用格式
        query = f"请使用amap_maps_text_search工具搜索{request.city}的{keywords}相关景点。\n[TOOL_CALL:amap_maps_text_search:keywords={keywords},city={request.city}]"
        return query

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        """截断过长的工具返回内容,避免发给规划模型的上下文过大、拖慢生成"""
        text = text or ""
        if len(text) <= limit:
            return text
        return f"{text[:limit]}\n...(内容过长已截断,原文共{len(text)}字符)"

    def _build_planner_query(self, request: TripRequest, attractions: str, weather: str, hotels: str = "") -> str:
        """构建行程规划查询"""
        query = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

**基本信息:**
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

**景点信息:**
{self._truncate(attractions, 4000)}

**天气信息:**
{self._truncate(weather, 1500)}

**酒店信息:**
{self._truncate(hotels, 3000)}

**要求:**
1. 每天安排2-3个景点
2. 每天必须包含早中晚三餐
3. 每天推荐一个具体的酒店(从酒店信息中选择)
3. 考虑景点之间的距离和交通方式
4. 返回完整的JSON格式数据
5. 景点的经纬度坐标要真实准确
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"

        return query
    
    def _parse_response(self, response: str, request: TripRequest) -> Optional[TripPlan]:
        """
        解析Agent响应
        
        Args:
            response: Agent响应文本
            request: 原始请求
            
        Returns:
            旅行计划
        """
        try:
            # 尝试从响应中提取JSON
            # 查找JSON代码块
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                # 直接查找JSON对象
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("响应中未找到JSON数据")
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 转换为TripPlan对象
            trip_plan = TripPlan(**data)
            
            return trip_plan
            
        except Exception as e:
            logger.warning("解析响应失败: {},将使用备用方案", e)
            return None
    
    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """创建备用计划(当Agent失败时)"""
        from datetime import datetime, timedelta

        # 常用城市中心坐标,避免失败回退时所有城市都落到北京坐标
        city_centers = {
            "北京": (116.407, 39.904),
            "上海": (121.474, 31.230),
            "广州": (113.264, 23.129),
            "深圳": (114.058, 22.543),
            "杭州": (120.155, 30.274),
            "成都": (104.066, 30.573),
            "西安": (108.940, 34.341),
            "重庆": (106.551, 29.563),
            "武汉": (114.305, 30.593),
            "南京": (118.797, 32.060),
        }
        # 未收录城市回退到中国地理中心
        base_lng, base_lat = city_centers.get(request.city, (104.195, 35.862))

        # 解析日期
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")

        # 创建每日行程
        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)

            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j+1}",
                        address=f"{request.city}市",
                        location=Location(longitude=base_lng + i*0.01 + j*0.005, latitude=base_lat + i*0.01 + j*0.005),
                        visit_duration=120,
                        description=f"这是{request.city}的著名景点",
                        category="景点"
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐")
                ]
            )
            days.append(day_plan)
        
        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程,建议提前查看各景点的开放时间。"
        )


# 全局多智能体系统实例
_multi_agent_planner = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例(单例模式)"""
    global _multi_agent_planner

    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()

    return _multi_agent_planner

