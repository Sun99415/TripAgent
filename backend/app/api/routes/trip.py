"""旅行规划API路由"""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ...models.schemas import (
    TripRequest,
    TripPlanResponse,
    ErrorResponse
)
from ...agents.trip_planner_agent import get_trip_planner_agent

router = APIRouter(prefix="/trip", tags=["旅行规划"])


@router.post(
    "/plan",
    response_model=TripPlanResponse,
    summary="生成旅行计划",
    description="根据用户输入的旅行需求,生成详细的旅行计划"
)
async def plan_trip(request: TripRequest):
    """
    生成旅行计划

    Args:
        request: 旅行请求参数

    Returns:
        旅行计划响应
    """
    try:
        print(f"\n{'='*60}")
        print(f"📥 收到旅行规划请求:")
        print(f"   城市: {request.city}")
        print(f"   日期: {request.start_date} - {request.end_date}")
        print(f"   天数: {request.travel_days}")
        print(f"{'='*60}\n")

        # 获取Agent实例
        print("🔄 获取多智能体系统实例...")
        agent = get_trip_planner_agent()

        # 生成旅行计划
        print("🚀 开始生成旅行计划...")
        trip_plan = agent.plan_trip(request)

        print("✅ 旅行计划生成成功,准备返回响应\n")

        return TripPlanResponse(
            success=True,
            message="旅行计划生成成功",
            data=trip_plan
        )

    except Exception as e:
        print(f"❌ 生成旅行计划失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"生成旅行计划失败: {str(e)}"
        )


@router.post(
    "/plan/stream",
    summary="流式生成旅行计划(SSE)",
    description="以 text/event-stream 逐阶段推送行程生成进度与最终结果"
)
def plan_trip_stream(request: TripRequest):
    """
    流式生成旅行计划。

    返回 SSE 事件流,每个事件为 JSON 字符串,事件类型见 plan_trip_events 文档:
    status(进度) / plan_delta(JSON增量) / done(完成) / error(失败)。
    """
    agent = get_trip_planner_agent()

    def event_generator():
        for event in agent.plan_trip_events(request):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/health",
    summary="健康检查",
    description="检查旅行规划服务是否正常"
)
async def health_check():
    """健康检查"""
    try:
        # 检查多智能体系统是否可用(初始化失败时返回503)
        planner = get_trip_planner_agent()

        return {
            "status": "healthy",
            "service": "trip-planner",
            "agents": {
                "attraction": {
                    "name": planner.attraction_agent.name,
                    "tools_count": len(planner.attraction_agent.list_tools())
                },
                "weather": {
                    "name": planner.weather_agent.name,
                    "tools_count": len(planner.weather_agent.list_tools())
                },
                "hotel": {
                    "name": planner.hotel_agent.name,
                    "tools_count": len(planner.hotel_agent.list_tools())
                },
                "planner": {
                    "name": planner.planner_agent.name,
                    "tools_count": len(planner.planner_agent.list_tools())
                }
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"服务不可用: {str(e)}"
        )

