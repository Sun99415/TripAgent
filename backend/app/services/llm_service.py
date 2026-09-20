"""LLM服务模块"""

import os
from hello_agents import HelloAgentsLLM
from ..config import get_settings

# 全局LLM实例
_llm_instance = None


def get_llm() -> HelloAgentsLLM:
    """
    获取LLM实例(单例模式)

    Returns:
        HelloAgentsLLM实例
    """
    global _llm_instance

    if _llm_instance is None:
        settings = get_settings()

        # HelloAgentsLLM会自动从环境变量读取配置
        # 包括 LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_ID, LLM_TIMEOUT 等
        # 注意: max_tokens 库本身不会从环境变量读取,需要显式传入
        init_kwargs = {}
        max_tokens_env = os.getenv("MAX_TOKENS")
        if max_tokens_env:
            init_kwargs["max_tokens"] = int(max_tokens_env)

        _llm_instance = HelloAgentsLLM(**init_kwargs)

        print(f"✅ LLM服务初始化成功")
        print(f"   提供商: {_llm_instance.provider}")
        print(f"   模型: {_llm_instance.model}")
        print(f"   超时: {_llm_instance.timeout}s")
        print(f"   最大token: {_llm_instance.max_tokens}")

    return _llm_instance


def reset_llm():
    """重置LLM实例(用于测试或重新配置)"""
    global _llm_instance
    _llm_instance = None

