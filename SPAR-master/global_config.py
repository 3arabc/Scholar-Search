#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : Global configuration settings for Scholar Paper Agent Retrieval
# ==================================================================
import os
import json
import arxiv
from typing import Dict, List, Any

# Debug mode
DEBUG = False

# =============================================================================
# OPENAI CONFIGURATION
# =============================================================================
API_KEY = os.getenv(
    "SILICONFLOW_API_KEY",
    "sk-uejssktdixvxpaorxeonyztxomulwlnxkqmnisonsiffepsn",
)
ENDPOINT = os.getenv(
    "SILICONFLOW_BASE_URL",
    "https://api.siliconflow.cn/v1/chat/completions",
)
DEPLOYMENT_NAME = "deepseek-ai/DeepSeek-V3.2"

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

# 二次筛选（已关闭：与第一阶段评分重复，且串行评分耗时巨大）
ENABLE_LLM_RERANK = False        # 关闭 LLM 二次过滤
LLM_RERANK_THRESHOLD = 0.7      # 保留分数阈值

SAVE_ID2DOCS = True
RELEVANCE_SCORE = 0.5
WEB_RETRY_NUM = 2 #wsl-71

# Query threshold settings #wsl-71阈值改小一点 #0.8，0.85
QUERY_LOW_THRESHOLD = 0.2
QUERY_HIGH_THRESHOLD = 0.7
CORRECT_SCORE_THRESHOLD = 0.7
EXPAND_SCORE_THRESHOLD = 0.8
QUERY_TO_SEARCH_THRESHOLD = 0.8

# Generation settings
LENGTH_GEN_QUERY_FROM_CITATION = 12288

# =============================================================================
# WEB API CONFIGURATION
# =============================================================================
TRY_COUNT = 4
LLM_TRY_COUNT = 4
LLM_PARALLEL_NUM = 4
LLM_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"


API_TRY_COUNT = 4
API_PARALLEL_REQUEST = 1

SLEEP_TIME_LLM = 2.0

# =============================================================================
# SEARCH HYPERPARAMETERS
# =============================================================================
DO_FUSION_JUDGE = True
FUSION_TEMPLATE = "AUTOMATIC"  # Options: "WITHEXPLAIN", "AUTOMATIC"

# Query processing settings
QUERY_NUM_PRUNED = 8  # 每层保留的扩展查询数（增大以覆盖更多方向）
RETRIEVAL_QUERY_BATCH_SIZE = 6  # Batch size for query processing to avoid excessive searching

# Document processing settings
DOCS_TO_EXPAND = 60  # 引用搜索的文档数（增大以覆盖更多引用）
REFERENCE_DOC_PRUNED = 40  # 每篇文档提取的参考文献数
REFERENCE_OCCUR_FREQUENCY = 0.6
REFERENCE_DOC_NUM_TO_GEN_NEW_QUERY = 15  # 用于生成新查询的文档数（增大以丰富上下文）

# 引用搜索配置
DO_REFERENCE_SEARCH = True  # 启用引用搜索，从相关论文的参考文献中发现更多论文

# Similarity thresholds（已恢复原版0.5，减少低质量文档进入后续环节）
REFERENCE_DOC_SIM_THRESHOLD = 0.5
BEGIN_SIM_THRESHOLD = 0.5
PASS_SIM_THRESHOLD = 0.5
REFERENCE_EXPAND_THRESHOLD = 0.7  # 引用扩展门槛：仅 sim_score ≥ 此值的论文才取其参考文献

# Search routes configuration
SEARCH_ROUTES: List[str] = ["arxiv", "openalex"]

# =============================================================================
# EXTERNAL API KEYS
# =============================================================================
# Register at: https://google.serper.dev/search
GOOGLE_SERPER_KEY = os.getenv("GOOGLE_SERPER_KEY", "xxx")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", None)
# Semantic Scholar API key (currently invalid)
S2_API_KEY = os.getenv("S2_API_KEY", None)

# =============================================================================
# SEARCH FEATURES
# =============================================================================
# RERANK feature toggle
RERANK =os.getenv("DO_RERANK",True)
RERANK = True #wsl-73

KEY_WORDS_NUM =5  # 关键词数量（从2提升到5，扩大搜索覆盖面）
LLM_PARREL_NUM=4  # 并行LLM评分线程数（已从2提升到4以加速）
# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================
"""
PROXIES: Dict[str, str] = {
    "http": os.getenv("HTTP_PROXY", "http://localhost:1080"),
    "https": os.getenv("HTTPS_PROXY", "http://localhost:1080")
}
"""
PROXIES: Dict[str, str] = {
    # "http": os.getenv("HTTP_PROXY", "http://127.0.0.1:7890"),  # 改为你的代理端口
    # "https": os.getenv("HTTPS_PROXY", "http://127.0.0.1:7890")  # Clash 默认 7890
}

# ArXiv client configuration
ARXIV_CLIENT = arxiv.Client(delay_seconds=0.05)

# =============================================================================
# RERANKING CONFIGURATION
# =============================================================================
ENABLE_RERANK = False
RERANK_MODEL = "deepseek-ai/DeepSeek-V3.2"  # 与主模型一致

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
def validate_config() -> bool:
    """
    Validate essential configuration settings.

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_keys = [API_KEY, ENDPOINT]

    if not all(key and key != "your_openai_api_key_here" for key in required_keys):
        print("Warning: OpenAI API configuration is incomplete")
        return False

    if QUERY_LOW_THRESHOLD >= QUERY_HIGH_THRESHOLD:
        print("Error: QUERY_LOW_THRESHOLD must be less than QUERY_HIGH_THRESHOLD")
        return False

    return True

# Validate configuration on import
if __name__ == "__main__":
    if validate_config():
        print("Configuration validation passed")
