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

#set SILICONFLOW_API_KEY=sk-uejssktdixvxpaorxeonyztxomulwlnxkqmnisonsiffepsn
#set GOOGLE_SERPER_KEY=28cca99adde1d0a52ecd6b3b481b37f093d499e3
#set OPENALEX_API_KEY=HVx0tDeogC37dQ10aIPTm5
#python demo_app_with_front.py
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
DEPLOYMENT_NAME = "Qwen3-8B"

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

# wsl-73精准过滤配置
ENABLE_LLM_RERANK = False       # 是否启用 LLM 二次过滤

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
LLM_TRY_COUNT = 2  #wsl-74重试次数
LLM_PARALLEL_NUM = 4
LLM_MODEL_NAME = "Qwen3-8B"  # "Qwen3-8B"


API_TRY_COUNT = 4
API_PARALLEL_REQUEST = 1

SLEEP_TIME_LLM = 1.0 #wsl-74重试等待时间

# =============================================================================
# SEARCH HYPERPARAMETERS
# =============================================================================
DO_FUSION_JUDGE = True
FUSION_TEMPLATE = "AUTOMATIC"  # Options: "WITHEXPLAIN", "AUTOMATIC"

# Query processing settings
QUERY_NUM_PRUNED = 4  #wsl-71 Number of queries to use for search
RETRIEVAL_QUERY_BATCH_SIZE = 6  # Batch size for query processing to avoid excessive searching

# Document processing settings
DOCS_TO_EXPAND = 40
REFERENCE_DOC_PRUNED = 40  #wsl-71 Number of references to extract from each relevant document
REFERENCE_OCCUR_FREQUENCY = 0.6
REFERENCE_DOC_NUM_TO_GEN_NEW_QUERY = 10  #wsl-71 Number of reference docs used to generate new queries

# Similarity thresholds #wsl-71相似度阈值，影响大 #0.6，0.5
REFERENCE_DOC_SIM_THRESHOLD = 0.4
BEGIN_SIM_THRESHOLD = 0.3
PASS_SIM_THRESHOLD = 0.3

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
DO_REFERENCE_SEARCH = False  # Toggle reference-based search
RERANK =os.getenv("DO_RERANK",True)
RERANK = True #wsl-73

KEY_WORDS_NUM =2
LLM_PARREL_NUM=4  #wsl-74并发
#wsl-710
SIM_THRESHOLD = 0.3
MAX_DOCS = 200               # 原来可能10，增大到200
# 重排序条件
RERANK_SORT_BY = 'year'   # 默认按年份排序
# 过滤配置
FILTER_YEAR_START = 2020
FILTER_YEAR_END = 2025
FILTER_MIN_CITATIONS = 10
FILTER_FIELDS = ["Computer Science"]  # 支持多个
# 是否启用领域过滤（如果 FILTER_FIELDS 为空，即使启用也无效）
FILTER_ENABLE_FIELDS = True

# 是否启用年份过滤
FILTER_ENABLE_YEAR = True

# 是否启用引用过滤
FILTER_ENABLE_CITATIONS = True

# 当文档缺少某字段时的处理方式：True 表示该条件通过（即不因此丢弃），False 表示丢弃
FILTER_MISSING_FIELD_PASS = False
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
RERANK_MODEL = "Qwen3-8B"

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
