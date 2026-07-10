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

# 尝试自动从同目录下的 .env 文件中加载敏感的 API Key，避免跨进程环境变量丢失与安全泄露
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip()

# Debug mode
DEBUG = False  # 调试模式开关，开启后会输出更详细的底层网络交互和打分调试日志

# =============================================================================
# OPENAI CONFIGURATION
# =============================================================================
API_KEY = os.getenv(  # OpenAI 或本地兼容接口的 API 密钥，优先读取环境变量中的值
    "OPENAI_API_KEY",
    "ollama",
)
ENDPOINT = os.getenv(  # API 连接的基准 URL，如本地 Ollama 接口、或者兼容 OpenAI 格式的中转域名
    "OPENAI_ENDPOINT",
    "http://localhost:11434/v1",
)
DEPLOYMENT_NAME = "gpt-4o"  # 默认的商业模型部署名称

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
SAVE_ID2DOCS = True  # 是否将解析到的文献 ID 和详细信息数据落库缓存，开启后能避免对同一篇文献进行重复的 API 网络抓取
RELEVANCE_SCORE = 0.5  # 文献判定相似度分数。大模型打分高于该分值的论文，才会被判定为高相关论文（进入最终输出列表）
WEB_RETRY_NUM = 1  # 外部网络 API 连接在超时或失败时的自动重试次数

# Query threshold settings
QUERY_LOW_THRESHOLD = 0.2  # 意图分析阶段判定问题相似度的最低过滤线，低于该阈值的子查询将被直接剪枝抛弃
QUERY_HIGH_THRESHOLD = 0.8  # 意图分析高相似度阈值
CORRECT_SCORE_THRESHOLD = 0.8  # 查询语句自适应纠正的打分及格阈值
EXPAND_SCORE_THRESHOLD = 0.85  # 触发意图向外层层拓展分支的决策分值阈值
QUERY_TO_SEARCH_THRESHOLD = 0.85  # 判定查询是否能被直接拿去搜索学术库的相似度阈值

# Generation settings
LENGTH_GEN_QUERY_FROM_CITATION = 12288  # 当从引文生成新 Query 时，限制的大模型最大输入 Context Token 长度

# =============================================================================
# WEB API CONFIGURATION
# =============================================================================
TRY_COUNT = 4  # 网络搜索 API 请求的总尝试次数
LLM_TRY_COUNT = 4  # 大模型文本生成在遭遇限流/解析失败时的最大重试次数
LLM_PARALLEL_NUM = 4  # 大模型打分或生成的最大并发度。调大可缩短端到端运行时长，但需防范大模型服务的限流（Ollama 推荐设为 2~4）
LLM_MODEL_NAME = "qwen3-4b"  # 真正调用的本地/云端大模型名称，当前绑定为您本地加载的 "qwen3-4b"


API_TRY_COUNT = 4  # 学术 API 请求时的最大尝试上限
API_PARALLEL_REQUEST = 1  # 学术接口并发请求数（防范如 ArXiv 的严格限流机制，故设为 1，不建议调大）

SLEEP_TIME_LLM = 2.0

# =============================================================================
# SEARCH HYPERPARAMETERS
# =============================================================================
DO_FUSION_JUDGE = True  # 是否启用决策打分融合，为 True 时大模型将利用更严密的评分模版输出
FUSION_TEMPLATE = "AUTOMATIC"  # 融合判定评分模版。选项: "WITHEXPLAIN"(附带推理过程打分) ； "AUTOMATIC"(纯打分，响应快且少消耗 token)

# Query processing settings
QUERY_NUM_PRUNED = 2  # 大模型生成了多个子查询后，限制每一题真正去执行检索的前 N 个最优子查询。设为 2 可在控制 API 费用的同时保留最核心分支
RETRIEVAL_QUERY_BATCH_SIZE = 6  # 查询任务的批处理队列大小，防止队列溢出导致过度并发检索

# Document processing settings
DOCS_TO_EXPAND = 40  # 检索树第一层在判定扩展阶段，用于候选合并与去重时的最大篇数上限
REFERENCE_DOC_PRUNED = 20  # 启用引文回溯时，允许从每篇高相关文献中最多提取其前 20 篇参考文献进行深挖
REFERENCE_OCCUR_FREQUENCY = 0.6  # 评估引用时，判定某引用为高频引用的交叉共现频率线
REFERENCE_DOC_NUM_TO_GEN_NEW_QUERY = 2  # 触发第二层新 Query 时，最多采用前 2 篇相关文献作为背景知识生成

# Similarity thresholds
REFERENCE_DOC_SIM_THRESHOLD = 0.6  # 提取参考文献的过滤阈值，只有相似度大于 0.6 的论文的参考文献才会被拉取
BEGIN_SIM_THRESHOLD = 0.5  # 起始检索匹配的底线分值
PASS_SIM_THRESHOLD = 0.5  # 判定为相关的底线分值，与 `score_thresh` 协同控制最后推荐的筛选关卡

# Search routes configuration
SEARCH_ROUTES: List[str] = ["arxiv", "openalex"]  # 启用的搜索引擎源。系统会并发去 arxiv, openalex 检索，可根据题目自动注入 "pubmed"

# =============================================================================
# EXTERNAL API KEYS
# =============================================================================
# Register at: https://google.serper.dev/search
GOOGLE_SERPER_KEY = os.getenv("GOOGLE_SERPER_KEY", "xxx")  # Google 搜索辅助引擎 Serper API 密钥，用于辅助进行论文检索

# Semantic Scholar API key (currently invalid)
SEMANTIC_SCHOLAR_API_KEY = os.getenv("S2_API_KEY", "")  # Semantic Scholar (S2) 的 API 密钥（可留空）

# =============================================================================
# SEARCH FEATURES
# =============================================================================
DO_REFERENCE_SEARCH = False  # 【核心提分开关】是否开启引用网络回溯查找。设为 True 时，会自动抓取文献的参考文献/被引网络，这能大幅提升 Crawler Recall 爬虫召回上限！
RERANK =os.getenv("DO_RERANK",True)  # 是否在最后一轮启用重排序打分

KEY_WORDS_NUM =2  # 大模型为子查询提取核心关键词的数量
LLM_PARREL_NUM=2  # 本地并发限制数
# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================
PROXIES: Dict[str, str] = {  # 本地代理配置。如果在中国大陆环境下请求 ArXiv/OpenAlex 超时，可在此配置本地 VPN 端口，如 {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    "http": os.getenv("HTTP_PROXY", ""),
    "https": os.getenv("HTTPS_PROXY", "")
}

# ArXiv client configuration
ARXIV_CLIENT = arxiv.Client(delay_seconds=0.05)

# =============================================================================
# RERANKING CONFIGURATION
# =============================================================================
ENABLE_RERANK = False
RERANK_MODEL = DEPLOYMENT_NAME

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
