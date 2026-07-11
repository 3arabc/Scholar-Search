# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

from global_config import *
from instruction import *
import traceback
from local_request_v2 import get_from_llm
from utils import fetch_string
import json
from log import logger
from typing import List, Dict, Any, Optional, Set
import concurrent.futures
from api_web import (
    google_search_arxiv_id,
    get_doc_info_from_semantic_scholar_by_arxivid,
    get_doc_info_from_api,
    search_paper_from_arxiv_by_arxiv_id,
    parallel_search_search_paper_from_arxiv,
    search_paper_via_query_from_openalex,
    search_paper_via_query_from_semantic,
    search_doc_via_url_from_openalex,
    search_from_pubmed,
    fetch_pubmed_json
)
from datetime import datetime, timedelta
import re
import numpy as np
import time

from dataclasses import dataclass
from base_class import SearchResult
from datetime import datetime
from collections import defaultdict

from local_db_v2 import db_path, ArxivDatabase
from rerank import Reranker

def get_info_from_local(id_list):
    already = []
    to_process = []
    if os.path.exists(db_path):
        with ArxivDatabase(db_path) as db:
            for _id in id_list:
                db_info = db.get(_id)
                if db_info is None:
                    to_process.append(_id)
                else:
                    already.append(_id)
        logger.info(f"already num is: {len(already)}, to_process num is :{len(to_process)}")
        return already,to_process
    else:
        return {},id_list


class MultiSearchAgent:
    """Agent for parallel multi-source academic paper search and result aggregation."""

    def __init__(self, max_workers: int = 3, batch_size: int = 10):
        """
        Initialize the multi-search agent.
        Args:
            max_workers: Maximum number of parallel search workers
            batch_size: Size of batches for paper detail retrieval
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.current_date = "2025-03-24"  # 当前日期，参考你的需求

    def extract_keywords(self, query: str, source: str = "semantic") -> List[str]:
        """Extract keywords from a query optimized for a specific source."""
        query = query.lower()
        model_inp = template_extract_keywords_source_aware.format(
            user_query=query, source=source
        )
        for _ in range(1):
            try:
                response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
                pattern = r"\[Start\](.*?)\[End\]"
                match = re.search(pattern, response)
                if match:
                    keywords = match.group(1).strip()
                    logger.info(f"Extracted keywords for {source}: {keywords}")
                    return [kw.strip() for kw in keywords.split(",") if kw.strip()][:KEY_WORDS_NUM]
            except:
                logger.error(f"Failed to extract keywords: {traceback.format_exc()}")
        #return []
        logger.warning(f"Keyword extraction failed, using original query as keyword: {query}") #wsl-72，无延申词可以使用原关键词搜索
        return [query]

    def _google_arxiv_search(
        self,
        queries: List[str],
        end_date: str = "",
        searched_docs: Dict[str, Any] = None,
    ) -> SearchResult:
        """Execute Google Scholar search for a list of queries."""
        merged_papers: Dict[str, Any] = {}
        query2docs: Dict[str, List] = {query: [] for query in queries}

        if searched_docs is None:
            searched_docs = {}

        try:
            # Step 1: 并行搜索 arxiv_ids
            results = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                future_to_query = {
                    executor.submit(
                        google_search_arxiv_id, query, API_TRY_COUNT, 15, end_date
                    ): query
                    for query in queries
                }
                for future in concurrent.futures.as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        results[query] = future.result()
                    except Exception as e:
                        logger.error(
                            f"Google search failed for query {query}: {str(e)}"
                        )
                        results[query] = []

            logger.info(f"google_search_arxiv_id results: {results}")

            # Step 2: 去重 arxiv_ids
            unique_arxiv: Set[str] = set()
            original_arxiv = []
            for arxiv_ids in results.values():
                original_arxiv.extend(arxiv_ids)
                for arxiv_id in arxiv_ids:
                    if arxiv_id not in searched_docs:
                        unique_arxiv.add(arxiv_id)

            logger.info(
                f"Original num: {len(original_arxiv)}, Unique num: {len(unique_arxiv)}"
            )

            # Step 3: 统一获取论文详情
            id2docs: Dict[str, Any] = {}
            if unique_arxiv:
                id2docs = parallel_search_search_paper_from_arxiv(
                    list(unique_arxiv),
                    max_workers=self.max_workers,
                    batch_size=self.batch_size,
                )

            # Step 4: 合并结果
            for query, arxiv_ids in results.items():
                for arxiv_id in arxiv_ids:
                    if arxiv_id in id2docs:
                        paper_info = id2docs[arxiv_id]
                        merged_papers[arxiv_id] = paper_info
                        query2docs[query].append(paper_info)
        except:
            logger.error(f"google search error: {traceback.format_exc()}")

        return SearchResult(
            source="arxiv", papers=merged_papers, query2paper=query2docs
        )

    def _semantic_search(
        self, keyword: str, raw_query: str, end_date: str = "", max_papers: int = 15
    ) -> SearchResult:
        """Execute Semantic Scholar search."""
        #logger.info(f"Searching Semantic Scholar for '{query}'")
        logger.info(f"Searching Semantic Scholar for '{keyword}'")  #wsl
        try:
            papers = search_paper_via_query_from_semantic(
                query=keyword, max_paper_num=max_papers
            )
            logger.info(f"Found {len(papers)} papers for '{keyword}' from Semantic")
            return SearchResult(
                source="semantic", papers=papers, keyword=keyword, raw_query=raw_query
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {traceback.format_exc()}")
            return SearchResult(source="semantic", papers={}, error=str(e))

    def _openalex_search(
        self, keyword: str, raw_query: str, end_date: str = ""
    ) -> SearchResult:
        """Execute OpenAlex search."""
        logger.info(f"Searching OpenAlex for '{keyword}'")
        try:
            papers = search_paper_via_query_from_openalex(keyword, per_page=50)  #wsl-77
            logger.info(f"Found {len(papers)} papers for '{keyword}' from OpenAlex")
            return SearchResult(
                source="openalex", papers=papers, keyword=keyword, raw_query=raw_query
            )
        except Exception as e:
            logger.error(f"OpenAlex search failed: {traceback.format_exc()}")
            return SearchResult(
                source="openalex", papers={}, raw_query=raw_query, error=str(e)
            )

    def _pubmed_search(
        self, keyword: str, raw_query: str, max_results: int = 10
    ) -> SearchResult:
        """Execute PubMed search."""
        logger.info(f"Searching PubMed for '{keyword}'")
        try:
            papers = search_from_pubmed(keyword, max_results=max_results)
            logger.info(f"Found {len(papers)} papers for '{keyword}' from PubMed")
            return SearchResult(
                source="pubmed",
                papers={p["paper_id"]: p for p in papers},
                keyword=keyword,
                raw_query=raw_query,
            )
        except Exception as e:
            logger.error(f"PubMed search failed: {traceback.format_exc()}")
            return SearchResult(
                source="pubmed", papers={}, raw_query=raw_query, error=str(e)
            )

    def _merge_paper_info(
        self, existing: Dict[str, Any], new: Dict[str, Any], source: str
    ) -> Dict[str, Any]:
        """Merge paper information from different sources."""
        merged = existing.copy()

        for field in [
            "abstract",
            "title",
            "publicationYear",
            "authors",
            "fieldsOfStudy",
        ]:
            if field not in merged or not merged[field]:
                merged[field] = new.get(field)

        if "sources" not in merged:
            merged["sources"] = [existing.get("source", "unknown")]
        merged["sources"].append(source)
        merged["sources"] = list(set(merged["sources"]))

        for score_field in ["citationCount", "referenceCount"]:
            if score_field in new:
                if score_field not in merged:
                    merged[score_field] = new[score_field]
                else:
                    merged[score_field] = max(merged[score_field], new[score_field])

        return merged

    def _merge_search_results_grouped(
        self, results: List[SearchResult], source: str
    ) -> SearchResult:
        """Merge multiple search results grouped by raw_query."""
        # Step 1: Group results by raw_query
        grouped_results = defaultdict(list)
        for result in results:
            if result.raw_query:
                grouped_results[result.raw_query].append(result)
            else:
                logger.warning(f"Result without raw_query: {result}")

        # Step 2: Merge results within each group
        merged_search_keywords = {}
        merged_query2paper = {}
        for raw_query, group in grouped_results.items():
            logger.info(f"[{source}]: Merging results for raw_query: {raw_query}")
            for result in group:
                if not result.papers:
                    logger.info(f"No papers found in this result: {result}, skipping")
                    continue

                if raw_query not in merged_query2paper:
                    merged_query2paper[raw_query] = []

                merged_query2paper[raw_query].extend(list(result.papers.values()))

                if raw_query not in merged_search_keywords:
                    merged_search_keywords[raw_query] = []
                merged_search_keywords[raw_query].append(result.keyword)

        return SearchResult(
            source=source,
            papers=[],
            query2paper=merged_query2paper,
            extra={"merged_query_to_keywords": merged_search_keywords},
        )

    def _merge_search_results(
        self, results: List[SearchResult], source: str
    ) -> SearchResult:
        """Merge multiple search results from the same source."""
        merged_papers = {}
        merged_querys = []
        merged_raw_querys = []
        extra = {}

        for result in results:
            if not result.papers:
                logger.info(f"No papers found in this result: {result}, skipping")
                continue
            keyword = result.keyword
            raw_query = result.raw_query
            papers = result.papers
            merged_querys.append(keyword)
            merged_raw_querys.append(raw_query)
            extra[keyword] = len(result.papers)
            for paper_id, paper_info in result.papers.items():
                if paper_id in merged_papers:
                    merged_papers[paper_id] = self._merge_paper_info(
                        merged_papers[paper_id], paper_info, source
                    )
                else:
                    if "source" not in paper_info:
                        paper_info["source"] = source
                    else:
                        paper_info["source"] = f"{paper_info['source']}|{source}"
                    paper_info["sources"] = [source]
                    merged_papers[paper_id] = paper_info

        return SearchResult(
            source=source,
            papers=merged_papers,
            query="|".json(merged_querys),
            extra=extra,
        )

    def search_papers(
        self,
        querys: List[str],
        sources: List[str] = ["arxiv", "semantic", "openalex"],
        end_date: str = "",
        searched_docs: dict = {},
        rerank: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute parallel search across multiple sources with a list of queries.

        Args:
            querys: List of search queries
            sources: List of search sources to use
            end_date: End date filter
            searched_docs: Dictionary of already searched documents to avoid duplicates
            rerank: Whether to rerank results

        Returns:
            Dict containing aggregated results and stats
        """
        if not querys:
            logger.error("Query list is empty")
            return {}
        logger.info(f"Searching with query list: {querys} across sources: {sources}")

        # Validate sources
        search_funcs = {
            "arxiv": self._google_arxiv_search,
            "semantic": self._semantic_search,
            "openalex": self._openalex_search,
            "pubmed": self._pubmed_search,
        }
        valid_sources = []
        for source in sources:
            if source in search_funcs:
                valid_sources.append(source)
            else:
                logger.error(f"Unknown search source: {source}")

        if not valid_sources:
            logger.error("No valid search sources specified")
            return {}

        # Step 1: Extract keywords for each query for each source
        keyword_extraction_sources = {"openalex", "pubmed"}
        query_keywords_by_source = {}  # Maps source -> query -> keywords
        keywords_combine_query = {}
        query_keywords2raw = {}

        if set(valid_sources).intersection(keyword_extraction_sources):
            for source in valid_sources:
                if source in keyword_extraction_sources:
                    query_keywords_by_source[source] = {}
                    keywords_combine_query[source] = {}
                    query_keywords2raw[source] = {}
                    source_keywords_already = []
                    # Process each query separately
                    for query_idx, query in enumerate(querys):
                        # Extract keywords optimized for this specific source and query
                        source_keywords = self.extract_keywords(query, source)
                        if source_keywords:
                            source_keywords_valid = []
                            for one in source_keywords:
                                if one not in source_keywords_already:
                                    source_keywords_already.append(one)
                                    source_keywords_valid.append(one)
                                else:
                                    logger.info(
                                        f"Keyword '{one}' already exists in source keywords for {source}"
                                    )

                            query_keywords_by_source[source][
                                query
                            ] = source_keywords_valid
                            keywords_combine_query[source][query] = "|".join(
                                source_keywords_valid
                            )
                            query_keywords2raw[source][
                                "|".join(source_keywords_valid)
                            ] = query
                            logger.info(
                                f"Query {query_idx+1}: Extracted {len(source_keywords_valid)} keywords for {source}"
                            )
                        else:
                            # Fallback to default keywords if extraction fails
                            query_keywords_by_source[source][query] = [query]
                            keywords_combine_query[source][query] = query
                            query_keywords2raw[source][query] = query
                            logger.warning(
                                f"Query {query_idx+1}: No keywords extracted for {source}, falling back to original query"
                            )

        # Step 2: Prepare and execute all search tasks in parallel
        search_tasks = []
        future_to_source = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Step 1: Submit tasks

            for source in sources:
                if source == "arxiv":
                    # Google: Use the entire query list
                    future = executor.submit(search_funcs[source], querys, end_date, searched_docs)
                    search_tasks.append(future)
                    future_to_source[future] = (source, "google_arxiv_query")

                elif source in keyword_extraction_sources:
                    for query in querys:
                        for keyword in query_keywords_by_source[source][query]:
                            future = executor.submit(search_funcs[source], keyword, query, end_date)
                            search_tasks.append(future)
                            future_to_source[future] = (source, f"{query[:20]}...: {keyword}")
            # Step 2: Collect results with timeout and error logging
            results_by_source = {source: [] for source in valid_sources}

            for future in concurrent.futures.as_completed(search_tasks):  # total timeout for all tasks
                source, query_or_keyword = future_to_source[future]
                try:
                    start_time = time.time()
                    result = future.result(timeout=5)
                    duration = time.time() - start_time
                    results_by_source[source].append(result)
                    logger.info(f"Completed {source} search for '{query_or_keyword}' in {duration:.1f}s")
                except concurrent.futures.TimeoutError:
                    logger.error(f"{source} search timed out for '{query_or_keyword}'")
                except Exception as e:
                    logger.error(f"{source} search failed for '{query_or_keyword}': {traceback.format_exc()}")
                    logger.debug(f"Future state: {future}")

        # Step 4: Merge results for each source
        merged_results = {}
        for source in valid_sources:
            logger.info(
                f"Source is :{source}, results_by_source num is {len(results_by_source[source])}"
            )
            if results_by_source[source]:
                if source == "arxiv":
                    merged_results[source] = results_by_source[source][0]
                else:
                    merged_results[source] = self._merge_search_results_grouped(
                        results_by_source[source], source
                    )
        logger.info(f"merged_results: {merged_results}")

        # Step 5: Merge results from all sources
        final_papers = {}
        final_query2docs = {}
        query_source_map = {}  # Track which sources were used for each query
        query_keywords2raw = {}
        for source, result in merged_results.items():
            if not result.query2paper:
                logger.info(f"result is empty, skip: {result}")
                continue
            if source == "arxiv" and result.query2paper:
                # For Google/ArXiv results which already track query->paper relationships
                for query, papers in result.query2paper.items():
                    # Use the original query without source prefix
                    if query not in final_query2docs:
                        final_query2docs[query] = []
                    # Track which source provided results for this query
                    # Add papers to query results (without adding source to paper object)
                    final_query2docs[query].extend(papers)
                    query_keywords2raw[query] = query
                    query_source_map[query] = source
                    final_papers.update(
                        {
                            paper.get("paper_id", paper.get("arxivId", "")): paper
                            for paper in papers
                        }
                    )

            else:
                for query, papers in result.query2paper.items():
                    logger.info(f"papers: {len(papers)}: {papers[0]}")

                    query_extracted_keywords = result.extra.get(
                        "merged_query_to_keywords", {}
                    ).get(query, [])
                    query_extracted_keywords_str = "|".join(query_extracted_keywords)
                    if query_extracted_keywords_str not in final_query2docs:
                        final_query2docs[query_extracted_keywords_str] = []
                    final_query2docs[query_extracted_keywords_str].extend(papers)
                    query_keywords2raw[query_extracted_keywords_str] = query
                    query_source_map[query_extracted_keywords_str] = source
                    final_papers.update({paper["paper_id"]: paper for paper in papers})

        logger.info(f"All retrieved papers: {len(final_papers)}")
        logger.info(
            f"Retrieved papers details: {[{query:len(final_query2docs[query])} for query in final_query2docs]}"
        )
        logger.info(f"Query source mapping: {query_source_map}")
        logger.info(f"Query keywords2raw: {query_keywords2raw}")

        # Return the query-source mapping along with the results
        return final_query2docs, final_papers, query_source_map, query_keywords2raw

def extract_json(text): #wsl-71
    """尝试从文本中提取合法的 JSON 对象"""
    text = text.strip()
    # 1. 直接解析
    try:
        return json.loads(text)
    except:
        pass
    # 2. 尝试提取 Markdown 代码块 ```json ... ```
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except:
            pass
    # 3. 尝试提取第一个大括号包围的内容
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None

def _generate_query_from_reference(
    user_query, one_doc, searched_queries
) -> Optional[str]:
    """Generate new query from reference"""


    model_inp = template_context_query_generation.format(
        user_query=user_query,
        searched_queries=searched_queries,
        doc_title=one_doc.get("title", ""),
        doc_abstract=one_doc.get("abstract", ""),
        doc_field=one_doc.get("fieldsOfStudy", ""),
    )

    logger.info(f"_generate_query_from_reference model info: {model_inp}")

    for _ in range(LLM_TRY_COUNT):
        try:
            response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
            logger.info(f"response: {response}")
            response = fetch_string(response)
            query_list = extract_json(response)
            if query_list is None:
                logger.warning("Failed to parse JSON, using defaults")
                # 使用默认值或重试
                #query_list = json.loads(response)
                return None
            output = []
            for new_query in query_list:
                if new_query == "":
                    continue
                if new_query not in searched_queries:
                    output.append(new_query)
                else:
                    logger.info(f"{new_query} already exist in {searched_queries}")
            return output
        except:
            logger.error(
                f"Failed to parse response: {response}, will retry {SLEEP_TIME_LLM} seconds...; Error: {traceback.format_exc()}"
            )
            time.sleep(SLEEP_TIME_LLM) #wsl-小bug
    return []


def similarity_code_v4(query, doc, search_time):
    try:
        output = {}
        model_inp = (
            template_sim_between_query_doc_v2_inst.format(
                searchTime=search_time,
                userQuery=query,
                Title=doc["title"],
                Abstract=doc["abstract"],
                Author=(
                    "; ".join([one["name"] for one in doc["authors"]])
                    if doc["authors"] is not None
                    else ""
                ),
                fieldsOfStudy=(
                    ";".join(doc["fieldsOfStudy"])
                    if doc["fieldsOfStudy"] is not None
                    else ""
                ),
                publicationYear=doc["publicationYear"] if doc["publicationYear"] is not None else "",
            )
            + template_sim_between_query_doc_v2_example
        )
        response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
        response = fetch_string(response)
        response_new = extract_json(response.strip())
        if response_new is None:
            logger.warning("Failed to parse JSON, using defaults")
            # 使用默认值或重试
            #response = json.loads(response.strip())
            return None
        else:
            response = response_new
        overall_score = [
            response[key]
            for key in [
                "topic_match",
                "contextual_relevance",
                "depth_completeness",
            ]
        ]
        output["sim_score"] = np.mean(overall_score) / 5.0
        output["sim_info_details"] = response
        return output
    except:
        logger.error(f"similarity_code_v4 error {traceback.format_exc()}")
        return {}

def llm_relevance_score(query: str, doc: Dict) -> float:  #wsl-73 二次筛选
    """
    使用 LLM 对单篇论文进行相关性评分，返回 0-1 之间的分数。
    采用更严格的评价标准，包括：
        - 标题与查询的匹配度
        - 摘要内容是否直接回答/相关
        - 研究领域是否匹配
        - 是否为综述性或技术性文章（根据查询意图）
    """
    prompt = f"""你是一位学术研究助手，需要评估以下论文与用户查询的相关性。

用户查询：{query}

论文信息：
标题：{doc.get('title', '')}
摘要：{doc.get('abstract', '')}
领域：{doc.get('fieldsOfStudy', '')}

请从以下维度对论文进行评分（0-1分，精确到小数点后2位）：
1. 主题匹配度（标题是否直接相关）
2. 内容深度（摘要是否覆盖查询核心问题）
3. 研究领域的一致性
4. 论文类型（如综述、实验、理论）是否符合需求

请仅输出一个分数（如 0.85），不要有任何其他文字。
"""
    for attempt in range(LLM_TRY_COUNT):
        try:
            response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
            # 提取分数（支持多种格式）
            match = re.search(r'(\d+\.\d+|\d+)', response.strip())
            if match:
                score = float(match.group(1))
                # 确保在 0-1 范围内
                return max(0.0, min(1.0, score))
            else:
                logger.warning(f"LLM returned no numeric score: {response}")
        except Exception as e:
            logger.error(f"LLM relevance scoring failed: {e}")
            time.sleep(SLEEP_TIME_LLM)
    # 如果多次失败，返回原始 sim_score（fallback）
    return doc.get('sim_score', 0.0)

def similarity_code_v5(query, doc):
    output = {}
    try:
        model_inp = evaluation_prompt.format(
            title=doc["title"],
            abstract=doc["abstract"],
            user_query=query,
        )
        response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)

        score_match = re.search(r"Score:\s*([0-1]\.\d+|\d+)", response)
        if score_match:
            score = float(score_match.group(1))
            output["sim_score"] = score
            output["sim_info_details"] = response
            return output
    except:
        logger.error(f"similarity_code_v5 error {traceback.format_exc()}")
        return {}


def _calculate_similarity_with_retry(
    query: str,
    search_time: str,
    doc: Dict,
    max_retries: int = LLM_TRY_COUNT,
    timeout: int = 20,
) -> float:
    """计算 query 与文档的相关性，带重试和超时"""
    output = {}
    for attempt in range(max_retries):
        response = None
        try:
            # # v4
            output = similarity_code_v4(query, doc, search_time)
            ## v5
            # output = similarity_code_v5(query,doc)
            if output:
                return output
        except Exception as e:
            logger.error(
                f"Similarity calculation failed for '{doc['title']}', attempt {attempt+1}/{max_retries}, error: {str(e)}, response: {response}"
            )
            time.sleep(timeout)  # 失败后等待 2 秒再重试
    return output  # 返回最低分，防止影响整体流程


class AcademicTreeSearchEngine:

    def __init__(
        self, max_depth=2, max_docs=10, similarity_threshold=0.5
    ):
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.mretrival_processer = MultiSearchAgent()
        self._emd_model = None  # 用于存储懒加载的实例
        self._selector = None  # 用于存储懒加载的实例
        self.max_docs = max_docs  # 用于控制返回数量

    @property
    def emd_model(self):
        if self._emd_model is None:
            from embedding_agent import BGEM3EmbeddingAgent

            self._emd_model = BGEM3EmbeddingAgent()
        return self._emd_model

    @property
    def selector(self):
        if self._selector is None:
            from pasa_agent import Agent as PasaAgent

            selector_path = (
                "/share/project/shixiaofeng/data/model_hub/pasa/pasa-7b-selector"
            )
            self._selector = PasaAgent(selector_path)
        return self._selector

    def expand_query_native(self,query:str):
        judge_info = {"expanded_queries_info":{}}
        for _ in range(WEB_RETRY_NUM):  #wsl-原WEB_TRYNUM
            try:
                # model_inp = template_query_fusion.format(user_query=query)
                model_inp = template_query_fusion_pasa.format(user_query=query)
                logger.debug(f"query correct model_inp: {model_inp}")
                response = get_from_llm(model_inp,
                                        4096,
                                        model_name=LLM_MODEL_NAME)
                response = fetch_string(response)
                logger.info(f"query correct response: {response}")
                response_new = extract_json(response)
                if response_new is None:
                    logger.warning("Failed to parse JSON, using defaults")
                    # 使用默认值或重试
                    #response_new = json.loads(response)
                    return None
                try:
                    judge_info["expanded_queries_info"]["expanded_queries"] = response_new
                    return judge_info
                except:
                    logger.error(
                        f"Query correction failed: {traceback.format_exc()}")
            except:
                logger.error(
                    f"Query correction failed: {traceback.format_exc()}")
        return judge_info



    def expand_query(self, query: str):
        """
        Enhanced query expansion with intent detection and source identification.

        This method:
        1. Analyzes the user's query intent and domain
        2. Identifies appropriate search sources (web, scholar, arxiv, etc.)
        3. Determines if query expansion would improve results
        4. Generates optimized expansions if needed

        Args:
            query: The original user query

        Returns:
            Dictionary containing query analysis and expansions
        """
        logger.info(f"Analyzing query intent: {query}")

        # Initialize response structure
        result = {
            "query_intent": "",
            "domain": "",
            "suitable_sources": [],
            "needs_expansion": False,
            "expansion_reason": "",
            "expanded_queries": [],
        }

        current_year = datetime.now().year
        # Step 1: Analyze query intent, domain and suitable sources
        try:
            intent_analysis = self._analyze_query_intent(query)
            if intent_analysis:
                result.update(intent_analysis)
                logger.info(f"Query intent analysis: {intent_analysis}")
            else:
                logger.warning("Query intent analysis failed, using defaults")
                result["query_intent"] = "general research"
                result["domain"] = "undefined"
                result["suitable_sources"] = ["arxiv"]
        except Exception as e:
            logger.error(f"Error in query intent analysis: {traceback.format_exc()}")
            result["query_intent"] = "general research"
            result["domain"] = "undefined"
            result["suitable_sources"] = ["arxiv", "openalex"]

        # Step 2: Determine if a query needs expansion
        try:
            expansion_analysis = self._evaluate_expansion_need(query, result["domain"])
            if expansion_analysis:
                result["needs_expansion"] = expansion_analysis["needs_expansion"]
                result["expansion_reason"] = expansion_analysis["reason"]
                logger.info(
                    f"Query expansion needed: {result['needs_expansion']}, reason: {result['expansion_reason']}"
                )
            else:
                logger.warning("Expansion analysis failed, defaulting to no expansion")
                result["needs_expansion"] = False
                result["expansion_reason"] = "Analysis failed, keeping original query"
        except Exception as e:
            logger.error(f"Error in expansion analysis: {traceback.format_exc()}")
            result["needs_expansion"] = False
            result["expansion_reason"] = "Analysis error, keeping original query"

        # Step 3: Generate expanded queries if needed
        if result["needs_expansion"]:
            try:
                expanded_queries = self._generate_expanded_queries(
                    query, result["domain"], result["query_intent"]
                )
                result["expanded_queries"] = expanded_queries
                logger.info(f"Generated {len(expanded_queries)} expanded queries")
            except Exception as e:
                logger.error(
                    f"Error generating expanded queries: {traceback.format_exc()}"
                )
                result["expanded_queries"] = []
                result["expansion_reason"] += " (expansion generation failed)"
        #wsl-78 # 确保至少有一个强制短语查询（防止 LLM 生成空列表或失败）
        phrase_query = self._generate_phrase_query(query)
        if phrase_query:
            if not result["expanded_queries"]:
                # 如果没有任何扩展查询，则使用强制短语作为唯一查询
                result["expanded_queries"] = [phrase_query]
                logger.info(f"Fallback: added forced phrase query: {phrase_query}")
            elif phrase_query not in result["expanded_queries"]:
                # 如果已有扩展查询，但强制短语不在其中，则追加
                result["expanded_queries"].append(phrase_query)
                logger.info(f"Added forced phrase query (final check): {phrase_query}")
        return result

    def _analyze_query_intent(self, query: str):
        """
        Analyze the query to determine intent, domain and suitable search sources.

        Args:
            query: The user query

        Returns:
            Dictionary with query intent analysis
        """
        from datetime import datetime

        current_year = datetime.now().year
        previous_year = current_year - 1
        try:
            # Prompt for query intent analysis
            prompt = template_query_intent.format(query=query,current_year=current_year,previous_year=previous_year)

            for attempt in range(LLM_TRY_COUNT):
                try:
                    response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                    response = fetch_string(response)
                    result = extract_json(response)
                    if result is None:
                        logger.warning("Failed to parse JSON, using defaults")
                        # 使用默认值或重试
                        #result = json.loads(response)
                        return None

                    # Validate the response has required fields
                    if all(
                        k in result
                        for k in ["query_intent", "domain", "suitable_sources","source_reason"]
                    ):
                        return result
                    logger.warning(f"Incomplete response from LLM: {result}")
                except Exception as e:
                    logger.warning(
                        f"LLM analysis failed (attempt {attempt+1}): {str(e)}"
                    )
                    time.sleep(SLEEP_TIME_LLM) #wsl-小bug

            logger.error("All attempts to analyze query intent failed")
            return None
        except Exception as e:
            logger.error(f"Error in query intent analysis: {traceback.format_exc()}")
            return None

    def _evaluate_expansion_need(self, query: str, domain: str):
        """
        Evaluate if the query would benefit from expansion.

        Args:
            query: The original query
            domain: The identified domain

        Returns:
            Dictionary with expansion decision and reason
        """
        try:
            # Prompt for evaluating if query needs expansion
            prompt = template_query_expand_judge_opt.format(query=query, domain=domain)

            for attempt in range(LLM_TRY_COUNT):
                try:
                    response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                    response = fetch_string(response)
                    result = extract_json(response)
                    if result is None:
                        logger.warning("Failed to parse JSON, using defaults")
                        # 使用默认值或重试
                        #result = json.loads(response)
                        return None

                    # Validate the response has required fields
                    if "needs_expansion" in result and "reason" in result:
                        return result
                    logger.warning(f"Incomplete response from LLM: {result}")
                except Exception as e:
                    logger.warning(
                        f"LLM expansion evaluation failed (attempt {attempt+1}): {str(e)}"
                    )
                    time.sleep(SLEEP_TIME_LLM)  #wsl

            logger.error("All attempts to evaluate expansion need failed")
            return None
        except Exception as e:
            logger.error(f"Error in expansion evaluation: {traceback.format_exc()}")
            return None

    '''
    #wsl-76 混合扩展策略，综合考虑查询意图、领域复杂度、特殊模式
    def _generate_expanded_queries(self, query: str, domain: str, intent: str):
        """
        根据查询特征和意图，采用混合策略生成扩展查询。
        策略顺序：
        1. 精确标题匹配 → 不扩展，直接返回原始查询。
        2. 综述/综述意图 → 使用 survey 模板。
        3. 复杂领域或明确技术细节 → 使用 domain-aware 模板。
        4. 其他情况 → 使用 PASA 模板（快速通用）。
        5. 所有 LLM 尝试失败 → 返回基于规则的 fallback 查询。
        """
        from datetime import datetime
        current_year = datetime.now().year
        previous_year = current_year - 1

        # ----- 1. 精确标题检测（不扩展）-----
        # 如果查询看起来像一篇论文标题（包含引号、或长度适中且无问号等）
        if self._is_exact_title_query(query):
            logger.info(f"Detected exact paper title, skipping expansion: {query}")
            return [query]

        # ----- 2. 判断是否为综述意图（使用 survey 模板）-----
        # 注意：这里的 _is_survey_focused 已经是您已有的方法，但可能较慢；我们先用关键词快速判断
        if self._is_survey_focused(intent) or self._has_survey_keywords(query):
            logger.info(f"Using survey-focused expansion for: {query}")
            prompt = template_query_fusion_survery_forcus.format(
                user_query=query,
                user_input_N=3,  # 只生成 3 条，避免过多
                current_year=current_year,
                previous_year=previous_year,
            )
            prompt_type = "survey"
        else:
            # ----- 3. 检查领域复杂度（若 domain 非空且非 undefined，使用 domain-aware）-----
            if domain and domain.lower() != "undefined" and self._is_complex_domain(domain):
                logger.info(f"Using domain-aware expansion for query in {domain}")
                prompt = template_domain_aware_query_expansion.format(
                    user_input_N=4,  # 生成 4 条
                    user_query=query,
                    intent=intent,
                    domain=domain,
                    current_year=current_year,
                    previous_year=previous_year,
                )
                prompt_type = "domain"
            else:
                # ----- 4. 默认使用 PASA 模板（快速）-----
                logger.info(f"Using PASA template for query expansion")
                prompt = template_query_fusion_pasa.format(user_query=query)
                prompt_type = "pasa"

        # 重试与解析逻辑（与您原有代码一致，但增加了超时和 continue）
        best_response = None
        best_query_count = 0
        logger.info(f"Expand query prompt for LLM: {prompt}")

        for attempt in range(LLM_TRY_COUNT):
            try:
                response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                response = fetch_string(response)
                logger.info(f"Expanded queries response: {response}")

                parsed_response = extract_json(response)
                if parsed_response is None:
                    logger.warning("Failed to parse JSON, attempting to extract from text")
                    # 尝试从文本中提取 JSON（已有逻辑）
                    continue

                expanded_queries = self._extract_queries_from_response(parsed_response, prompt_type)
                if expanded_queries:
                    # 去重并过滤空
                    expanded_queries = [q.strip() for q in expanded_queries if q.strip()]
                    # 去重，保留顺序
                    seen = set()
                    unique_queries = []
                    for q in expanded_queries:
                        if q not in seen:
                            seen.add(q)
                            unique_queries.append(q)
                    expanded_queries = unique_queries

                    if len(expanded_queries) > 0:
                        best_response = expanded_queries
                        best_query_count = len(expanded_queries)
                        # 如果已经生成了足够的查询（≥3），直接返回
                        if best_query_count >= 3:
                            return best_response

            except Exception as e:
                logger.warning(f"LLM expansion failed (attempt {attempt + 1}): {str(e)}")
                time.sleep(SLEEP_TIME_LLM)
                continue

        # 如果 LLM 扩展失败或结果不理想，使用 fallback
        if best_response and len(best_response) > 0:
            logger.info(f"Using best response from {LLM_TRY_COUNT} attempts: {len(best_response)} queries")
            return best_response

        logger.error("All LLM expansion attempts failed, using rule-based fallback")
        return self._generate_rule_based_expansions(query, domain)

    # ---------- 新增辅助方法 ----------
    def _is_exact_title_query(self, query: str) -> bool:
        """判断查询是否为精确的论文标题（例如包含引号，或者长度在合理范围且无疑问词）"""
        # 如果查询被引号包围
        if (query.startswith('"') and query.endswith('"')) or (query.startswith("'") and query.endswith("'")):
            return True
        # 如果查询长度适中（10~80字符）且不含问号、不含“what”等疑问词
        question_words = {"what", "who", "which", "where", "when", "why", "how", "is", "are", "do", "does", "can",
                          "could", "would"}
        if 10 < len(query) < 80 and not any(query.lower().startswith(w) for w in question_words):
            # 且包含至少一个常见学术词汇（可选）
            return True
        return False

    def _has_survey_keywords(self, query: str) -> bool:
        """检查查询是否包含综述相关关键词"""
        survey_terms = {"survey", "review", "overview", "state-of-the-art", "literature", "comprehensive", "summary"}
        return any(term in query.lower() for term in survey_terms)

    def _generate_rule_based_expansions(self, query: str, domain: str) -> List[str]:
        """当 LLM 失败时，使用简单的规则生成几个变体查询，确保搜索能继续"""
        # 1. 提取关键词（这里可以复用 extract_keywords，但可能需要确保返回多个）
        # 简单做法：按空格拆分，取前几个词
        words = query.split()
        if len(words) <= 3:
            return [query]  # 太短就不扩展
        # 生成几个变体：去掉停用词、添加 "survey" 等
        stopwords = {"a", "an", "the", "of", "for", "on", "at", "to", "in", "with", "without", "by"}
        filtered = [w for w in words if w.lower() not in stopwords]
        if len(filtered) >= 3:
            base = " ".join(filtered[:4])
            return [
                       query,
                       f"survey on {base}",
                       f"recent advances in {base}",
                       f"{base} methods"
                   ][:3]  # 最多3条
        else:
            return [query]
    '''
    #wsl-78
    def _generate_phrase_query(self, query: str) -> str:
        """
        自动从原始查询中提取核心词，生成一个引号包裹的精确短语查询。
        该查询会被强制加入扩展列表，提高精准命中率。
        """
        import re
        # 常见停用词（包含疑问词、通用学术词汇）
        stopwords = {
            "which", "what", "who", "where", "when", "why", "how", "is", "are", "was", "were",
            "do", "does", "did", "can", "could", "would", "should", "may", "might", "must",
            "the", "a", "an", "of", "for", "on", "at", "to", "in", "with", "without", "by",
            "papers", "studies", "research", "work", "works", "contribute", "advancement",
            "provide", "tell", "list", "name", "mention", "about", "that", "through",
            "using", "based", "approach", "method", "technique", "framework", "model",
            "algorithm", "system", "task", "problem", "solution"
        }
        # 分词（保留字母数字和下划线）
        words = re.findall(r'\b[a-zA-Z0-9_\-]+\b', query.lower())
        # 过滤停用词和过短词
        core_words = [w for w in words if w not in stopwords and len(w) > 2]
        if len(core_words) < 2:
            # 如果核心词太少，直接返回原查询并用引号包裹
            return f'"{query.strip()}"'
        # 取前6个核心词作为短语（保留原始顺序）
        phrase = " ".join(core_words[:6])
        return f'"{phrase}"'

    def _generate_expanded_queries(self, query: str, domain: str, intent: str):
        """
        Generate expanded queries based on the original query, domain and intent.

        This method dynamically selects the appropriate query expansion strategy based on:
        1. Query complexity and specificity
        2. Domain characteristics
        3. Research intent (e.g., survey, methodology, application)

        Args:
            query: The original query
            domain: The identified domain
            intent: The query intent

        Returns:
            List of expanded queries
        """
        try:
            from datetime import datetime

            current_year = datetime.now().year
            previous_year = current_year - 1

            # wsl-76------ 强制使用 PASA 模板 -----
            logger.info(f"Using forced PASA template for query expansion")
            prompt = template_query_fusion_pasa.format(user_query=query)
            prompt_type = "pasa"
            '''
            # Determine the appropriate template based on query analysis
            if FUSION_TEMPLATE == "AUTOMATIC" and  self._is_survey_focused(intent):
                # For survey-focused queries, prioritize finding comprehensive reviews
                logger.info(f"Using survey-focused expansion for query: {query}")
                prompt = template_query_fusion_survery_forcus.format(
                    user_query=query,
                    user_input_N=3, #wsl-74
                    current_year=current_year,
                    previous_year=previous_year,
                )
                prompt_type = "survey"
            elif FUSION_TEMPLATE == "AUTOMATIC" and self._is_complex_domain(domain):
                # For queries in complex or specialized domains, use domain-aware expansion
                logger.info(f"Using domain-aware expansion for query in {domain}")
                prompt = template_domain_aware_query_expansion.format(
                    user_input_N=5,
                    user_query=query,
                    intent=intent,
                    domain=domain,
                    current_year=current_year,
                    previous_year=previous_year,
                )
                prompt_type = "domain"
            elif FUSION_TEMPLATE == "PASA":
                # Use PASA template if explicitly configured
                logger.info(f"Using PASA template for query expansion")
                prompt = template_query_fusion_pasa.format(user_query=query)
                prompt_type = "pasa"
            elif FUSION_TEMPLATE == "WITHEXPLAIN":
                # Use withexplain template if explicitly configured
                logger.info(f"Using withexplain for query: {query}")
                prompt = (
                    template_query_fusion_with_score_inst
                    + template_query_fusion_with_score_user.format(
                        user_query=query, user_input_N=5
                    )
                )
                prompt_type = "withexplain"
            else:
                # Default to domain-aware expansion for all other cases
                logger.info(f"Using default domain-aware expansion")
                prompt = template_domain_aware_query_expansion.format(
                    user_input_N=5, user_query=query, intent=intent, domain=domain
                )
                prompt_type = "domain"
            '''

            # Track attempts and keep best result
            best_response = None
            best_query_count = 0

            logger.info(f"Expand query prompt for LLM: {prompt}")
            for attempt in range(LLM_TRY_COUNT):
                try:
                    response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                    response = fetch_string(response)
                    logger.info(f"Expanded queries response: {response}")

                    # Parse the response based on its format
                    try:
                        parsed_response = extract_json(response)
                        if parsed_response is None:
                            logger.warning("Failed to parse JSON, using defaults")
                            # 使用默认值或重试
                            #parsed_response = json.loads(response)
                            return None
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response: {str(e)}")
                        # Attempt to extract JSON from text if standard parsing fails
                        match = re.search(r"\{.*\}", response, re.DOTALL)
                        if match:
                            try:
                                parsed_response = extract_json(match.group(0))
                                if parsed_response is None:
                                    logger.warning("Failed to parse JSON, using defaults")
                                    # 使用默认值或重试
                                    #parsed_response = json.loads(match.group(0))
                                    return None
                            except:
                                logger.warning("Failed to extract JSON from response")
                                continue
                        else:
                            # Try to extract a list if JSON object extraction failed
                            match = re.search(r"\[.*\]", response, re.DOTALL)
                            if match:
                                try:
                                    parsed_response = extract_json(match.group(0))
                                    if parsed_response is None:
                                        logger.warning("Failed to parse JSON, using defaults")
                                        # 使用默认值或重试
                                        #parsed_response = json.loads(match.group(0))
                                        return None
                                except:
                                    logger.warning(
                                        "Failed to extract JSON list from response"
                                    )
                                    continue
                            else:
                                continue

                    # Extract queries with unified format handling
                    expanded_queries = self._extract_queries_from_response(
                        parsed_response, prompt_type
                    )
                    logger.info(f"Extracted queries: {expanded_queries}")

                    if expanded_queries:
                        #wsl-78 ----- 新增：强制添加精确短语查询 -----
                        phrase_query = self._generate_phrase_query(query)
                        if phrase_query not in expanded_queries:
                            expanded_queries.append(phrase_query)
                            logger.info(f"Added forced phrase query: {phrase_query}")
                        # --------------------------------------
                        # Track best response by number of queries
                        if len(expanded_queries) > best_query_count:
                            best_response = expanded_queries
                            best_query_count = len(expanded_queries)

                        # Return immediately if we got a good number of queries
                        if len(expanded_queries) >= 3:
                            return expanded_queries

                except Exception as e:
                    logger.warning(
                        f"LLM expansion failed (attempt {attempt+1}): {str(e)}"
                    )
                    time.sleep(SLEEP_TIME_LLM)
                    continue  #wsl-74 继续下一次尝试

            # If we tried all attempts but still have a valid best response, return it
            if best_response and len(best_response) > 0:
                #wsl-78 添加强制短语查询（如果尚未包含）
                phrase_query = self._generate_phrase_query(query)
                if phrase_query not in best_response:
                    best_response.append(phrase_query)
                    logger.info(f"Added forced phrase query to best_response: {phrase_query}")
                logger.info(
                    f"Using best response from {LLM_TRY_COUNT} attempts: {len(best_response)} queries"
                )
                return best_response

            # Fallback to basic expansion if all attempts fail
            logger.error(
                "All attempts to generate expanded queries failed, using fallback"
            )
            #return self._generate_fallback_queries(query, domain)
            fallback_queries = self._generate_fallback_queries(query, domain)
            phrase_query = self._generate_phrase_query(query)
            if phrase_query not in fallback_queries:
                fallback_queries.append(phrase_query)
                logger.info(f"Added forced phrase query to fallback: {phrase_query}")
            return fallback_queries

        except Exception as e:
            logger.error(f"Error generating expanded queries: {traceback.format_exc()}")
            # Emergency fallback
            return self._generate_emergency_fallback_queries(query)

    def _extract_queries_from_response(self, response, prompt_type):
        """
        Extract queries from different response formats and standardize.

        Args:
            response: The parsed JSON response
            prompt_type: The type of prompt used (survey, domain, pasa, withexplain)

        Returns:
            List of query strings
        """
        expanded_queries = []

        try:
            # Handle different response formats based on prompt type
            if isinstance(response, list):
                # Direct list of strings (PASA format)
                expanded_queries = [q for q in response if isinstance(q, str)]
            elif isinstance(response, dict):
                # Dictionary with expanded_queries field
                if "expanded_queries" in response:
                    queries_data = response["expanded_queries"]
                    if isinstance(queries_data, list):
                        for item in queries_data:
                            if isinstance(item, str):
                                expanded_queries.append(item)
                            elif isinstance(item, dict) and "query" in item:
                                expanded_queries.append(item["query"])
                # Some responses might have a different structure
                elif prompt_type == "withexplain" and "rewritten_queries" in response:
                    for item in response["rewritten_queries"]:
                        if isinstance(item, dict) and "rewritten_query" in item:
                            expanded_queries.append(item["rewritten_query"])

            # Log metadata if available (for monitoring/improvement)
            if isinstance(response, dict):
                if "summary" in response:
                    logger.info(f"Query expansion summary: {response['summary']}")
                if "domain_keywords" in response:
                    logger.info(f"Domain keywords: {response['domain_keywords']}")

            return expanded_queries
        except Exception as e:
            logger.error(f"Error extracting queries from response: {str(e)}")
            return []

    def _generate_fallback_queries(self, query, domain):
        """
        Generate fallback queries when regular expansion fails.

        Args:
            query: The original query
            domain: The identified domain

        Returns:
            List of fallback queries
        """
        logger.info(f"Using fallback query expansion for: {query}")
        return [
            f"survey papers on {query}",
            f"literature review {query}",
            f"state-of-the-art {query}",
            f"recent advances in {query}",
            f"{domain} {query} methodologies",
        ]

    def _generate_emergency_fallback_queries(self, query):
        """
        Generate emergency fallback queries when all else fails.

        Args:
            query: The original query

        Returns:
            List of minimal fallback queries
        """
        logger.info(f"Using emergency fallback expansion for: {query}")
        return [
            f"survey papers on {query}",
            f"literature review {query}",
            f"state-of-the-art {query}",
        ]

    def _is_survey_focused(self, intent: str) -> bool:
        """
        Determine if the query intent is focused on finding survey or review papers.
        Uses fast keyword matching first, then falls back to model-based detection if needed.

        Args:
            intent: The query intent string

        Returns:
            Boolean indicating if the intent is survey-focused
        """
        intent_lower = intent.lower()

        # Fast path: Check for explicit survey indicators
        survey_indicators = [
            "survey",
            "review",
            "overview",
            "state-of-the-art",
            "literature",
            "comprehensive",
            "summary",
            "taxonomy",
            "comparative",
            "meta-analysis",
        ]

        if any(indicator in intent_lower for indicator in survey_indicators):
            logger.info(f"Detected survey intent via keywords in: {intent}")
            return True

        # Fast path: Check for implicit survey patterns
        implicit_patterns = [
            r"what (is|are) the current",
            r"(summarize|summarizing) (recent|current)",
            r"broad (understanding|overview)",
            r"comprehensive (analysis|study)",
            r"(existing|available) (approaches|methods)",
            r"compare (different|various)",
            r"trends in",
        ]

        if any(re.search(pattern, intent_lower) for pattern in implicit_patterns):
            logger.info(f"Detected survey intent via patterns in: {intent}")
            return True

        # Medium path: Check for contextual clues
        contextual_indicators = [
            # Academic literature orientation
            ("literature", "field"),
            ("papers", "compare"),
            ("research", "directions"),
            ("developments", "field"),
            # Broad scope indicators
            ("comprehensive", "understanding"),
            ("overview", "approaches"),
            ("different", "techniques"),
            # Historical/evolutionary interest
            ("evolution", "development"),
            ("progress", "area"),
            ("history", "development"),
        ]

        if any(
            all(term in intent_lower for term in pair) for pair in contextual_indicators
        ):
            logger.info(f"Detected survey intent via contextual pairs in: {intent}")
            return True

        # Slow path: Use model for uncertain cases
        intent_cache_key = f"survey_intent:{intent_lower}"

        # Check if we have a cached result
        if hasattr(self, "_survey_intent_cache"):
            if intent_cache_key in self._survey_intent_cache:
                return self._survey_intent_cache[intent_cache_key]
        else:
            # Initialize cache if it doesn't exist
            self._survey_intent_cache = {}

        # Only call LLM for intents we're unsure about
        try:
            prompt = f"""Determine if this academic research intent is primarily focused on finding SURVEY or REVIEW papers rather than primary research:

Research intent: "{intent}"

A survey/review-focused intent typically seeks:
1. Comprehensive overviews of a research area
2. Comparisons of different approaches or methodologies
3. Summaries of the state-of-the-art
4. Historical development or evolution of concepts
5. Taxonomies or categorizations of approaches

Respond with only "Yes" if the intent is primarily seeking survey/review papers, or "No" if it's seeking specific primary research papers."""

            response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
            is_survey = "yes" in response.lower()

            # Cache the result for future use
            self._survey_intent_cache[intent_cache_key] = is_survey
            logger.info(f"Model determined survey intent as {is_survey} for: {intent}")
            return is_survey
        except Exception as e:
            # Fall back to more conservative check on error
            logger.error(f"Error determining survey intent: {str(e)}")
            return "overview" in intent_lower or "review" in intent_lower

    def _is_complex_domain(self, domain: str) -> bool:
        """
        Determine if the domain is specialized using both rules and model-based assessment.

        Args:
            domain: The domain string

        Returns:
            Boolean indicating if the domain is complex/specialized
        """
        domain_lower = domain.lower()

        # Fast path: Check against known complex domains first
        complex_domains = {
            "quantum computing",
            "genomics",
            "bioinformatics",
            "neuroscience",
            "computational linguistics",
            "cryptography",
            "nanomaterials",
            "immunology",
            "pharmacology",
            "astrophysics",
            "high energy physics",
            "theoretical computer science",
            "robotics",
            "material science",
        }

        # If it's in our known list, return immediately
        if any(complex_domain in domain_lower for complex_domain in complex_domains):
            return True

        # Fast path: Technical term indicators
        technical_indicators = [
            "quantum",
            "computational",
            "theoretical",
            "stochastic",
            "bayesian",
        ]

        if any(indicator in domain_lower for indicator in technical_indicators):
            return True

        # Fast path: Check multi-word domains that typically indicate complexity
        if len(domain_lower.split()) >= 3:
            return True

        # Slow path: Use the model for uncertain cases, but cache results
        domain_cache_key = f"domain_complexity:{domain_lower}"

        # Check if we have a cached result
        if hasattr(self, "_domain_complexity_cache"):
            if domain_cache_key in self._domain_complexity_cache:
                return self._domain_complexity_cache[domain_cache_key]
        else:
            # Initialize cache if it doesn't exist
            self._domain_complexity_cache = {}

        # For domains we're uncertain about, use LLM to assess complexity
        try:
            prompt = template_query_domain_complex.format(domain=domain)
            response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
            is_complex = "yes" in response.lower()

            # Cache the result for future use
            self._domain_complexity_cache[domain_cache_key] = is_complex
            return is_complex
        except Exception as e:
            # Fall back to heuristic on error
            logger.error(f"Error determining domain complexity: {str(e)}")
            return len(domain_lower.split()) >= 2  # More conservative fallback

    def search_papers_mroute(
        self, queries, end_date="", searched_docs=dict(), sources=["google"]
    ):
        # sources = ["google", "openalex"]
        output, id2docs, query_source_map, query_keywords2raw = (
            self.mretrival_processer.search_papers(
                querys=queries,
                end_date=end_date,
                searched_docs=searched_docs,
                sources=sources,
            )
        )
        #wsl-77 ===== 新增：对每个查询的结果进行 BGE 重排序 =====
        reranked_output = {}
        for query, docs in output.items():
            if docs:
                # 调用 rerank_score_bge，返回排序后的文档列表
                sorted_docs = self.rerank_score_bge(query, docs)
                # 取前 max_docs 个
                reranked_output[query] = sorted_docs[:self.max_docs]
            else:
                reranked_output[query] = []

        # 重新构建 id2docs（使用重排序后的文档）
        final_id2docs = {}
        for docs in reranked_output.values():
            for doc in docs:
                doc_id = doc.get("paper_id", doc.get("arxivId", ""))
                if doc_id:
                    final_id2docs[doc_id] = doc

        return output, id2docs, query_source_map, query_keywords2raw

    def search_papers(self, queries, end_date="", searched_docs=dict()):
        results = {}
        if end_date == "":
            end_date = self.current_date

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=API_PARALLEL_REQUEST #wsl
        ) as executor:
            future_to_query = {
                executor.submit(
                    google_search_arxiv_id, query, API_TRY_COUNT, 10, end_date
                ): query
                for query in queries
            }
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results[query] = future.result()
                except Exception as e:
                    logger.error(f"Search failed for query {query}: {str(e)}")
                    results[query] = []

        logger.info(f"google_search_arxiv_id: {results}")

        unique_arxiv = set()
        original_arxiv = []
        for arxiv_ids in results.values():
            original_arxiv.extend(arxiv_ids)
            for arxiv_id in arxiv_ids:
                if arxiv_id not in searched_docs:
                    unique_arxiv.add(arxiv_id)

        logger.info(
            f"original num is: {len(original_arxiv)}, unique num is: {len(list(unique_arxiv))}"
        )

        id2docs = parallel_search_search_paper_from_arxiv(
            list(unique_arxiv), max_workers=API_PARALLEL_REQUEST, batch_size=8
        )

        output = {}
        for query, arxiv_ids in results.items():
            output[query] = [
                id2docs[arxiv_id] for arxiv_id in arxiv_ids if arxiv_id in id2docs
            ]
        return output, id2docs

    def calculate_sim_bge(
        self, query, docs, search_time="", score_thresh=BEGIN_SIM_THRESHOLD, source=""
    ):
        logger.info("calculate_sim_bge ...")
        relevace_docs = []
        irrelevace_docs = []
        '''
        golden_paper_info = [
            "Title:{}\nAbstract:{}Authors:{}".format(
                doc.get("title", ""),
                doc.get("abstract", ""),
                ";".join([one["name"] for one in doc.get("authors", [])]),
            )
            for doc in docs
        ]
        '''
        golden_paper_info = [
            "Title:{}\nAbstract:{}".format(
                doc.get("title", ""),
                doc.get("abstract", ""),
            )
            for doc in docs
        ]
        score_info_list = self.emd_model.get_score(
            query, golden_paper_info, batch_size=6
        )
        for doc, sim_score in zip(docs, score_info_list):
            # 如果都没有，设为空列表
            # 注意：不能设为 None，因为过滤逻辑可能检查
            #wsl-710 从原始文档中提取年份和引用数（若存在）
            year = doc.get("publicationYear") or doc.get("year")
            citations = doc.get("citationCount") or doc.get("citations")
            # 提取领域（OpenAlex 通常有 fieldsOfStudy）
            fields = doc.get("fieldsOfStudy") or doc.get("concepts")
            simple_info = {
                "arxivId": doc["arxivId"],
                "paper_id": doc.get("paper_id", doc.get("arxivId")),
                "sim_score": sim_score,
                "source": source,
                #"sim_info_details": {
                #    "reason": "calculate sim from beg-m3",
                #    "sim_score": sim_score,
                #},
                # 新增字段
                "publicationYear": year if year is not None else None,
                "citationCount": citations if citations is not None else 0,
                "fieldsOfStudy": fields if fields else [],  # 保持列表形式，缺失则为空列表
            }
            if sim_score >= score_thresh:
                relevace_docs.append(simple_info)
            else:
                irrelevace_docs.append(simple_info)

        return relevace_docs, irrelevace_docs

    def calculate_sim_pasa(
        self, query, docs, search_time="", score_thresh=PASS_SIM_THRESHOLD, source=""
    ):
        logger.info("calculate_sim_pasa..")
        relevace_docs = []
        irrelevace_docs = []

        prompt_template = (
            "You are an elite researcher in the field of AI, conducting research on {user_query}. "
            "Evaluate whether the following paper fully satisfies the detailed requirements of the user query "
            "and provide your reasoning. Ensure that your decision and reasoning are consistent.\n\n"
            "Searched Paper:\nTitle: {title}\nAbstract: {abstract}\n\n"
            "User Query: {user_query}\n\n"
            "Output format: Decision: True/False\nReason:... \nDecision:"
        )

        # 对doc进行过滤，如果存在字段缺失，那么这个数据丢弃
        docs = [doc for doc in docs if doc.get("title", "") and doc.get("abstract", "")]

        golden_paper_info = [
            prompt_template.format(
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                user_query=query,
            )
            for paper in docs
        ]

        score_info_list = self.selector.batch_infer_score(golden_paper_info, 4)
        for doc, sim_score in zip(docs, score_info_list):
            simple_info = {
                "arxivId": doc.get("arxivId",""),
                "paper_id": doc.get("paper_id", doc.get("arxivId")),
                "sim_score": sim_score,
                "source": source,
                "sim_info_details": {
                    "reason": "calculate sim from pasa-scorer",
                    "sim_score": sim_score,
                },
            }
            if sim_score >= score_thresh:
                relevace_docs.append(simple_info)
            else:
                irrelevace_docs.append(simple_info)

        return relevace_docs, irrelevace_docs
    '''
    def rerank_score_bge(self, query, docs):
        logger.info("rerank_score_bge ...")

        golden_paper_info = [
            "Title:{}\nAbstract:{}Authors:{}".format(
                doc.get("title", ""),
                doc.get("abstract", ""),
                ";".join([one["name"] for one in doc.get("authors", [])]),
            )
            for doc in docs
        ]
        score_info_list = self.emd_model.get_score(
            query, golden_paper_info, batch_size=12
        )

        assert len(score_info_list) == len(golden_paper_info)
        paired_docs_scores = list(zip(docs, score_info_list))
        # 根据分数从高到低排序
        paired_docs_scores.sort(key=lambda x: x[1], reverse=True)

        scorted_docs = []
        for doc, score in paired_docs_scores:
            # logger.info(f"Score: {score}, Title: {doc.get('title', 'N/A')}")
            doc["rerank_score_bge"] = score
            scorted_docs.append(doc)
        #wsl-77 提取查询中的潜在专有名词（可根据需求自定义）
        core_terms = [w for w in query.split() if len(w) > 3 and w.lower() not in ['model', 'based', 'learning']]
        for doc in scorted_docs:
            title = doc.get('title', '').lower()
            bonus = 0.0
            for term in core_terms:
                if term.lower() in title:
                    bonus += 0.1
            doc['rerank_score_bge'] += bonus
        # 重新按新分数排序
        scorted_docs.sort(key=lambda x: x['rerank_score_bge'], reverse=True)

        return scorted_docs
    '''
    #wsl-77在 BGE 向量相似度重排序的基础上，对标题中包含查询核心术语（如模型名、专有名词）的论文给予额外加分
    def rerank_score_bge(self, query, docs):
        logger.info("rerank_score_bge ...")

        # ----- 步骤1: 从查询中提取核心术语（保留专有名词、模型名等） -----
        import re
        # 定义通用停用词和泛词（避免干扰）
        stopwords = {
            'the', 'a', 'an', 'of', 'for', 'on', 'at', 'to', 'in', 'with', 'without',
            'by', 'and', 'or', 'but', 'from', 'up', 'about', 'into', 'through', 'during',
            'including', 'etc', 'papers', 'studies', 'work', 'research', 'contribute',
            'advancement', 'which', 'what', 'how', 'why', 'when', 'where', 'can', 'could',
            'would', 'should', 'might', 'may', 'does', 'do', 'is', 'are', 'was', 'were',
            'has', 'have', 'been', 'being', 'will', 'shall', 'need', 'using', 'based'
        }
        # 分词并过滤
        tokens = re.findall(r'\b[a-zA-Z0-9_\-]+\b', query.lower())
        core_terms = set([t for t in tokens if t not in stopwords and len(t) > 2])

        # 如果查询中有引号内容（如 "Dream to Control"），优先保留
        quoted = re.findall(r'"([^"]+)"', query)
        for q in quoted:
            core_terms.update(q.lower().split())

        # 额外保留包含大写字母的术语（可能是缩写或专有名词），因为原始查询可能没有引号
        # 这里简单处理：如果原始查询中有大写单词，加入
        uppercase_terms = re.findall(r'\b([A-Z][A-Za-z0-9_\-]*)\b', query)
        core_terms.update([t.lower() for t in uppercase_terms if len(t) > 1])

        logger.info(f"Core terms for title bonus: {core_terms}")

        # ----- 构建增强的文档表示（标题重复，摘要保留）-----
        golden_paper_info = []
        for doc in docs:
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            authors = ";".join([a.get("name", "") for a in doc.get("authors", [])])
            # 将标题重复两遍，并加上明确字段标签
            enhanced_text = f"Title: {title}\nTitle: {title}\nAbstract: {abstract}\nAuthors: {authors}"
            golden_paper_info.append(enhanced_text)

        # ----- 计算 BGE 相似度 -----
        score_info_list = self.emd_model.get_score(query, golden_paper_info, batch_size=12)
        assert len(score_info_list) == len(docs)

        # ----- 合并分数并添加标题匹配加分（加大权重）-----
        scored_docs = []
        for doc, score in zip(docs, score_info_list):
            title = doc.get('title', '').lower()
            bonus = 0.0
            # 对每个核心术语，出现在标题中加 0.15（原0.1），上限0.5
            for term in core_terms:
                if term in title:
                    bonus += 0.15
            bonus = min(bonus, 0.5)
            # 如果摘要也包含核心词，额外加 0.05（但避免过度）
            abstract = doc.get('abstract', '').lower()
            for term in core_terms:
                if term in abstract:
                    bonus += 0.05
            bonus = min(bonus, 0.6)  # 总加分不超过0.6
            total = score + bonus
            doc['rerank_score_bge'] = total
            doc['title_match_bonus'] = bonus
            scored_docs.append(doc)

        # 按总分降序排序
        scored_docs.sort(key=lambda x: x.get('rerank_score_bge', 0), reverse=True)
        return scored_docs

    def calculate_similarity(  #wsl-74相似度计算改为bge
            self, query, docs, search_time="", score_thresh=0.5, source=""
    ):
        """
        使用 BGE 向量相似度替代 LLM 打分，大幅提升速度。
        """
        logger.info(f"calculate_similarity (BGE) for {len(docs)} docs, query: {query[:50]}...")
        # 直接调用已有的 BGE 方法，保持参数兼容
        return self.calculate_sim_bge(
            query=query,
            docs=docs,
            search_time=search_time,  # calculate_sim_bge 也有该参数，但实际未使用
            score_thresh=score_thresh,
            source=source
        )
    '''def calculate_similarity(
        self, query, docs, search_time="", score_thresh=0.5, source=""
    ):
        logger.debug(f"calculate_similarity, query: {query}; doc is: {docs[0].keys()}")
        relevace_docs = []
        irrelevace_docs = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=LLM_PARREL_NUM
        ) as executor:
            future_to_doc = {
                executor.submit(
                    _calculate_similarity_with_retry, query, search_time, doc
                ): doc
                for doc in docs
            }
            for future in concurrent.futures.as_completed(future_to_doc):
                doc = future_to_doc[future]
                res = future.result(timeout=2)
                try:
                    if res:
                        doc.update(res)
                    else:
                        print(traceback.format_exc())
                        doc["sim_score"] = -1  # 失败则设为0，这个数据就不要了
                        doc["sim_info_details"] = {}
                except:
                    print(traceback.format_exc())
                    doc["sim_score"] = -1  # 失败则设为0，这个数据就不要了
                    doc["sim_info_details"] = {}
                finally:
                    simple_info = {
                        "arxivId": doc["arxivId"],
                        "paper_id": doc.get("paper_id", doc.get("arxivId")),
                        "sim_score": doc["sim_score"],
                        "sim_info_details": doc["sim_info_details"],
                        "source": source,
                    }
                    if doc["sim_score"] >= score_thresh:
                        relevace_docs.append(simple_info)
                    else:
                        irrelevace_docs.append(simple_info)

        return relevace_docs, irrelevace_docs
        '''

    def get_doc_references(self, doc_info):
        try:
            if "arxivId" in doc_info:
                if not doc_info.get("arxivId", ""):
                    return doc_info
                doc_info_new = get_doc_info_from_semantic_scholar_by_arxivid(
                    doc_info["arxivId"]
                )
                if doc_info_new is not None:
                    doc_info.update(doc_info_new)
                    return doc_info

            elif "PMID" in doc_info:  #wsl
                # current doc has references, but the info is simple, get full info
                logger.info(f"source is pumbed, {doc_info.get('PMID', '')}")
                valid_pmid = [one["pmid"] for one in doc_info.get("references", [])]
                already_info,valid_pmid = get_info_from_local(valid_pmid)
                pmid_info_lst = fetch_pubmed_json(valid_pmid)
                doc_info["references"] = already_info+pmid_info_lst

            elif "referenceWorksOpenAlex" in doc_info:
                references = search_doc_via_url_from_openalex(
                    doc_info["referenceWorksOpenAlex"]
                )
                doc_info["references"] = references
                return doc_info


        except Exception as e:
            logger.error(
                f"Failed to get references for {doc_info}: {traceback.format_exc()}"
            )
        return doc_info

    def generate_queries_from_docs(self, query, docs, searched_queries):
        results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=LLM_PARREL_NUM
        ) as executor:
            future_to_citation = {
                executor.submit(
                    _generate_query_from_reference,
                    query,
                    ref_doc,
                    searched_queries,
                ): [ref_doc, node]
                for ref_doc, node in docs
            }

            for future in concurrent.futures.as_completed(future_to_citation):
                ref_doc, node = future_to_citation[future]
                res = future.result(timeout=2)
                if res:
                    for new_q in res:
                        results.append([new_q, node])
        return results


# msearch_agent = MultiSearchAgent()
# query = [
#     "Provide me with some top-tier journal papers to expand my ideas on using synthetic data to augment supervised fine-tuning (SFT) while ensuring data quality and diversity, maintaining a balance between the two."
# ]
# res = msearch_agent.search_papers(query)
# print(res[0])
# print(res[1])
