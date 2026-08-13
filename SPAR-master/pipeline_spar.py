# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

from collections import deque
from datetime import datetime, timedelta
from global_config import *
from graphviz import Digraph
from instruction import *
from local_db_v2 import db_path, ArxivDatabase
from log import logger
from search_engine import AcademicTreeSearchEngine,llm_relevance_score
from search_node import SearchNode
from typing import List, Dict, Optional
import json
import re
import time
import tqdm
import traceback
from rerank import Reranker


class AcademicSearchTree:
    """
    Tree-based academic paper search engine that:
    1. Performs iterative search using query expansion
    2. Filters results by relevance score
    3. Explores paper references for deeper search

    The search uses a tree structure where:
    - Root node represents the initial query
    - First level nodes are expanded queries
    - Subsequent levels are queries generated from document context

    The search process involves:
    1. Query expansion: Generate alternative formulations of the initial query
    2. Document retrieval: Search for papers matching each query
    3. Relevance calculation: Score papers based on relevance to the initial query
    4. Reference exploration: Find additional papers by following citations
    5. Query generation: Create new queries based on retrieved documents

    Search stops when either:
    - Enough highly relevant papers are found (> max_docs)
    - Maximum search depth is reached
    """

    def __init__(
        self,
        max_depth: int = 1,
        max_docs: int = 200,
        similarity_threshold: float = 0.6,
        search_engine=None,
        enable_llm_rerank=True,  # wsl-73 二次筛选
        llm_threshold=0.7,
        filter_config=None
    ):
        # Search parameters
        self.max_depth = max_depth
        self.max_docs = max_docs
        self.sim_threshold = similarity_threshold
        # Search state
        self.root = SearchNode()
        # Current search metadata
        self.search_time = None
        self.current_date = None
        self.high_score_thresh = 0.75
        self.search_engine = AcademicTreeSearchEngine()
        self.reranker = Reranker()
        self.enable_llm_rerank = enable_llm_rerank  # wsl-73 二次筛选
        self.llm_threshold = llm_threshold
        self.filter_config = filter_config or {}

    def _cleanup_resources(self):
        """Perform cleanup of resources after search is completed"""
        try:
            # Release any resources that need explicit cleanup
            if (
                hasattr(self.search_engine, "_emd_model")
                and self.search_engine._emd_model
            ):
                # If there's a cleanup method available, call it
                if hasattr(self.search_engine._emd_model, "cleanup"):
                    self.search_engine._emd_model.cleanup()

            # Clear large in-memory data structures
            if hasattr(self.root, "cal_sim_docs"):
                self.root.cal_sim_docs.clear()

            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    #wsl-710 过滤
    def _filter_docs(self, docs_dict: dict, filter_config: dict = None) -> dict:
        """
        根据传入的 filter_config 过滤文档，如果 filter_config 为 None，则从 global_config 读取。
        根据全局配置过滤文档：
        - 年份范围（FILTER_YEAR_START ~ FILTER_YEAR_END）
        - 引用数（>= FILTER_MIN_CITATIONS）
        - 研究领域（包含 FILTER_FIELDS 中任意一个）
        返回过滤后的字典（仅保留符合条件的文档）。
        """
        logger.info(f"_filter_docs received filter_config: {filter_config}")
        if not docs_dict:
            return {}

        if filter_config is None:
            from global_config import (
                FILTER_YEAR_START, FILTER_YEAR_END,
                FILTER_MIN_CITATIONS, FILTER_FIELDS,
                FILTER_ENABLE_YEAR, FILTER_ENABLE_CITATIONS, FILTER_ENABLE_FIELDS,
                FILTER_MISSING_FIELD_PASS
            )
        else:
            # 从传入的 dict 中提取参数
            FILTER_YEAR_START = filter_config.get('year_start')
            FILTER_YEAR_END = filter_config.get('year_end')
            FILTER_MIN_CITATIONS = filter_config.get('min_citations')
            FILTER_FIELDS = filter_config.get('fields', [])
            FILTER_MISSING_FIELD_PASS = filter_config.get('missing_field_pass', True)

        filtered = {}
        for doc_id, doc in docs_dict.items():

            # 默认通过
            passed = True

            # 1. 年份过滤（如果启用）
            if FILTER_YEAR_START is not None or FILTER_YEAR_END is not None:
                year = doc.get("publicationYear") or doc.get("year")
                if year is not None:
                    try:
                        year = int(year)
                        if FILTER_YEAR_START is not None and year < FILTER_YEAR_START:
                            passed = False
                        if FILTER_YEAR_END is not None and year > FILTER_YEAR_END:
                            passed = False
                    except:
                        # 解析失败，按缺失处理
                        if not FILTER_MISSING_FIELD_PASS:
                            passed = False
                else:
                    # 字段缺失
                    if not FILTER_MISSING_FIELD_PASS:
                        passed = False

            # 2. 引用数过滤（如果启用）
            if FILTER_MIN_CITATIONS is not None and passed:
                citations = doc.get("citationCount") or doc.get("citations")
                if citations is not None:
                    try:
                        citations = int(citations)
                        if FILTER_MIN_CITATIONS is not None and citations < FILTER_MIN_CITATIONS:
                            passed = False
                    except:
                        if not FILTER_MISSING_FIELD_PASS:
                            passed = False
                else:
                    if not FILTER_MISSING_FIELD_PASS:
                        passed = False

            # 3. 领域过滤（如果启用且 FILTER_FIELDS 非空）
            if  FILTER_FIELDS and passed:
                fields = doc.get("fieldsOfStudy") or doc.get("concepts")
                if fields:
                    # 如果是列表，检查是否至少有一个匹配
                    if isinstance(fields, list):
                        matched = False
                        for field in fields:
                            if isinstance(field, dict):
                                field_name = field.get("name") or field.get("display_name") or ""
                            else:
                                field_name = str(field)
                            if any(f.lower() in field_name.lower() for f in FILTER_FIELDS):
                                matched = True
                                break
                        if not matched:
                            passed = False
                    else:
                        # 如果是字符串，直接匹配
                        field_str = str(fields)
                        if not any(f.lower() in field_str.lower() for f in FILTER_FIELDS):
                            passed = False
                else:
                    # 字段缺失
                    if not FILTER_MISSING_FIELD_PASS:
                        passed = False

            # 如果所有条件都通过，保留文档
            if passed:
                filtered[doc_id] = doc

        logger.info(f"Filtering: kept {len(filtered)} out of {len(docs_dict)} documents")
        return filtered

    def meet_stop_condition(self, current_depth=0):
        """
        Determines if the search should stop based on relevance threshold and depth.

        The search stops when either:
        1. We've found enough highly relevant documents (> self.max_docs)
        2. We've reached the maximum search depth

        Args:
            current_depth: Current search depth

        Returns:
            Boolean indicating whether to stop the search
        """
        # Identify highly relevant documents
        relevance_docs = set(
            doc_id
            for doc_id, doc_info in self.root.searched_docs.items()
            if doc_info.get("sim_score", -1) > self.high_score_thresh
        )
        self.root.hight_relevance_docs = relevance_docs

        # Log current search state
        logger.info(
            f"Query: {self.user_query[:30]}... | "
            f"Depth: {current_depth}/{self.max_depth} | "
            f"Total docs: {len(self.root.searched_docs)} | "
            f"Highly relevant: {len(relevance_docs)}/{self.max_docs} | "
            f"Normal relevance: {len(self.root.docs)}"
        )

        # Check stopping conditions
        depth_exceeded = current_depth >= self.max_depth
        enough_relevant_docs = len(relevance_docs) > self.max_docs

        if enough_relevant_docs:
            logger.info(
                f"Stopping search: Found {len(relevance_docs)} highly relevant docs (threshold: {self.max_docs})"
            )
        elif depth_exceeded:
            logger.info(f"Stopping search: Reached maximum depth {self.max_depth}")

        # return enough_relevant_docs or depth_exceeded
        return depth_exceeded

    def _save_id2info(self, id2docs):
        """
        Save document information to local database with error handling.

        Args:
            id2docs: Dictionary mapping document IDs to document information
        """
        if not id2docs:
            logger.info("No documents to save to local DB")
            return

        logger.info(f" Saving {len(id2docs)} documents to local database")
        start_time = time.time()
        success_count = 0

        try:
            with ArxivDatabase(db_path) as db:
                for arxiv, info in id2docs.items():
                    if info.get("source", "") == "Search From Local":
                        continue
                    try:
                        # Create new info dict, removing keys containing "sim_score"
                        cleaned_info = {
                            k: v for k, v in info.items() if "sim_score" not in k
                        }
                        db.update_or_insert(arxiv, cleaned_info)
                        success_count += 1
                    except Exception as e:
                        logger.error(f"Failed to save document {arxiv}: {str(e)}")

            logger.info(
                f" Saved {success_count}/{len(id2docs)} documents to local DB in {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            logger.error(f"Database operation failed: {traceback.format_exc()}")

    def query_fusion(self):
        """
        Expands the initial query into multiple search queries.
        Returns a list of expanded queries without creating search nodes.
        """
        logger.info("Running step1: query expansion")
        try:
            query_node_relations = {}
            expanded_queries_info = self.search_engine.expand_query(self.root.query_str)
            # logger.info(f"expanded_queries_info: {expanded_queries_info}")
            expanded_queries_info["QUERY_NUM_PRUNED"] = QUERY_NUM_PRUNED
            self.root.extra["expanded_queries_info"] = expanded_queries_info
            expanded_queries = expanded_queries_info["expanded_queries"]

            # Add the original query to the expanded list
            if self.root.query_str not in expanded_queries:
                expanded_queries = [self.root.query_str] + expanded_queries

            for query in expanded_queries:
                node = SearchNode(
                    query_str=query,
                    status="START",
                )
                query_node_relations[query] = {
                    "own_node": node,
                    "parent_node": self.root,
                }

            return expanded_queries, query_node_relations
        except Exception as e:
            logger.error(f"Query fusion failed: {traceback.format_exc()}")
            # Fallback to just the original query if expansion fails
            return [self.root.query_str], {}

    def query_level_search(
        self,
        expanded_queries,
        query_node_relations,
        next_level,
        search_date,
        current_depth,
        forced_keywords: list = None,
    ):
        """
        Performs search for all queries at the current level and processes the results.
        Creates search nodes based on the query results.

        Args:
            expanded_queries: List of expanded query strings
            next_level: List to populate with nodes for the next level
            search_date: End date for search
            current_depth: Current search depth

        Returns:
            Tuple of (created search nodes, next level nodes)
        """
        if not expanded_queries:
            logger.warning("No queries provided for query_level_search")
            return [], next_level

        logger.info(
            f"Running query_level_search with {len(expanded_queries)} queries at depth {current_depth}"
        )
        # logger.info(f"expanded_queries: {expanded_queries}")
        # logger.info(f"query_node_relations: {query_node_relations}")

        # Determine sources based on expanded_queries_info if available
        if hasattr(self.root, "extra") and "expanded_queries_info" in self.root.extra:
            expanded_info = self.root.extra["expanded_queries_info"]
            if "suitable_sources" in expanded_info:
                sources = expanded_info["suitable_sources"]
                logger.info(f"Using suitable sources from query expansion: {sources}")
            else:
                sources = SEARCH_ROUTES
                logger.info(
                    f"No suitable_sources found in expanded_queries_info, using default: {sources}"
                )
        else:
            # Use default sources based on depth
            sources = SEARCH_ROUTES if current_depth == 1 else ["arxiv"]
            logger.info(f"Using depth-based sources: {sources}")

        if "arxiv" not in sources:
            sources.insert(0, "arxiv")
            logger.info(f"Adding 'arxiv' to sources: {sources}")

        # 使用意图分析确定的 sources，不做强制覆盖
        if current_depth > 1:
            # 深层搜索用 arxiv 为主（速度快），保留 openalex 做辅助
            if "arxiv" not in sources:
                sources.insert(0, "arxiv")
            if "openalex" not in sources:
                sources.append("openalex")
        else:
            # 第1层使用完整 sources（来自意图分析或配置）
            if "arxiv" not in sources:
                sources.insert(0, "arxiv")
        logger.info(f"Using sources: {sources}")

        try:
            # Execute batch search across multiple sources
            batch_result, id2docs, query_source_map, query_keywords2raw = (
                self.search_engine.search_papers_mroute(
                    expanded_queries,
                    end_date=search_date,
                    searched_docs=self.root.searched_docs,
                    sources=sources,
                    forced_keywords=forced_keywords,
                )
            )

            # Update search state
            # only store the query string, do not store the keywords
            self.root.searched_queries.update(expanded_queries)
            self.root.add_signature_for_doc(list(id2docs.values()))

            # Save to local DB if enabled
            if SAVE_ID2DOCS:
                self._save_id2info(id2docs)

            # Create nodes based on search results
            current_level_node = []

            for query, retrival_docs in batch_result.items():
                query_source = query_source_map[query]

                if not retrival_docs:
                    status = "Failed"
                else:
                    status = "Finshed"

                logger.info(f"{query[:60]}... --- source: {query_source}: {status}" if len(query) > 60 else f"{query} --- source: {query_source}: {status}")

                if query in query_node_relations:
                    assert (
                        query_keywords2raw[query] == query
                    ), f"query_keywords2raw: {query_keywords2raw[query]} != {query}"
                    node = query_node_relations[query]["own_node"]
                    assert (
                        node.query_str == query
                    ), f"node.query_str: {node.query_str} != {query}"
                    parent_node = query_node_relations[query]["parent_node"]
                    node.status = status
                    node.source = query_source
                    node.parent = parent_node
                    node.raw_query = query
                    parent_node.add_child(node)
                    current_level_node.append(node)
                else:
                    raw_query = query_keywords2raw[query]
                    parent_node = query_node_relations[raw_query]["parent_node"]
                    new_node = SearchNode(
                        query_str=query,
                        status=status,
                        parent=parent_node,
                        source=query_source,
                        raw_query=raw_query,
                    )
                    parent_node.add_child(new_node)
                    current_level_node.append(new_node)
            # ==== 统一评分：按 source 合并去重 → 评分一次 → 分发回各 node ====
            logger.info(f"current level node: {len(current_level_node)}")

            # ── Step A: 按 source 收集所有论文，去重，记录来源 query ──
            source_paper_pool = {}  # source → {paper_id: {doc, found_by_queries: set()}}
            for node in current_level_node:
                raw_docs = batch_result.get(node.query_str, [])
                if not raw_docs:
                    continue
                if node.source not in source_paper_pool:
                    source_paper_pool[node.source] = {}
                pool = source_paper_pool[node.source]
                for doc in raw_docs:
                    pid = doc.get("paper_id", doc.get("arxivId"))
                    if not doc.get("title") or not doc.get("abstract"):
                        continue
                    if pid in self.root.cal_sim_docs:
                        continue  # 已在之前深度中评分过，跳过
                    if pid not in pool:
                        pool[pid] = {"doc": doc, "found_by_queries": set()}
                    pool[pid]["found_by_queries"].add(node.query_str)

            # ── Step B: 对每个 source 统一评分 ──
            source_score_map = {}  # source → {paper_id: score_info}
            for source, pool in source_paper_pool.items():
                unique_docs = [p["doc"] for p in pool.values()]
                if not unique_docs:
                    continue
                logger.info(
                    f"[{source}] Unified scoring: {len(unique_docs)} unique papers "
                    f"(from {len(pool)} unique IDs)"
                )
                relevant, irrelevant = self.search_engine.calculate_similarity(
                    query=self.root.query_str,
                    docs=unique_docs,
                    search_time=self.search_date,
                    score_thresh=self.sim_threshold,
                    source=f"from retrieval, source: [{source}]",
                )
                # 建立 paper_id → score_info 的查找表，同时注入来源信息
                lookup = {}
                for r in relevant:
                    pid = r.get("paper_id", r.get("arxivId"))
                    r["found_by_queries"] = list(pool.get(pid, {}).get("found_by_queries", []))
                    lookup[pid] = r
                for ir in irrelevant:
                    pid = ir.get("paper_id", ir.get("arxivId"))
                    ir["found_by_queries"] = list(pool.get(pid, {}).get("found_by_queries", []))
                    lookup[pid] = ir
                source_score_map[source] = lookup

                self.root.add_signature_for_doc(relevant)
                self.root.cal_sim_docs.update({
                    one.get("paper_id", one.get("arxivId")): one
                    for one in relevant + irrelevant
                })

            # ── Step C: 将评分结果分发回各 node ──
            valid_doc_count = 0
            rel_doc_count = 0
            for node in tqdm.tqdm(
                current_level_node,
                total=len(current_level_node),
                desc="Distributing scored results",
            ):
                try:
                    raw_docs = batch_result.get(node.query_str, [])
                    if not raw_docs:
                        node.status = "Failed"
                        next_level.append(node)
                        continue

                    lookup = source_score_map.get(node.source, {})
                    node_docs = []
                    node_irrelevant = []
                    seen_in_node = set()
                    for doc in raw_docs:
                        pid = doc.get("paper_id", doc.get("arxivId"))
                        if not doc.get("title") or not doc.get("abstract"):
                            continue
                        if pid in seen_in_node:
                            continue
                        seen_in_node.add(pid)
                        score_info = lookup.get(pid)
                        if score_info is None:
                            continue  # 已在 cal_sim_docs 中（跨层去重）
                        if score_info.get("sim_score", 0) >= self.sim_threshold:
                            node_docs.append(score_info)
                        else:
                            node_irrelevant.append(score_info)

                    valid_doc_count += len(node_docs) + len(node_irrelevant)

                    if not node_docs and not node_irrelevant:
                        node.status = "Failed"
                        if node not in next_level:
                            next_level.append(node)
                    elif node_docs:
                        node.status = "Expand"
                        rel_doc_count += len(node_docs)
                    else:
                        node.status = "NO Relevance"

                    node.docs.extend(node_docs)
                    node.irrelevant_docs.extend(node_irrelevant)

                except Exception as e:
                    logger.error(
                        f"Error processing node {node.query_str}: {traceback.format_exc()}"
                    )
                    node.status = "Error"

            logger.info(
                f"Query level search completed: {valid_doc_count} valid docs, "
                f"{rel_doc_count} relevant docs, failed node num: {len(next_level)}"
            )
            return current_level_node, next_level

        except Exception as e:
            logger.error(f"Batch search failed: {traceback.format_exc()}")
            return [], next_level

    def reference_level_search(self, level_node):
        """
        Explores references of retrieved documents to find additional relevant papers.

        Args:
            level_node: List of nodes whose documents' references will be explored

        Returns:
            Updated list of nodes with references added
        """
        if not level_node:
            logger.warning("No nodes provided for reference_level_search")
            return level_node

        logger.info(f"Running reference search on {len(level_node)} nodes")

        try:
            # Collect all relevant documents for reference exploration
            MIN_EXPAND_DOCS = 5
            all_rel_docs = []
            for node in level_node:
                # Try threshold first; if too few, fall back to top N by sim_score
                expand_docs = [
                    doc for doc in (node.docs + node.irrelevant_docs)
                    if doc.get("sim_score", 0) >= REFERENCE_EXPAND_THRESHOLD
                ]
                if len(expand_docs) < MIN_EXPAND_DOCS:
                    # Fallback: take top MIN_EXPAND_DOCS docs regardless of score
                    node_all = sorted(
                        node.docs + node.irrelevant_docs,
                        key=lambda d: d.get("sim_score", 0),
                        reverse=True,
                    )
                    expand_docs = node_all[:MIN_EXPAND_DOCS]
                    logger.debug(
                        f"Reference expand threshold too strict ({len(expand_docs)} docs ≥ {REFERENCE_EXPAND_THRESHOLD}), "
                        f"falling back to top {MIN_EXPAND_DOCS} docs for node '{node.query_str[:40]}'"
                    )
                for doc in expand_docs:
                    all_rel_docs.append([node, doc])

            # Sort by similarity score (most relevant first)
            all_rel_docs = sorted(
                all_rel_docs, key=lambda x: x[1].get("sim_score", 0), reverse=True
            )

            # Limit to top documents to prevent excessive exploration
            docs_to_expand = min(len(all_rel_docs), DOCS_TO_EXPAND)
            all_rel_docs = all_rel_docs[:docs_to_expand]
            logger.info(
                f"Selected {docs_to_expand} documents for reference exploration"
            )

            # Process documents in batches
            batch_size = 2
            ref_count = 0
            relevant_ref_count = 0

            # Track which papers we've already processed to avoid duplicates
            processed_paper_ids = set()

            for index in tqdm.tqdm(
                range(0, len(all_rel_docs), batch_size),
                total=len(all_rel_docs) // batch_size
                + (1 if len(all_rel_docs) % batch_size else 0),
                desc="Reference Level Search",
            ):
                # Break early if stopping condition is met
                if self.meet_stop_condition():
                    logger.info(
                        "Stopping reference search early: stopping condition met"
                    )
                    # Mark remaining nodes as stopped
                    for i in range(index, len(all_rel_docs), batch_size):
                        if i < len(all_rel_docs):
                            node, _ = all_rel_docs[i]
                            node.status = "STOP"
                    return level_node

                # Process the current batch
                start_idx = index
                end_idx = min(index + batch_size, len(all_rel_docs))
                batch = all_rel_docs[start_idx:end_idx]

                for node, doc in batch:
                    try:
                        # Skip if we've already processed this paper
                        paper_id = doc.get("paper_id", "")
                        if paper_id in processed_paper_ids:
                            continue
                        processed_paper_ids.add(paper_id)

                        # Get document info from search state
                        doc_info = self.root.searched_docs.get(doc["paper_id"])
                        if not doc_info:
                            logger.warning(
                                f"Document info not found for {doc['paper_id']}"
                            )
                            continue

                        # Get references if not already available
                        refs = doc_info.get("references", [])
                        if not refs or not [
                            valid
                            for valid in refs
                            if valid.get("title") and valid.get("abstract")
                        ]:
                            doc_info = self.search_engine.get_doc_references(doc_info)
                            refs = doc_info.get("references", [])

                        if not refs:
                            logger.warning(
                                f"No references found for document: {doc_info.get('title', 'Unknown')}"
                            )
                            continue

                        # Update document signatures
                        self.root.add_signature_for_doc([doc_info])
                        self.root.add_signature_for_doc(refs)

                        # Filter valid references
                        valid_refs = [
                            ref
                            for ref in refs
                            if ref.get("title") and ref.get("abstract")
                        ]

                        ref_count += len(valid_refs)
                        doc_info["references"] = valid_refs

                        # Save to local DB if enabled
                        if SAVE_ID2DOCS:
                            self._save_id2info(
                                {ref["paper_id"]: ref for ref in valid_refs}
                            )
                            self._save_id2info({doc_info["paper_id"]: doc_info})

                        # Record references
                        node.references.extend([ref["paper_id"] for ref in valid_refs])

                        if not valid_refs:
                            logger.info(
                                f"No valid references found for {doc.get('title', 'Unknown')}"
                            )
                            continue

                        # Calculate similarity for references
                        relevant_refs, irrelevance_refs = (
                            self.search_engine.calculate_similarity(
                                self.root.query_str,
                                valid_refs,
                                search_time=self.search_date,
                                score_thresh=self.sim_threshold,
                                source=f"from reference, parent: {doc.get('arxivId', 'unknown')}",
                            )
                        )

                        relevant_ref_count += len(relevant_refs)

                        # Update document collections
                        self.root.add_signature_for_doc(
                            relevant_refs
                        )
                        self.root.cal_sim_docs.update(
                            {
                                one["paper_id"]: one
                                for one in relevant_refs + irrelevance_refs
                            }
                        )

                        # Update node references
                        node.relevance_refs.extend(relevant_refs)
                        node.irrelevant_refs.extend(irrelevance_refs)

                        node.docs.extend(relevant_refs)
                        node.irrelevant_docs.extend(irrelevance_refs)

                    except Exception as e:
                        logger.error(
                            f"Error processing references for {doc.get('title', 'Unknown')}: {str(e)}"
                        )

            logger.info(
                f"Reference search completed: {ref_count} refs found, {relevant_ref_count} relevant"
            )
            return level_node

        except Exception as e:
            logger.error(f"Reference level search failed: {traceback.format_exc()}")
            return level_node

    def query_expand_from_context(self, level_node, next_level, search_queue):
        logger.info("Generate new query for next level expand ...")
        nex_level_prepare = []
        generate_new_query = []

        current_level_all_valid_docs = []
        for node in tqdm.tqdm(
            level_node, total=len(level_node), desc="query_expand_from_context"
        ):
            valid_docs = [
                [doc, node]
                for doc in node.docs
                if doc["paper_id"] not in self.root.doc_used_to_gen_query
            ]
            current_level_all_valid_docs.extend(valid_docs)

        logger.info(
            f"current_level_all_valid_docs: {len(current_level_all_valid_docs)}"
        )
        if len(current_level_all_valid_docs) == 0:
            logger.info("No relevance doc find, generate some new query")
            valid_docs_info = [[{}, self.root]]
        else:
            current_level_all_valid_docs_sort = sorted(
                current_level_all_valid_docs,
                key=lambda x: x[0]["sim_score"],
                reverse=True,
            )
            valid_docs = current_level_all_valid_docs_sort[
                :REFERENCE_DOC_NUM_TO_GEN_NEW_QUERY
            ]

            valid_docs_info = [
                [self.root.searched_docs[doc["paper_id"]], node]
                for doc, node in valid_docs
            ]

            valid_docs_id = [doc["paper_id"] for doc, node in valid_docs]
            self.root.doc_used_to_gen_query.update(valid_docs_id)

        new_queries = self.search_engine.generate_queries_from_docs(
            self.root.query_str, valid_docs_info, list(self.root.searched_queries)
        )

        query_samples = [q[:50] for q, _ in new_queries[:3]]
        logger.info(f"generate {len(new_queries)} queries (samples: {query_samples}{'...' if len(new_queries) > 3 else ''})")

        for query, parent_node in new_queries:
            if query not in generate_new_query:
                generate_new_query.append(query)
                child = SearchNode(query_str=query)
                nex_level_prepare.append([parent_node, child])
            else:
                logger.debug(f"Query already generated, skip: {query[:40]}")

        query_node_relations = {}
        querys_to_next_level = []

        logger.info(f"next_level: {len(next_level)}")
        if next_level:
            for one in next_level:
                query_node_relations[one.query_str] = {
                    "own_node": one,
                    "parent_node": one.parent,
                }
                querys_to_next_level.append(one.query_str)

        logger.info(f"nex_level_prepare: {len(nex_level_prepare)}")
        if nex_level_prepare:
            if QUERY_NUM_PRUNED < len(nex_level_prepare):
                nex_level_prepare_shuffle = self._select_diverse_queries(
                    nex_level_prepare, QUERY_NUM_PRUNED
                )
            else:
                nex_level_prepare_shuffle = nex_level_prepare
            for node_inf in nex_level_prepare_shuffle:
                parent_node, child_node = node_inf
                # child_node.parent = parent_node
                # parent_node.add_child(child_node)
                child_node.status = "Expand"

                querys_to_next_level.append(child_node.query_str)
                query_node_relations[child_node.query_str] = {
                    "own_node": child_node,
                    "parent_node": parent_node,
                }
            search_queue.append(querys_to_next_level)

        return level_node, search_queue, query_node_relations

    def search(self, initial_query: str, end_date="", filter_params: dict = None, sort_by: str = 'year', selected_queries: list = None, expanded_queries: list = None, selected_keywords: list = None) -> List:
        """
        Main search method that:
        1. Initializes search tree with root query
        2. Performs iterative BFS search
        3. Expands search via reference exploration
        4. Returns ranked relevant documents

        Args:
            initial_query: User's search query
            end_date: Optional cutoff date for papers

        Returns:
            Dictionary of relevant documents
        """
        if filter_params is not None:
            self.filter_config = filter_params
        logger.info(f"search() received filter_params: {filter_params}")

        search_start_time = time.time()

        # Set search date
        self.search_date = ""

        # Initialize search tree
        logger.info(" Initializing academic search tree")
        self.root = SearchNode(
            query_str=initial_query,
            status="INIT",
        )
        self.user_query = initial_query
        self.forced_keywords = selected_keywords  # 用户预选的关键词（仅 depth=1 生效）

        try:
            # 如果传入了预生成的扩展查询（来自预览阶段），直接使用，跳过 query_fusion
            if expanded_queries is not None:
                logger.info(f"Using pre-generated expanded queries ({len(expanded_queries)} queries)")
                query_node_relations = {}
                for q in expanded_queries:
                    node = SearchNode(query_str=q, status="START")
                    query_node_relations[q] = {"own_node": node, "parent_node": self.root}
                self.root.extra["expanded_queries_info"] = {
                    "suitable_sources": ["arxiv", "openalex"],
                    "expanded_queries": expanded_queries,
                }
            else:
                # Start with query fusion to generate initial queries
                expanded_queries, query_node_relations = self.query_fusion()

            # 如果用户指定了 selected_queries，只保留选中的查询
            if selected_queries is not None:
                selected_set = set(selected_queries)
                filtered = [q for q in expanded_queries if q in selected_set]
                if filtered:
                    logger.info(f"Using {len(filtered)}/{len(expanded_queries)} selected queries")
                    expanded_queries = filtered
                    query_node_relations = {k: v for k, v in query_node_relations.items() if k in selected_set}
                else:
                    logger.warning(f"No selected queries matched expanded queries, using all {len(expanded_queries)}")

            search_queue = deque([expanded_queries])

            # Track search progress
            current_depth = 0
            iteration = 0

            # Main search loop
            while search_queue and not self.meet_stop_condition(current_depth):
                iteration += 1
                next_level = []
                level_queries = search_queue.popleft()
                current_depth += 1

                assert isinstance(
                    level_queries, list
                ), f"level_queries: {level_queries}"

                logger.info(f"=== Iteration {iteration}, Depth {current_depth} ===")
                logger.info(f"Processing {len(level_queries)} queries at this level")

                # Phase 1: Query-level search (now takes query strings instead of nodes)
                logger.info(f"Phase 1: Running query-level search")
                level_node, next_level = self.query_level_search(
                    level_queries,
                    query_node_relations,
                    next_level,
                    self.search_date,
                    current_depth,
                    forced_keywords=self.forced_keywords if current_depth == 1 else None,
                )

                # ===== 每层评分后立即过滤低分论文 =====
                if self.root.searched_docs and self.sim_threshold > 0:
                    before = len(self.root.searched_docs)
                    self.root.searched_docs = {
                        pid: doc for pid, doc in self.root.searched_docs.items()
                        if doc.get('sim_score', 0) is not None and doc.get('sim_score', 0) >= self.sim_threshold
                    }
                    after = len(self.root.searched_docs)
                    if after < before:
                        logger.info(f"Depth {current_depth} sim filter: {before} → {after} docs")

                # Check if we've found enough documents
                if self.meet_stop_condition(current_depth):
                    logger.info(
                        "Stopping after query-level search: stopping condition met"
                    )
                    for node in level_node:
                        node.status = "STOP"
                    break

                # Phase 2: Reference-based search
                if DO_REFERENCE_SEARCH:
                    logger.info(f"Phase 2: Running reference-level search")
                    level_node = self.reference_level_search(level_node=level_node)

                    if self.meet_stop_condition(current_depth):
                        logger.info(
                            "Stopping after reference search: stopping condition met"
                        )
                        break

                # Phase 3: Generate new queries for next iteration
                logger.info(f"Phase 3: Expanding to next level")
                level_node, search_queue, query_node_relations = (
                    self.query_expand_from_context(level_node, next_level, search_queue)
                )

                logger.info(
                    f"Added {len(query_node_relations)} nodes to search queue for next iteration"
                )
                logger.info(f"Search queue size: {len(search_queue)}")

            # Optional reranking
            # if RERANK:
            #     logger.info("Reranking final document list")

            #     reranked_top = self.reranker.rerank_query_and_doc_list(
            #         self.root.searched_docs, self.user_query
            #     )
            #     self.root.reranked_top_docs = reranked_top

            search_time = time.time() - search_start_time
            doc_count = len(self.root.searched_docs)
            logger.info(
                f"Search completed in {search_time:.2f}s. Found {doc_count} documents."
            )

            # Generate performance statistics
            high_rel_count = len(
                [
                    doc
                    for docid, doc in self.root.searched_docs.items()
                    if doc.get("sim_score", 0) > self.high_score_thresh
                ]
            )
            logger.info(
                f"Found {high_rel_count} highly relevant documents (score > {self.high_score_thresh})"
            )

            # wsl-73二次筛选
            if ENABLE_LLM_RERANK:
                logger.info("Applying LLM fine-grained filtering on reranked top docs...")
                # 获取重排序后的文档（如果存在）
                reranked_docs = self.root.reranked_top_docs if self.root.reranked_top_docs else list(
                    self.root.searched_docs.values())
                # 只对前 TOP_FOR_LLM_FILTER 篇打分
                TOP_FOR_LLM_FILTER = 50  # 可调
                docs_to_filter = reranked_docs[:TOP_FOR_LLM_FILTER]
                filtered_docs = {}
                for doc in docs_to_filter:
                    doc_id = doc.get("paper_id", "")
                    if not doc_id:
                        continue
                    llm_score = llm_relevance_score(self.user_query, doc)
                    doc['llm_score'] = llm_score
                    # 降低阈值，避免误杀
                    if llm_score >= 0.3:  # 调低阈值，宁可多留一些
                        filtered_docs[doc_id] = doc
                    else:
                        logger.debug(f"Filtered doc {doc_id} with LLM score {llm_score:.2f}")
                # 将未过滤的文档（超出TOP_FOR_LLM_FILTER的部分）也保留，但分数用原分数
                for doc in reranked_docs[TOP_FOR_LLM_FILTER:]:
                    doc_id = doc.get("paper_id", "")
                    if doc_id and doc_id not in filtered_docs:
                        filtered_docs[doc_id] = doc
                self.root.searched_docs = filtered_docs
                logger.info(f"After LLM filter: {len(filtered_docs)} documents kept")
            # wsl-710 ===== 新增：应用硬性过滤条件 =====
            if self.root.searched_docs:
                self.root.searched_docs = self._filter_docs(
                    self.root.searched_docs,
                    filter_config=self.filter_config
                )
            #if self.root.searched_docs:
            #    self.root.searched_docs = self._filter_docs(self.root.searched_docs)
            #    logger.info(f"After filter: {len(self.root.searched_docs)} documents remain")

            if RERANK:
                logger.info("Applying LLM reranking on all collected docs...")
                all_docs = list(self.root.searched_docs.values())
                if all_docs:
                    reranked_list = self.reranker.rerank_query_and_doc_list(
                        all_docs, self.user_query, score_name="sim_score",sort_by=sort_by
                    )
                    if reranked_list:
                        # 截断到 self.max_docs（保持排序顺序）
                        reranked_list = reranked_list[:self.max_docs]
                        # 构建有序字典（Python 3.7+ 保留插入顺序）
                        new_searched = {}
                        for doc in reranked_list:
                            doc_id = doc.get("paper_id", "")
                            if doc_id:
                                # 保留 rerank_score 作为 sim_score 以便下游使用
                                doc['sim_score'] = doc.get('rerank_score', doc.get('sim_score', 0.0))
                                new_searched[doc_id] = doc
                        self.root.searched_docs = new_searched
                        self.root.reranked_top_docs = reranked_list  # 保存排序列表以备后用
                        logger.info(f"Reranking done. Kept {len(new_searched)} docs.")
                    else:
                        logger.warning("Reranking returned empty, keeping original.")
                        # 如果重排序失败，回退到原有按 sim_score 排序截断
                        if self.root.searched_docs:
                            sorted_items = sorted(
                                self.root.searched_docs.items(),
                                key=lambda item: item[1].get('sim_score', 0.0),
                                reverse=True
                            )
                            filtered = {}
                            kept = 0
                            for doc_id, doc in sorted_items:
                                if doc.get('sim_score', 0.0) >= SIM_THRESHOLD and kept < self.max_docs:
                                    filtered[doc_id] = doc
                                    kept += 1
                            self.root.searched_docs = filtered
                            logger.info(f"Fallback filter: kept {kept} docs")
            else:
                # 不启用 RERANK，执行原有的按 sim_score 排序截断
                if self.root.searched_docs:
                    sorted_items = sorted(
                        self.root.searched_docs.items(),
                        key=lambda item: item[1].get('sim_score', 0.0),
                        reverse=True
                    )
                    filtered = {}
                    kept = 0
                    for doc_id, doc in sorted_items:
                        if doc.get('sim_score', 0.0) >= SIM_THRESHOLD and kept < self.max_docs:
                            filtered[doc_id] = doc
                            kept += 1
                    self.root.searched_docs = filtered
                    logger.info(f"Final filter: kept {kept} docs (threshold={SIM_THRESHOLD}, max={self.max_docs})")

            # 最后返回结果（需修改 _collect_results 避免打乱顺序）
            return self._collect_results()

        except Exception as e:
            logger.error(f"Search failed: {traceback.format_exc()}")
            # Return whatever results we've gathered so far
            return self._collect_results()
        finally:
            # Always clean up resources
            self._cleanup_resources()

    def _collect_results(self) -> Dict:
        """Collect search results with diversity optimization, but respect reranked order if available."""
        # 如果重排序结果存在，直接返回（保持顺序）
        if hasattr(self.root, 'reranked_top_docs') and self.root.reranked_top_docs:
            # 构建 all_papers 字典（用于前端）
            all_papers = {}
            for doc in self.root.reranked_top_docs:
                doc_id = doc.get("paper_id", "")
                if doc_id:
                    all_papers[doc_id] = doc
            # 返回结构（模拟原 _collect_results 返回值）
            return all_papers   # 但原 _collect_results 返回的是字典，不是列表
        # 否则执行原有逻辑（多样性选择）
        # Get all documents
        all_docs = self.root.searched_docs
        sorted_docs = sorted(
            all_docs.items(),
            key=lambda x: x[1].get("sim_score", 0),
            reverse=True,
        )
        return dict(sorted_docs[:self.max_docs])

    def _select_diverse_queries(self, candidates, k):
        """
        基于 Jaccard 多样性的贪心查询选择。
        从候选查询中选择 k 个在内容上最多样化的查询，替代随机采样。
        """
        if len(candidates) <= k:
            return candidates

        def tokenize(s):
            return set(re.findall(r'\w+', s.lower()))

        def jaccard_sim(t1, t2):
            inter = len(t1 & t2)
            union = len(t1 | t2)
            return inter / union if union > 0 else 0.0

        # 计算每个候选查询的 token 集合
        candidates_with_tokens = [
            (parent_node, child_node, tokenize(child_node.query_str))
            for parent_node, child_node in candidates
        ]

        # 贪心选择：先选第一个，然后每次选与已选集合最不相似的
        selected = [candidates_with_tokens[0]]
        remaining = candidates_with_tokens[1:]

        while len(selected) < k and remaining:
            # 对每个剩余候选，计算它与已选集合的最小 Jaccard 距离（最大差异 = 最小相似度）
            best_idx = 0
            best_min_sim = float('inf')  # 找最小相似度（最大差异）
            for i, (p_node, c_node, tokens) in enumerate(remaining):
                max_sim_to_selected = max(
                    jaccard_sim(tokens, s_tokens)
                    for _, _, s_tokens in selected
                )
                if max_sim_to_selected < best_min_sim:
                    best_min_sim = max_sim_to_selected
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return [[p, c] for p, c, _ in selected]

    def _rank_query_doc_list(self, docs):
        """
        1. First, divide the data into buckets by `sim_score`, with each bucket divided into 0.5 buckets (0-0.5, 0.5-1.0, 1.0-1.5,...).
        2. In each bucket:
            - Sort by `citationCount` in descending order.
            - If `citationCount` is the same, sort by `year` in descending order.
        """

        from collections import defaultdict

        def bucketize(score):
            return round(score / 0.05) * 0.05

        bucketed_docs = defaultdict(list)
        for doc in docs:
            sim_score = doc.get("sim_score", 0.0)
            bucket_key = bucketize(sim_score)

            if doc["year"] is None:
                doc["year"] = -1
            bucketed_docs[bucket_key].append(doc)

        logger.debug(f"bucketed_docs: {bucketed_docs.keys()}")

        sorted_docs = []
        for sim_score in sorted(bucketed_docs.keys(), reverse=True):
            sorted_bucket = sorted(
                bucketed_docs[sim_score],
                key=lambda d: (-d.get("citationCount", 0), -d.get("year", 0)),
            )
            sorted_docs.extend(sorted_bucket)

        return sorted_docs

    def visualize_tree(
        self,
        filename: str = "search_tree",
        save_format: str = "pdf",
        view: bool = False,
    ):
        """
        Visualize the search tree and save it to a file, showing more information:
        - Search query for each node
        - Number of retrieved documents
        - Query weight (if applicable)

        Args:
            filename (str): Filename to save (without extension).
            save_format (str): File format, e.g., "pdf", "png", "svg".
            view (bool): Whether to automatically open the generated file.
        """
        dot = Digraph(comment="Search Tree")
        if not self.root:
            logger.error("Search tree is empty. No visualization will be created.")
            return

        search_queue = deque([(self.root, "0")])
        node_counter = 0

        def process_query(query_str, split=10):
            query_str_processed = ""
            query_str_spl = query_str.split(" ")
            for idx in range(0, len(query_str_spl), split):
                spl = " ".join(query_str_spl[idx : idx + split])
                query_str_processed += spl + "\n"
            return query_str_processed

        while search_queue:
            node, node_id = search_queue.popleft()
            if node_id == "0":
                node_label = process_query(node.query_str)
                dot.node(node_id, "TreeSearch\nStart", shape="hexagon")
            else:
                # if node.status in ["ACHIEVED AND STOP"]:
                #     continue
                num_docs = len(node.docs) if node.docs else 0
                num_relevalce_refs = (
                    len(node.relevance_refs) if node.relevance_refs else 0
                )
                num_references = len(node.references)
                remove_num_docs = (
                    len(node.irrelevant_docs) if node.irrelevant_docs else 0
                )
                source = node.source
                query_str_processed = process_query(node.query_str, 4)
                if node.raw_query != node.query_str:
                    raw_query_processed = process_query(node.raw_query, 4)
                    node_label = f"[{source}]: {query_str_processed}\n[RAW-QUERY]: {raw_query_processed}\nAllRelDocs: [{num_docs}] AllIrrelDocs: [{remove_num_docs}]\nAllRefs: [{num_references}] RelRefs: [{num_relevalce_refs}]"

                else:
                    node_label = f"[{source}]: {query_str_processed}\nAllRelDocs: [{num_docs}] AllIrrelDocs: [{remove_num_docs}]\nAllRefs: [{num_references}] RelRefs: [{num_relevalce_refs}]"

                if hasattr(node, "weight"):
                    node_label += f"\nWeight: {getattr(node, 'weight', 1.0):.2f}"

                node_label += f"\nStatus:{node.status}"
                dot.node(node_id, node_label, shape="box")

            for i, child in enumerate(node.children):
                node_counter += 1
                child_id = str(node_counter)
                dot.edge(node_id, child_id)
                search_queue.append((child, child_id))

        logger.info(f"filename: {filename}")
        filepath = dot.render(filename=filename, format=save_format, view=view)
        logger.info(f"搜索树已保存为文件: {filepath}")
