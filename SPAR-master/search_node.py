# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

from typing import List, Optional, Callable, Dict


class SearchNode:
    """
    Represents a node in the academic search tree.
    Each node contains:
    - A search query
    - Relevant and irrelevant documents
    - References
    - Child nodes for query expansion
    """

    def __init__(
        self,
        query_str: str = "",
        query_weight: float = 1.0,
        status: str = "INIT",
        parent: Optional["SearchNode"] = None,
        source: List[str] = None,
        raw_query: str= "",
        reranker: Optional["Reranker"] = None,  # wsl<--- 新增参数
        **attrs
    ):
        # Query information
        self.query_str = query_str  # Search query string
        self.query_weight = query_weight  # Query importance weight
        self.status = status  # Node status (INIT, SEARCH, END, etc.)
        self.searched_queries = set()
        self.doc_used_to_gen_query = set()
        self.source = source or []  # Search channels for this query
        self.raw_query = raw_query
        # Search results
        self.docs = []  # Relevant documents
        self.irrelevant_docs = []  # Irrelevant documents
        self.relevance_refs = []  # Relevant reference documents
        self.irrelevant_refs = []  # Irrelevant reference documents
        self.references = []  # References from relevant docs
        self.children = []  # Child query nodes
        self.searched_docs = dict()
        self.reranked_top_docs = []
        self.hight_relevance_docs = set()  # 高相关度doc列表
        self.cal_sim_docs = dict()  # 计算过分数的doc
        # Node metadata
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0
        self.extra = attrs.get("extra", {})  # Additional attributes
        self.reranker = None  # wsl-72

    def convert_to_dict(self) -> dict:
        """Convert node to dictionary format for serialization"""
        self.extra["searched_docs"] = self.searched_docs
        self.extra["searched_queries"] = list(self.searched_queries)
        self.extra["hight_relevance_docs"] = list(self.hight_relevance_docs)
        return {
            "search_query": self.query_str,
            "query_weight": self.query_weight,
            "children": [child.convert_to_dict() for child in self.children],
            "docs": [dict(doc) for doc in self.docs],
            "irrelevant_docs": [dict(doc) for doc in self.irrelevant_docs],
            "references": [ref for ref in self.references],
            "depth": self.depth,
            "search_status": self.status,
            "source": self.source,
            "reranked_top_docs": self.reranked_top_docs,  # wsl<--- 新增
            "extra": self.extra,
        }

    def sort_doc(self) -> None:
        """Sort documents by similarity score in descending order"""
        self.docs = sorted(self.docs, key=lambda x: x["sim_score"], reverse=True)
        self.irrelevant_docs = sorted(
            self.irrelevant_docs, key=lambda x: x["sim_score"], reverse=True
        )

    def add_searched_query(self, queries):
        for query in queries:
            self.searched_queries.add(query)

    def add_signature_for_doc(self, docs):
        for doc in docs:
            if "paper_id" not in doc and "arxivId" in doc:
                doc["paper_id"] = doc["arxivId"]
            if doc["paper_id"] in self.searched_docs:
                self.searched_docs[doc["paper_id"]].update(doc)
            else:
                self.searched_docs[doc["paper_id"]] = doc

    def add_child(self, child: "SearchNode") -> None:
        """Add a child node"""
        child.parent = self
        child.depth = self.depth + 1
        # wsl 如果子节点没有自己的 reranker，继承父节点的
        if child.reranker is None and self.reranker is not None:
            child.reranker = self.reranker
        self.children.append(child)

    def apply_reranking(self, user_query: str, score_name: str = "sim_score") -> None: #wsl-启用重排序
        """
        使用注入的 Reranker 对当前节点的 docs 进行重排序。
        结果将存储在 self.reranked_top_docs 中。
        """
        if not self.reranker:
            logger.warning("No reranker instance provided. Skipping reranking.")
            return

        if not self.docs:
            logger.info("No documents to rerank in this node.")
            return

        logger.info(f"Applying reranking for query: {user_query}")
        # 调用重排序器（注意：rerank.py 返回的是排序后的列表）
        reranked = self.reranker.rerank_query_and_doc_list(
            all_docs=self.docs,
            user_query=user_query,
            score_name=score_name
        )

        # 如果重排序成功返回了结果（非空），则存储；否则保留原有排序
        if reranked:
            self.reranked_top_docs = reranked
            logger.info(f"Reranking complete. Top {len(reranked)} docs stored.")
        else:
            logger.warning("Reranking returned empty results, keeping original order.")

    def get_best_docs(self) -> List[Dict]: #wsl
        """
        获取当前节点最优的文档列表。
        优先返回重排序后的结果；若无，则返回按 sim_score 降序排序的原始结果。
        """
        if self.reranked_top_docs:
            return self.reranked_top_docs
        else:
            # 这里可以复用 sort_doc 的逻辑，但 sort_doc 会修改 self.docs 本身。
            # 为了不改变原始顺序，这里返回一个排序后的副本。
            return sorted(self.docs, key=lambda x: x.get("sim_score", 0), reverse=True)

    @property
    def has_results(self) -> bool:
        """Check if node has any search results"""
        return len(self.docs) > 0

    @property
    def total_docs(self) -> int:
        """Get total number of documents (relevant + irrelevant)"""
        return len(self.docs) + len(self.irrelevant_docs)