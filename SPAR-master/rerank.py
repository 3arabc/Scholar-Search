# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
import re
from log import logger
import traceback
from datetime import datetime, timedelta
from local_request_v2 import get_from_llm
from global_config import RERANK_MODEL
from typing import List, Dict, Union
import json
import os
from log import logger #wsl-71

class Reranker(object):

    def rerank_query_and_doc_list(self, all_docs, user_query, score_name="sim_score", sort_by='year'):
        """
        根据 sort_by 参数对文档进行排序：
            'year'      : 按出版年份降序（最新优先），无效年份排在最后
            'citations' : 按引用数降序（最多优先），无效引用排在最后
            'similarity': 按相似度降序（最高优先），无效相似度排在最后（但相似度通常都存在）
        """
        logger.info(f"Reranking documents by {sort_by} (descending), invalid values at the end...")
        self.user_query = user_query
        self.score_name = score_name

        if isinstance(all_docs, dict):
            all_docs = list(all_docs.values())

        if not all_docs:
            logger.warning("No documents to rerank")
            return []

        def sort_key(doc):
            if sort_by == 'year':
                year = doc.get("publicationYear") or doc.get("year")
                try:
                    year_val = int(year) if year is not None else None
                except (ValueError, TypeError):
                    year_val = None
                # 有效标志：1有效，0无效
                valid = 1 if year_val is not None else 0
                # 排序值：有效时取负值（降序），无效时取 0
                sort_val = -year_val if year_val is not None else 0
                return (valid, sort_val)

            elif sort_by == 'citations':
                citations = doc.get("citationCount") or doc.get("citations")
                try:
                    cit_val = int(citations) if citations is not None else None
                except (ValueError, TypeError):
                    cit_val = None
                valid = 1 if cit_val is not None else 0
                sort_val = -cit_val if cit_val is not None else 0
                return (valid, sort_val)

            else:  # 'similarity'
                sim = doc.get(score_name, 0.0)
                # 相似度通常总是存在，但为了统一，也做有效性检查
                try:
                    sim_val = float(sim) if sim is not None else None
                except (ValueError, TypeError):
                    sim_val = None
                valid = 1 if sim_val is not None else 0
                sort_val = -sim_val if sim_val is not None else 0
                return (valid, sort_val)

        sorted_docs = sorted(all_docs, key=sort_key)

        # 为每个文档添加 rerank_score（与排序键对应，方便下游使用）
        for doc in sorted_docs:
            if sort_by == 'year':
                year = doc.get("publicationYear") or doc.get("year")
                try:
                    doc['rerank_score'] = int(year) if year is not None else 0
                except:
                    doc['rerank_score'] = 0
            elif sort_by == 'citations':
                citations = doc.get("citationCount") or doc.get("citations")
                try:
                    doc['rerank_score'] = int(citations) if citations is not None else 0
                except:
                    doc['rerank_score'] = 0
            else:  # 'similarity'
                doc['rerank_score'] = doc.get(score_name, 0.0)

        logger.info(f"Reranking completed. Top document: {sorted_docs[0].get('title', '') if sorted_docs else 'None'}")
        return sorted_docs

    '''
    def rerank_query_and_doc_list(self,all_docs,user_query,score_name="sim_score"):
        """
        使用 LLM 对 top-K 文档进行重排序。
        重排依据：年份（最新优先）> 引用数（多优先）> 相似度（高优先）。
        """
        logger.info("Reranking documents using LLM with priority: Year > Citations > Similarity")
        self.user_query = user_query
        self.score_name = score_name

        if isinstance(all_docs, dict):
            all_docs = list(all_docs.values())

        if not all_docs:
            logger.warning("No documents to rerank")
            return []

        # 1. 按原始分数排序，取前 K 篇（K=50，可配置）
        TOP_K = 20
        sorted_by_sim = sorted(all_docs, key=lambda x: x.get(score_name, 0), reverse=True)
        top_docs = sorted_by_sim[:TOP_K]

        if not top_docs:
            logger.warning("No top documents to rerank")
            return []

        # 2. 提取时间约束
        time_constraints = self._extract_time_constraints(user_query)

        # 3. 准备提示词（已修改，明确优先级）
        prompt = self._prepare_reranking_prompt(top_docs, time_constraints)

        logger.info(f"LLM Reranking prompt length: {len(prompt)} characters")

        logger.debug(f"prompt: {prompt}")
        try:
            # 4. 调用 LLM 获取重排结果
            reranked_results = self.llm_rerank_documents(prompt)

            # 5. 更新文档分数
            top_docs = self._update_documents_with_reranking(reranked_results, top_docs)

            # 6. 按新的 rerank_score 排序（降序）
            top_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

            logger.info("LLM reranking completed successfully")
        except Exception as e:
            logger.error(f"LLM reranking failed: {traceback.format_exc()}")
            # 失败时回退到原始排序（按相似度）
            return top_docs

        return top_docs
    '''

    def _extract_time_constraints(self, query):
        """
        Extract time constraints from the query string.
        Returns a dictionary with time constraint information.
        """
        time_constraints = {
            "has_time_requirement": False,
            "recency_required": False,
            "specific_timeframe": None,
            "year_limit": None
        }

        # Check for recency indicators
        recency_terms = ["recent", "latest", "newest", "current", "modern", "today", "last year"]
        if any(term in query for term in recency_terms):
            time_constraints["has_time_requirement"] = True
            time_constraints["recency_required"] = True

        # Check for specific timeframes (e.g., "in the last 3 years", "since 2020")
        year_pattern = r"(since|after|from|in the last|within)?\s*(\d{1,2})\s*(year|yr)s?"
        specific_year_pattern = r"(since|after|from)?\s*(20\d{2}|19\d{2})"

        year_match = re.search(year_pattern, query)
        if year_match:
            time_constraints["has_time_requirement"] = True
            time_constraints["specific_timeframe"] = f"last {year_match.group(2)} years"
            time_constraints["year_limit"] = int(year_match.group(2))

        specific_year_match = re.search(specific_year_pattern, query)

        if specific_year_match:
            time_constraints["has_time_requirement"] = True
            time_constraints["specific_timeframe"] = f"since {specific_year_match.group(2)}"
            time_constraints["year_limit"] = int(specific_year_match.group(2))

        logger.info(f"Extracted time constraints: {time_constraints}")
        return time_constraints

    #wsl-710
    def _prepare_reranking_prompt(self, docs, time_constraints):
        """
        Prepare a short prompt for LLM reranking.
        Priority: recency (year) > citations > relevance.
        """
        prompt = f"Rerank these {len(docs)} papers for query: '{self.user_query}'\n"
        prompt += "Score each paper from 0 to 1 based on: (1) Year (newer better), (2) Citation count (more better), (3) Topic relevance.\n"
        prompt += "List papers:\n"
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Unknown")
            year = doc.get("publicationYear", doc.get("year", "unknown"))
            citations = doc.get("citationCount", 0)
            sim = doc.get(self.score_name, 0.0)
            prompt += f"{i}. {title} (Year: {year}, Citations: {citations}, Sim: {sim:.2f})\n"
        prompt += "Output format: Document [index]: [score] - [reason]\n"
        return prompt
    '''
    #wsl-710
    def _prepare_reranking_prompt(self, docs, time_constraints):
        """
        Prepare a prompt for the LLM to rerank documents.
        评分优先级（从高到低）：年份（最新优先）> 引用数（最多优先）> 相似度（最高优先）
        """
        current_year = datetime.now().year

        # ---------- 新的评分说明（替换原有四大因素）----------
        prompt = (
            f"Please rerank the following {len(docs)} academic papers in response to the query: '{self.user_query}'\n\n"
            "IMPORTANT: When assigning new relevance scores, **strictly follow this priority order**:\n"
            "1. **PUBLICATION YEAR**: Prefer the most recent papers. A paper from 2024 should score significantly higher than one from 2010.\n"
            "2. **CITATION COUNT**: Prefer papers with higher citations. More citations indicate greater academic impact.\n"
            "3. **TOPIC RELEVANCE**: How closely the paper's title and abstract match the user's query.\n\n"
            "If the query contains time constraints (e.g., 'recent', 'last 5 years'), give **even more weight** to recency.\n"
            "Your final score must be between 0 and 1, and should clearly reflect this priority.\n\n"
            "For each paper, provide:\n"
            "1. A new numerical rank (1 being the highest)\n"
            "2. A brief justification (1-2 sentences)\n"
            "3. A new relevance score between 0-1 that strictly follows the priority above\n\n"
            "List of papers with original details (title, year, venue, authors, original relevance):\n"
        )

        # ---------- 保留原有文档详情（完全不变）----------
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Unknown Title")
            year = doc.get("year", doc.get("publicationYear", "Unknown Year"))
            venue = doc.get("venue", doc.get("journal", "Unknown Venue"))
            authors = ", ".join([a.get("name", "Unknown") for a in doc.get("authors", [])][:3])
            if len(doc.get("authors", [])) > 3:
                authors += " et al."
            sim_score = doc.get(self.score_name, 0.0)
            age_note = ""
            if year and year != "Unknown Year":
                paper_age = current_year - int(year)
                if time_constraints.get("year_limit"):
                    if paper_age <= time_constraints["year_limit"]:
                        age_note = f" (meets {time_constraints['specific_timeframe']} requirement)"
                    else:
                        age_note = f" (outside {time_constraints['specific_timeframe']} requirement)"
            prompt += f"{i}. {title} ({year}{age_note}) - {venue}\n   Authors: {authors}\n   Original relevance: {sim_score:.3f}\n\n"

        # ---------- 输出格式要求（保持不变）----------
        prompt += "Please provide your reranking with new scores and concise justifications in the following format for each document:\n"
        prompt += "Document [index]: [score] - [justification]\n"
        prompt += "For example:\n"
        prompt += "Document 1: 0.95 - Very recent, highly cited, and directly relevant.\n"
        prompt += "Document 2: 0.70 - Recent but low citations, somewhat relevant.\n"

        return prompt
    '''
    '''
    #wsl-76改了重排序部分按照权威性 → 原始查询相关性 → 时效性 → 可复现性
    def _prepare_reranking_prompt(self, docs, time_constraints):
        """
        Prepare a prompt for the LLM to rerank documents.
        """
        current_year = datetime.now().year

        prompt = (
            f"Please rerank the following {len(docs)} academic papers in response to the query: '{self.user_query}'\n\n"
            "IMPORTANT: When assigning new scores, use the following priority order (highest to lowest):\n"
            "1. AUTHORITY – publication venue prestige and author prominence (most important)\n"
            "2. RELEVANCE – how closely the paper matches the core topic and intent of the query\n"
            "3. TIMELINESS – recency of publication, respecting any time constraints in the query\n"
            "4. REPRODUCIBILITY – availability of code/data (least important, but still considered)\n\n"
            "Consider these factors in detail:\n"
        )

        # 1. Authority
        prompt += (
            "1. Authority:\n"
            "   - Publication venue prestige (top conferences/journals rank higher)\n"
            "   - Author prominence (authors with higher h-index or citation counts rank higher)\n\n"
        )

        # 2. Relevance (直接与原始查询相关)
        prompt += (
            "2. Relevance:\n"
            "   - How closely the paper's content addresses the user's query, including title, abstract, and specific research aspects\n\n"
        )

        # 3. Timeliness
        prompt += "3. Timeliness:\n"
        if time_constraints["has_time_requirement"]:
            if time_constraints["recency_required"]:
                prompt += "   - The query asks for recent/current papers, so strongly prefer newer papers\n"
            if time_constraints["specific_timeframe"]:
                prompt += f"   - The query asks for papers {time_constraints['specific_timeframe']}, so prefer papers within this timeframe\n"
        else:
            prompt += "   - Since this is an academic literature search, favor papers published in the last 3 years unless an older paper is exceptionally foundational (e.g., cited > 5000 times).\n"
        prompt += "\n"

        # 4. Reproducibility
        prompt += (
            "4. Reproducibility & Open Science:\n"
            "   - Prefer papers that provide open-source code, publicly available datasets, or detailed experimental setups.\n"
            "   - If a paper lacks code or data, penalize it unless it is a purely theoretical breakthrough.\n\n"
        )

        # 明确最终评分应综合上述优先级
        prompt += (
            "For each paper, provide:\n"
            "1. A new numerical rank (1 being the highest)\n"
            "2. A brief justification (1-2 sentences)\n"
            "3. A new relevance score between 0-1 that reflects the priority order above (with authority and relevance weighted most heavily)\n\n"
            "List of papers with original relevance scores (title, year, venue, authors, relevance):\n"
        )

        # 添加文档详情（原有代码保持不变）
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Unknown Title")
            year = doc.get("year", doc.get("publicationYear", "Unknown Year"))
            venue = doc.get("venue", doc.get("journal", "Unknown Venue"))
            authors = ", ".join([a.get("name", "Unknown") for a in doc.get("authors", [])][:3])
            if len(doc.get("authors", [])) > 3:
                authors += " et al."
            sim_score = doc.get(self.score_name, 0.0)
            age_note = ""
            if year and year != "Unknown Year":
                paper_age = current_year - int(year)
                if time_constraints.get("year_limit"):
                    if paper_age <= time_constraints["year_limit"]:
                        age_note = f" (meets {time_constraints['specific_timeframe']} requirement)"
                    else:
                        age_note = f" (outside {time_constraints['specific_timeframe']} requirement)"
            prompt += f"{i}. {title} ({year}{age_note}) - {venue}\n   Authors: {authors}\n   Original relevance: {sim_score:.3f}\n\n"

        prompt += "Please provide your reranking with new scores and concise justifications in the following format for each document:\n"
        prompt += "Document [index]: [score] - [justification]\n"
        prompt += "For example:\n"
        prompt += "Document 1: 0.95 - Highly authoritative venue, directly addresses the query topic.\n"
        prompt += "Document 2: 0.70 - Somewhat relevant but from a less prestigious venue.\n"

        return prompt
    '''
    '''
    def _prepare_reranking_prompt(self, docs, time_constraints):
        """
        Prepare a prompt for the LLM to rerank documents.
        """
        current_year = datetime.now().year

        prompt = (
            f"Please rerank the following {len(docs)} academic papers in response to the query: '{self.user_query}'\n\n"
            "Consider these factors in your reranking:\n"
            "1. Authority:\n"
            "   - Publication venue prestige (top conferences/journals rank higher)\n"
            "   - Author prominence (authors with higher h-index or citation counts rank higher)\n"
            "2. Timeliness:\n"
        )

        if time_constraints["has_time_requirement"]:
            if time_constraints["recency_required"]:
                prompt += "   - The query specifically asks for recent/current papers, so strongly prefer newer papers\n"

            if time_constraints["specific_timeframe"]:
                prompt += f"   - The query asks for papers {time_constraints['specific_timeframe']}, so prefer papers within this timeframe\n"
        #else:
        #    prompt += "   - Generally prefer more recent papers, but don't overly penalize influential older papers\n"
        else:  #wsl改-时效性（已还原为原版温和表述）
            prompt += "   - Generally prefer more recent papers, but don't overly penalize influential older papers\n"


        prompt += (
            "3. Maintain reasonable relevance to the original query\n\n"
            "For each paper, provide:\n"
            "1. A new numerical rank (1 being the highest)\n"
            "2. A brief justification (1-2 sentences)\n"
            "3. A new relevance score between 0-1 that incorporates both relevance and the factors above\n\n"
            "List of papers with original relevance scores (title, year, venue, authors, relevance):\n"
        )

        # Add document details to prompt
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Unknown Title")
            year = doc.get("year", doc.get("publicationYear","Unknown Year"))
            venue = doc.get("venue", doc.get("journal", "Unknown Venue"))
            authors = ", ".join([a.get("name", "Unknown") for a in doc.get("authors", [])][:3])
            if len(doc.get("authors", [])) > 3:
                authors += " et al."
            sim_score = doc.get(self.score_name, 0.0)

            # Include timeliness information if relevant
            age_note = ""
            if year and year != "Unknown Year":
                paper_age = current_year - int(year)
                if time_constraints["has_time_requirement"] and time_constraints["year_limit"]:
                    if paper_age <= time_constraints["year_limit"]:
                        age_note = f" (meets {time_constraints['specific_timeframe']} requirement)"
                    else:
                        age_note = f" (outside {time_constraints['specific_timeframe']} requirement)"

            prompt += f"{i}. {title} ({year}{age_note}) - {venue}\n   Authors: {authors}\n   Original relevance: {sim_score:.3f}\n\n"

        prompt += "Please provide your reranking with new scores and concise justifications in the following format for each document:\n"
        prompt += "Document [index]: [score] - [justification]\n"
        prompt += "For example:\n"
        prompt += "Document 1: 0.95 - Highly relevant as it directly addresses the query topic with empirical evidence.\n" #wsl
        prompt += "Document 2: 0.70 - Somewhat relevant but focuses on a tangential aspect of the query.\n"

        return prompt
    '''

    def llm_rerank_documents(self, prompt):
        """
        Use an LLM to rerank documents based on the provided prompt.

        Returns a list of dictionaries containing reranked documents information.
        """
        logger.debug(f"llm_rerank_documents: {prompt}")
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = get_from_llm(prompt, model_name=RERANK_MODEL)
                logger.debug(f"response: {response}")
                # Parse the LLM response to extract reranking information
                reranked_results = self._parse_llm_reranking_response(response)

                # If we got valid results, return them
                if reranked_results:
                    logger.debug(f"reranked_results: {reranked_results}")
                    return reranked_results

                # Otherwise, retry
                logger.warning(f"Empty reranking results received (attempt {retry_count + 1}/{max_retries}). Retrying...")
                retry_count += 1

                # Add a slight modification to the prompt to encourage different response
                if retry_count < max_retries:
                    prompt += f"\n\nPlease ensure you provide a complete reranking for all documents in the exact format requested."

            except Exception as e:
                logger.error(f"LLM reranking failed (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                retry_count += 1

                if retry_count < max_retries:
                    logger.info(f"Retrying reranking...")

        # If we've exhausted all retries, return empty list
        logger.error(f"Failed to get valid reranking results after {max_retries} attempts")
        return []

    def _parse_llm_reranking_response(self, response: str) -> List[Dict[str, Union[float, str]]]:
        """Parse LLM reranking response to extract scores and justifications.

        Args:
            response: LLM response string

        Returns:
            List of dictionaries with 'score' and 'justification' keys
        """
        results = []
        # Look for lines with the pattern "Document X: Y.Z - justification"
        pattern = r"Document\s+(\d+):\s+([\d.]+)\s+-\s+(.+)"

        for line in response.split("\n"):
            match = re.search(pattern, line)
            if match:
                document_idx = int(match.group(1)) - 1  # Convert to zero-based index
                score = float(match.group(2))
                justification = match.group(3).strip()

                # Ensure the document_idx is valid
                while len(results) <= document_idx:
                    results.append({})

                results[document_idx] = {
                    "rerank_score": score,
                    "justification": justification
                }
        # return [r for r in results if r]  #wsl-bug
        return results

    def _update_documents_with_reranking(self, reranked_results, original_docs):  #wsl-返回完整的字段用于node排序
        for idx, result in enumerate(reranked_results):
            if idx < len(original_docs) and result:
                original_docs[idx]["rerank_score"] = result.get("rerank_score")
                original_docs[idx]["rerank_justification"] = result.get("justification", "")
        return original_docs  # 返回包含所有原始字段的完整列表
def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = "".join(letters)
    return result.lower()


def cal_micro(pred_set, label_set):
    if not label_set and not pred_set:
        print("Warning: Both pred_set and label_set are empty.")
        return 0, 0, 0
    if not label_set:
        print("Warning: label_set is empty.")
        return 0, len(pred_set), 0
    if not pred_set:
        return 0, 0, len(label_set)
    tp = len(pred_set & label_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)
    return tp, fp, fn

