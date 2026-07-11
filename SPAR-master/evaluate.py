# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : Auto-generated evaluation script for SPAR
# [Descriptions] : Complete evaluation pipeline for SPAR academic search system.
#                  Supports two benchmark datasets:
#                    - AutoScholarQuery (matches by arxiv_id + title)
#                    - OwnBenchmark/spar_bench (matches by title + paperID)
#                  Computes precision, recall, F1 (micro/macro/@k) and outputs
#                  console report + Excel + visualizations.
# ==================================================================

import json
import os
import sys
import re
import time
import traceback
import argparse
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# ========== 项目模块导入 ==========
try:
    from tqdm import tqdm
except ImportError:
    # 简易 fallback
    def tqdm(iterable, **kwargs):
        return iterable

# ========== 配置 ==========
# 支持常用指标缩写映射
METRIC_ALIASES = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "f1_score": "f1",
    "p": "precision",
    "r": "recall",
}

# 标题归一化缓存
_title_cache = {}

# ====================================================================
#  文本匹配工具
# ====================================================================

def normalize_title(title: str) -> str:
    """归一化论文标题用于模糊匹配"""
    if title in _title_cache:
        return _title_cache[title]
    # 转小写
    t = title.lower().strip()
    # 去除标点符号和多余空格
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    # 去除常见前缀/后缀噪音词
    t = re.sub(r"^(a|an|the)\s+", "", t)
    _title_cache[title] = t
    return t


def title_similarity(t1: str, t2: str) -> float:
    """计算两个标题的文本相似度 (0-1)"""
    return SequenceMatcher(None, normalize_title(t1), normalize_title(t2)).ratio()


def match_title(pred_title: str, gold_titles: List[str], threshold: float = 0.85) -> Optional[str]:
    """
    在 gold 标题列表中匹配 pred_title。
    优先返回精确匹配（归一化后相等），否则按相似度阈值匹配。
    """
    pred_norm = normalize_title(pred_title)
    # 精确匹配
    for gt in gold_titles:
        if normalize_title(gt) == pred_norm:
            return gt
    # 模糊匹配
    best_match = None
    best_score = 0.0
    for gt in gold_titles:
        score = title_similarity(pred_title, gt)
        if score > best_score:
            best_score = score
            best_match = gt
    if best_score >= threshold:
        return best_match
    return None


# ====================================================================
#  论文 ID 提取工具
# ====================================================================

_arxiv_pattern = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")


def extract_arxiv_id(text: str) -> Optional[str]:
    """从字符串中提取 arxiv ID（去除版本号）"""
    m = _arxiv_pattern.search(str(text))
    if m:
        return m.group(1)
    return None


def extract_paper_id(doc: dict) -> str:
    """从论文字典中提取唯一标识符（优先 arxivId）"""
    # 多种 ID 字段
    for key in ("arxivId", "arxiv_id", "paper_id", "id", "paperID", "PMID"):
        val = doc.get(key)
        if val:
            val_str = str(val)
            # 如果是 arxiv ID 格式
            aid = extract_arxiv_id(val_str)
            if aid:
                return aid
            return val_str
    # 从 URL 或其他字段提取
    for key in ("url", "openalex_id", "externalIds"):
        val = doc.get(key)
        if val:
            if isinstance(val, dict):
                for v in val.values():
                    aid = extract_arxiv_id(str(v))
                    if aid:
                        return aid
            aid = extract_arxiv_id(str(val))
            if aid:
                return aid
    return ""


# ====================================================================
#  Ground Truth 加载
# ====================================================================

def load_benchmark(benchmark_name: str, sample_num: int = None) -> List[dict]:
    """
    加载 benchmark 数据集。
    支持 AutoScholarQuery 和 OwnBenchmark。
    返回 list[dict]，每个 dict 包含 question, answer_titles, answer_ids, qid
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.join(base_dir, "benchmark")

    if benchmark_name == "AutoScholarQuery":
        src_file = os.path.join(benchmark_dir, "AutoScholarQuery_test.jsonl")
    elif benchmark_name == "OwnBenchmark":
        src_file = os.path.join(benchmark_dir, "spar_bench.jsonl")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Benchmark file not found: {src_file}")

    records = []
    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            question = data.get("question", data.get("query", ""))
            answer_titles = []
            answer_ids = set()

            if benchmark_name == "AutoScholarQuery":
                # 格式: answer (标题列表), answer_arxiv_id (ID列表)
                answer_titles = data.get("answer", [])
                raw_ids = data.get("answer_arxiv_id", [])
                for aid in raw_ids:
                    aid = str(aid).strip()
                    if aid:
                        # 标准化 arxiv ID
                        extracted = extract_arxiv_id(aid)
                        answer_ids.add(extracted if extracted else aid)
                qid = data.get("qid", f"q_{len(records)}")

            else:  # OwnBenchmark / spar_bench
                # 格式: answer (标题列表), source_meta.answers (详细列表)
                answer_titles = data.get("answer", [])
                answers_meta = data.get("source_meta", {}).get("answers", [])
                for a in answers_meta:
                    pid = a.get("paperID", "")
                    if pid:
                        extracted = extract_arxiv_id(pid)
                        answer_ids.add(extracted if extracted else pid)
                    #wsl-76 新增：提取 OpenAlex ID
                    oa_id = a.get("openalex_id")
                    if oa_id:
                        answer_ids.add(str(oa_id))
                    else:
                        oa_url = a.get("openAlexUrl", "")
                        if oa_url:
                            match = re.search(r"/(W\d+)$", oa_url)
                            if match:
                                answer_ids.add(match.group(1))
                    # 补充：从 URL 提取
                    if not pid:
                        for k in ("url", "openAlexUrl", "semanticUrl"):
                            v = a.get(k, "")
                            if v:
                                extracted = extract_arxiv_id(v)
                                if extracted:
                                    answer_ids.add(extracted)
                qid = data.get("qid", f"q_{len(records)}")

            records.append({
                "question": question,
                "answer_titles": answer_titles,
                "answer_ids": answer_ids,
                "qid": qid,
                "raw": data,
            })

    # 采样
    if sample_num and sample_num < len(records):
        import random
        random.seed(123)
        records = random.sample(records, sample_num)

    print(f"[数据集] {benchmark_name}: 共 {len(records)} 条查询")
    return records


# ====================================================================
#  预测结果提取（从搜索树结果 JSON 中）
# ====================================================================

def extract_predicted_ids(result_json: dict) -> Tuple[List[dict], Set[str]]:
    """
    从搜索树结果 JSON 中提取所有预测到的论文。
    返回 (all_docs, unique_ids)，
    其中 unique_ids 是论文去重后的 ID 集合（用于计算 metrics）。
    """
    all_docs = []
    seen_ids = set()

    def _collect(node_dict: dict):
        # 收集相关文档 (docs)
        for doc in node_dict.get("docs", []):
            doc_id = extract_paper_id(doc)
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
            elif not doc_id and doc.get("title"):
                # 无 ID 但有标题，按标题去重
                title = doc.get("title", "")
                if title and title not in seen_ids:
                    seen_ids.add(title)
                    all_docs.append(doc)

        # 也收集搜索树中缓存的 searched_docs
        extra = node_dict.get("extra", {})
        searched = extra.get("searched_docs", {})
        for doc_id, doc in searched.items():
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

        # 递归收集子节点
        for child in node_dict.get("children", []):
            _collect(child)

    _collect(result_json)
    return all_docs, seen_ids


def load_result_file(result_path: str) -> Optional[dict]:
    """加载单条搜索结果的 JSON 文件"""
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [警告] 无法加载结果文件 {result_path}: {e}")
        return None


def find_result_files(result_dir: str) -> Dict[str, str]:
    """
    扫描结果目录，返回 {question_md5: filepath} 映射。
    结果文件的命名格式为 <md5>.json。
    """
    files = {}
    for fname in os.listdir(result_dir):
        if fname.endswith(".json") and len(fname) == 36 + 5:  # md5(32).json(5)
            fpath = os.path.join(result_dir, fname)
            files[fname[:32]] = fpath
    return files


# ====================================================================
#  评估指标计算
# ====================================================================

def compute_metrics_tp_fp_fn(pred_ids: Set[str], gold_ids: Set[str]) -> Tuple[int, int, int]:
    """
    计算 TP, FP, FN。
    优先使用 ID 匹配，ID 为空时尝试标题匹配返回 None（此 query 跳过）。
    """
    if not gold_ids:
        return 0, len(pred_ids), 0  # 无 ground truth，全部 FP
    if not pred_ids:
        return 0, 0, len(gold_ids)  # 无预测，全部 FN

    tp = len(pred_ids & gold_ids)
    fp = len(pred_ids - gold_ids)
    fn = len(gold_ids - pred_ids)
    return tp, fp, fn


def calc_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """从 TP/FP/FN 计算精确率、召回率、F1"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_query(
    pred_ids: Set[str],
    gold_ids: Set[str],
    pred_titles: List[str],
    gold_titles: List[str],
    k_values: List[int] = None,
) -> dict:
    """
    对单条查询计算各类评估指标。
    支持 ID 匹配和标题回退匹配。
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    # ---- ID 匹配 ----
    tp_ids, fp_ids, fn_ids = compute_metrics_tp_fp_fn(pred_ids, gold_ids)
    precision_ids, recall_ids, f1_ids = calc_precision_recall_f1(tp_ids, fp_ids, fn_ids)

    # ---- 标题回退匹配（对没有 ID 的预测做补充） ----
    # 找出 gold 中无 ID 的标题
    gold_without_id = set()
    for gt in gold_titles:
        matched = False
        for gid in gold_ids:
            if gid.lower() in gt.lower() or gt.lower() in gid.lower():
                matched = True
                break
        if not matched:
            gold_without_id.add(gt)

    # 标题匹配增强：用标题去匹配 gold 中无 ID 的部分
    title_tp = 0
    title_fp_extra = 0
    title_matched_gold = set()

    for pred_title in pred_titles:
        match = match_title(pred_title, list(gold_titles), threshold=0.85)
        if match:
            # 检查这个 gold title 是否已经被 ID 匹配覆盖了
            gt_idx = gold_titles.index(match)
            if gt_idx not in [i for i, gt in enumerate(gold_titles) if gt in gold_without_id]:
                # 已被 ID 匹配覆盖
                pass
            elif match not in title_matched_gold:
                title_tp += 1
                title_matched_gold.add(match)
        else:
            # 检查这个预测标题是否对应 gold 中已有的 ID
            already_in_gold = False
            for gt in gold_titles:
                if title_similarity(pred_title, gt) > 0.9:
                    gt_idx_in_gold = gold_titles.index(gt)
                    if gt_idx_in_gold not in [i for i, gt in enumerate(gold_titles) if gt in gold_without_id]:
                        already_in_gold = True
                        break
            if not already_in_gold:
                title_fp_extra += 1

    # 合并 ID 和标题匹配结果
    combined_tp = tp_ids + title_tp
    combined_fn = fn_ids + max(0, len(gold_without_id) - title_tp)
    combined_fp = fp_ids + title_fp_extra

    precision_combined, recall_combined, f1_combined = calc_precision_recall_f1(
        combined_tp, combined_fp, combined_fn
    )

    # ---- @k 评估 ----
    at_k = {}
    for k in k_values:
        k_pred = set(list(pred_ids)[:k]) if pred_ids else set()
        k_tp = len(k_pred & gold_ids)
        k_precision = k_tp / k if k > 0 else 0.0
        k_recall = k_tp / len(gold_ids) if gold_ids else 0.0
        k_f1 = (
            2 * k_precision * k_recall / (k_precision + k_recall)
            if (k_precision + k_recall) > 0
            else 0.0
        )
        at_k[k] = {
            "tp": k_tp,
            "precision": k_precision,
            "recall": k_recall,
            "f1": k_f1,
        }

    return {
        "tp": combined_tp,
        "fp": combined_fp,
        "fn": combined_fn,
        "tp_ids": tp_ids,
        "fp_ids": fp_ids,
        "fn_ids": fn_ids,
        "precision_ids": precision_ids,
        "recall_ids": recall_ids,
        "f1_ids": f1_ids,
        "precision": precision_combined,
        "recall": recall_combined,
        "f1": f1_combined,
        "num_gold": len(gold_ids | gold_without_id),  # ground truth 总数
        "num_pred": len(pred_ids) + title_fp_extra,    # 预测总数
        "at_k": at_k,
    }


# ====================================================================
#  运行完整搜索 + 评估
# ====================================================================

def run_search_and_evaluate(
    benchmark_name: str,
    sample_num: int = 50,
    max_depth: int = 2,
    score_thresh: float = 0.5,
    relevance_doc_num: int = 10,
    output_dir: str = None,
    skip_existing: bool = True,
) -> Tuple[List[dict], str]:
    """
    运行搜索并对 benchmark 进行评估。
    返回 (per_query_results, output_folder_path)。
    """
    # 导入搜索模块（延迟导入，避免无 API key 时直接报错）
    from global_config import SEARCH_ROUTES, LLM_MODEL_NAME
    from pipeline_spar import AcademicSearchTree
    from utils import get_md5

    # 构建输出目录
    if output_dir is None:
        routes_str = "-".join(SEARCH_ROUTES)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./gen_result/{benchmark_name}_{sample_num}_{routes_str}_depth{max_depth}_sim{score_thresh}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    records = load_benchmark(benchmark_name, sample_num=sample_num)

    # 创建搜索 agent
    search_agent = AcademicSearchTree(
        max_depth=max_depth,
        max_docs=relevance_doc_num,
        similarity_threshold=score_thresh,
    )

    # 扫描已有结果
    existing_results = {}
    if skip_existing:
        for fname in os.listdir(output_dir):
            if fname.endswith(".json") and len(fname) > 32:
                q_hash = fname[:32]
                existing_results[q_hash] = os.path.join(output_dir, fname)

    # 运行搜索
    per_query_results = []
    for i, record in enumerate(tqdm(records, desc="搜索与评估")):
        question = record["question"]
        q_hash = get_md5(question)

        result_data = None

        # 检查是否已有缓存结果
        result_path = existing_results.get(q_hash)
        if result_path and os.path.exists(result_path):
            result_data = load_result_file(result_path)
            print(f"  [使用缓存] {i+1}/{len(records)}: {question[:60]}...")
        else:
            print(f"\n  [搜索] {i+1}/{len(records)}: {question[:60]}...")
            try:
                sorted_docs = search_agent.search(question, end_date="")
                result_data = search_agent.root.convert_to_dict()
                # 保存结果
                dest_file = os.path.join(output_dir, f"{q_hash}.json")
                with open(dest_file, "w", encoding="utf-8") as fw:
                    json.dump(result_data, fw, indent=2, ensure_ascii=False)
                print(f"  [已保存] {dest_file}")
            except Exception as e:
                print(f"  [错误] {question[:60]} 搜索失败: {e}")
                traceback.print_exc()
                continue

        if not result_data:
            print(f"  [跳过] {question[:60]} 无结果数据")
            continue

        # 提取预测结果
        all_docs, pred_ids = extract_predicted_ids(result_data)
        pred_titles = [d.get("title", "") for d in all_docs if d.get("title")]
        #wsl-76 ---- 插入过滤代码 ----
        threshold = 0.4  # 可以调低，因为重排序后的分数可能更分散
        filtered_pred_ids = set()
        filtered_pred_titles = []
        for doc in all_docs:
            # 优先使用 rerank_score_bge，若不存在则用 sim_score
            score = doc.get("rerank_score_bge", doc.get("sim_score", 0.0))
            if score >= threshold:
                doc_id = doc.get("paper_id", "")
                if doc_id:
                    filtered_pred_ids.add(doc_id)
                    title = doc.get("title", "")
                    if title:
                        filtered_pred_titles.append(title)
        pred_ids = filtered_pred_ids
        pred_titles = filtered_pred_titles
        # 评估
        eval_result = evaluate_query(
            pred_ids=pred_ids,
            gold_ids=record["answer_ids"],
            pred_titles=pred_titles,
            gold_titles=record["answer_titles"],
        )

        per_query_results.append({
            "qid": record["qid"],
            "question": question,
            "num_gold": record["answer_ids"],
            "num_pred": pred_ids,
            "eval": eval_result,
        })

        # 打印单条结果
        eval_str = eval_result
        score_str = (
            f"P={eval_str['precision']:.3f} R={eval_str['recall']:.3f} "
            f"F1={eval_str['f1']:.3f}  "
            f"TP={eval_str['tp']} FP={eval_str['fp']} FN={eval_str['fn']}"
        )
        print(f"  [结果] {score_str}")

    return per_query_results, output_dir


# ====================================================================
#  加载已有结果进行评估（无需重新搜索）
# ====================================================================

def evaluate_existing_results(
    benchmark_name: str,
    result_dir: str,
    sample_num: int = None,
) -> Tuple[List[dict], str]:
    """
    加载已有搜索结果进行重新评估。
    result_dir: run_spr_agent.py 生成的结果目录。
    """
    records = load_benchmark(benchmark_name, sample_num=sample_num)
    from utils import get_md5

    # 构建 question -> result 映射
    question_to_result = {}
    for fname in os.listdir(result_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(result_dir, fname)
            result = load_result_file(fpath)
            if result:
                q = result.get("search_query", "")
                if q:
                    question_to_result[q] = result

    print(f"[加载] 在 {result_dir} 中找到 {len(question_to_result)} 条搜索结果")

    # 逐条评估
    per_query_results = []
    matched = 0
    for record in tqdm(records, desc="评估"):
        question = record["question"]
        result_data = question_to_result.get(question)

        if result_data is None:
            # 尝试用 MD5 匹配
            q_hash = get_md5(question)
            result_path = os.path.join(result_dir, f"{q_hash}.json")
            if os.path.exists(result_path):
                result_data = load_result_file(result_path)

        if result_data is None:
            continue

        matched += 1
        all_docs, pred_ids = extract_predicted_ids(result_data)
        pred_titles = [d.get("title", "") for d in all_docs if d.get("title")]

        #wsl-76 ---- 插入过滤代码 ----
        threshold = 0.4  # 可以调低，因为重排序后的分数可能更分散
        filtered_pred_ids = set()
        filtered_pred_titles = []
        for doc in all_docs:
            # 优先使用 rerank_score_bge，若不存在则用 sim_score
            score = doc.get("rerank_score_bge", doc.get("sim_score", 0.0))
            if score >= threshold:
                doc_id = doc.get("paper_id", "")
                if doc_id:
                    filtered_pred_ids.add(doc_id)
                    title = doc.get("title", "")
                    if title:
                        filtered_pred_titles.append(title)
        pred_ids = filtered_pred_ids
        pred_titles = filtered_pred_titles

        eval_result = evaluate_query(
            pred_ids=pred_ids,
            gold_ids=record["answer_ids"],
            pred_titles=pred_titles,
            gold_titles=record["answer_titles"],
        )

        per_query_results.append({
            "qid": record["qid"],
            "question": question,
            "num_gold": len(record["answer_ids"]),
            "num_pred": len(pred_ids),
            "eval": eval_result,
        })

    print(f"[匹配] 成功匹配 {matched}/{len(records)} 条查询的结果")
    return per_query_results, result_dir


# ====================================================================
#  汇总统计与输出
# ====================================================================

def aggregate_metrics(per_query_results: List[dict]) -> dict:
    """
    汇总所有查询的评估结果，计算 micro/macro 平均和 @k 指标。
    """
    if not per_query_results:
        return {"error": "No results to aggregate"}

    # ---- Micro 平均（全局 TP/FP/FN 累加） ----
    total_tp = sum(r["eval"]["tp_ids"] for r in per_query_results)
    total_fp = sum(r["eval"]["fp_ids"] for r in per_query_results)
    total_fn = sum(r["eval"]["fn_ids"] for r in per_query_results)
    micro_precision, micro_recall, micro_f1 = calc_precision_recall_f1(total_tp, total_fp, total_fn)

    # ---- Macro 平均（每条查询独立算指标再平均） ----
    macro_precision_list = [r["eval"]["precision_ids"] for r in per_query_results]
    macro_recall_list = [r["eval"]["recall_ids"] for r in per_query_results]
    macro_f1_list = [r["eval"]["f1_ids"] for r in per_query_results]

    # 过滤掉无效值（全是 0 的查询）
    valid_p = [v for v in macro_precision_list if v > 0 or True]
    valid_r = [v for v in macro_recall_list if v > 0 or True]
    valid_f = [v for v in macro_f1_list if v > 0 or True]

    macro_precision = np.mean(macro_precision_list) if macro_precision_list else 0.0
    macro_recall = np.mean(macro_recall_list) if macro_recall_list else 0.0
    macro_f1 = np.mean(macro_f1_list) if macro_f1_list else 0.0

    # ---- 标题增强后的指标 ----
    total_tp_combined = sum(r["eval"]["tp"] for r in per_query_results)
    total_fp_combined = sum(r["eval"]["fp"] for r in per_query_results)
    total_fn_combined = sum(r["eval"]["fn"] for r in per_query_results)
    combined_precision, combined_recall, combined_f1 = calc_precision_recall_f1(
        total_tp_combined, total_fp_combined, total_fn_combined
    )

    # ---- @k 指标汇总 ----
    k_values = sorted(per_query_results[0]["eval"]["at_k"].keys()) if per_query_results else []
    at_k_summary = {}
    for k in k_values:
        p_at_k = np.mean([r["eval"]["at_k"][k]["precision"] for r in per_query_results])
        r_at_k = np.mean([r["eval"]["at_k"][k]["recall"] for r in per_query_results])
        f1_at_k = np.mean([r["eval"]["at_k"][k]["f1"] for r in per_query_results])
        at_k_summary[k] = {
            "precision": p_at_k,
            "recall": r_at_k,
            "f1": f1_at_k,
        }

    # ---- 统计分布 ----
    precisions = [r["eval"]["precision_ids"] for r in per_query_results]
    recalls = [r["eval"]["recall_ids"] for r in per_query_results]
    f1s = [r["eval"]["f1_ids"] for r in per_query_results]

    stats = {
        "num_queries": len(per_query_results),
        "num_queries_with_results": sum(1 for r in per_query_results if r["eval"]["tp"] > 0),
        "num_queries_empty_gold": sum(1 for r in per_query_results if r["num_gold"] == 0),
        "num_queries_empty_pred": sum(1 for r in per_query_results if r["num_pred"] == 0),

        # Micro 平均
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,

        # 标题增强合并
        "combined_tp": total_tp_combined,
        "combined_fp": total_fp_combined,
        "combined_fn": total_fn_combined,
        "combined_precision": combined_precision,
        "combined_recall": combined_recall,
        "combined_f1": combined_f1,

        # Macro 平均
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_precision_std": np.std(precisions) if precisions else 0.0,
        "macro_recall_std": np.std(recalls) if recalls else 0.0,
        "macro_f1_std": np.std(f1s) if f1s else 0.0,

        # 分布
        "precision_list": precisions,
        "recall_list": recalls,
        "f1_list": f1s,

        # @k
        "at_k": at_k_summary,

        # 每查询详情
        "per_query": [
            {
                "qid": r["qid"],
                "question": r["question"][:80],
                "num_gold": r["num_gold"] if isinstance(r["num_gold"], int) else len(r["num_gold"]),
                "num_pred": r["num_pred"] if isinstance(r["num_pred"], int) else len(r["num_pred"]),
                "tp": r["eval"]["tp_ids"],
                "fp": r["eval"]["fp_ids"],
                "fn": r["eval"]["fn_ids"],
                "precision": r["eval"]["precision_ids"],
                "recall": r["eval"]["recall_ids"],
                "f1": r["eval"]["f1_ids"],
            }
            for r in per_query_results
        ],
    }

    return stats


def print_report(stats: dict, title: str = "评估报告"):
    """在控制台打印格式化的评估报告"""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    # 基本统计
    print(f"\n[统计] 基本统计:")
    print(f"  查询总数:          {stats['num_queries']}")
    print(f"  有结果查询数:      {stats['num_queries_with_results']}")
    print(f"  无结果查询数:      {stats['num_queries_empty_pred']}")
    if stats['num_queries_empty_gold'] > 0:
        print(f"  空 ground truth 数: {stats['num_queries_empty_gold']}")

    # ID 匹配
    print(f"\n[ID匹配] ID 匹配评估:")
    print(f"  TP={stats['total_tp']}  FP={stats['total_fp']}  FN={stats['total_fn']}")
    print(f"  Micro 精确率 (Precision): {stats['micro_precision']:.4f}")
    print(f"  Micro 召回率 (Recall):    {stats['micro_recall']:.4f}")
    print(f"  Micro F1 分数:            {stats['micro_f1']:.4f}")
    print(f"  ------------------------------")
    print(f"  Macro 精确率 (Precision): {stats['macro_precision']:.4f} (+-{stats['macro_precision_std']:.4f})")
    print(f"  Macro 召回率 (Recall):    {stats['macro_recall']:.4f} (+-{stats['macro_recall_std']:.4f})")
    print(f"  Macro F1 分数:            {stats['macro_f1']:.4f} (+-{stats['macro_f1_std']:.4f})")

    # 标题增强合并
    if stats.get("combined_tp", 0) != stats["total_tp"]:
        print(f"\n[标题合并] ID+标题合并评估:")
        print(f"  TP={stats['combined_tp']}  FP={stats['combined_fp']}  FN={stats['combined_fn']}")
        print(f"  Micro 精确率 (Precision): {stats['combined_precision']:.4f}")
        print(f"  Micro 召回率 (Recall):    {stats['combined_recall']:.4f}")
        print(f"  Micro F1 分数:            {stats['combined_f1']:.4f}")

    # @k 评估
    if stats.get("at_k"):
        print(f"\n[@k] @k 评估 (Macro 平均):")
        print(f"  {'k':>5}  {'P@k':>8}  {'R@k':>8}  {'F1@k':>8}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
        for k, v in sorted(stats["at_k"].items()):
            print(f"  {k:>5}  {v['precision']:.4f}  {v['recall']:.4f}  {v['f1']:.4f}")

    # 高分 / 低分查询
    per_query = stats.get("per_query", [])
    if per_query:
        scored = [(q["f1"], q["qid"], q["question"], q["precision"], q["recall"]) for q in per_query]
        scored.sort(reverse=True)
        print(f"\n[Top] Top-5 最佳查询 (按 F1):")
        for f1, qid, q, p, r in scored[:5]:
            print(f"  F1={f1:.3f}  P={p:.3f}  R={r:.3f}  [{qid}] {q[:60]}")

        print(f"\n[Bottom] Bottom-5 最差查询 (按 F1):")
        for f1, qid, q, p, r in scored[-5:]:
            print(f"  F1={f1:.3f}  P={p:.3f}  R={r:.3f}  [{qid}] {q[:60]}")

    # 分数分布
    f1_list = stats.get("f1_list", [])
    if f1_list:
        f1_arr = np.array(f1_list)
        print(f"\n[分布] F1 分数分布:")
        print(f"  均值={np.mean(f1_arr):.4f}  中位数={np.median(f1_arr):.4f}")
        print(f"  标准差={np.std(f1_arr):.4f}  最小值={np.min(f1_arr):.4f}  最大值={np.max(f1_arr):.4f}")
        print(f"  F1=0 的查询数: {np.sum(f1_arr == 0)}/{len(f1_arr)}")
        print(f"  F1>0.5 的查询数: {np.sum(f1_arr > 0.5)}/{len(f1_arr)}")

    print(f"\n{sep}\n")


def save_results_to_excel(stats: dict, output_path: str, benchmark_name: str):
    """将评估结果保存为格式化的 Excel 文件"""
    from utils import save_to_excel

    # 构建主数据 DataFrame
    per_query = stats.get("per_query", [])
    if not per_query:
        print("[警告] 无数据可保存到 Excel")
        return

    df_data = []
    for q in per_query:
        df_data.append({
            "Model Name": f"SPAR_{benchmark_name}",
            "Score Threshold": 0.5,
            "F1": q["f1"],
            "Recall After Sim Filter": q["recall"],
            "Precision": q["precision"],
            "Recall Raw Doc num mean": q["recall"],
            "Recall After Filter Doc num mean": q["recall"],
            "TP": q["tp"],
            "FP": q["fp"],
            "FN": q["fn"],
            "Num Gold": q["num_gold"],
            "Num Pred": q["num_pred"],
            "QID": q["qid"],
            "Question": q["question"],
            "DESCRIB": f"SPAR evaluation on {benchmark_name}",
        })

    df_new = pd.DataFrame(df_data)

    # 添加汇总行
    summary_row = {
        "Model Name": f"SPAR_{benchmark_name}",
        "Score Threshold": "AGGREGATE",
        "F1": stats["micro_f1"],
        "Recall After Sim Filter": stats["micro_recall"],
        "Precision": stats["micro_precision"],
        "Recall Raw Doc num mean": stats["macro_recall"],
        "Recall After Filter Doc num mean": stats["micro_recall"],
        "TP": stats["total_tp"],
        "FP": stats["total_fp"],
        "FN": stats["total_fn"],
        "Num Gold": "-",
        "Num Pred": "-",
        "QID": "MICRO_AVG",
        "Question": "Micro Average (global TP/FP/FN)",
        "DESCRIB": f"Micro average over {stats['num_queries']} queries",
    }
    # 使用 loc 插入汇总行
    df_new = pd.concat([pd.DataFrame([summary_row]), df_new], ignore_index=True)

    # 保存
    sheet_name = f"{benchmark_name}_eval"
    save_to_excel(df_new, output_path, sheet_name)
    print(f"[Excel] 已保存到 {output_path}")


def draw_metric_distributions(stats: dict, output_dir: str, benchmark_name: str):
    """绘制指标分布图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[绘图] matplotlib/seaborn 未安装，跳过绘图")
        return

    # 创建输出目录
    img_dir = os.path.join(output_dir, "metrics", "img")
    os.makedirs(img_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"SPAR Evaluation on {benchmark_name}\n{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 fontsize=14, fontweight="bold")

    # 1. F1 直方图
    ax1 = axes[0, 0]
    f1_list = stats.get("f1_list", [])
    if f1_list:
        ax1.hist(f1_list, bins=20, range=(0, 1), color="steelblue", edgecolor="white", alpha=0.8)
        ax1.axvline(np.mean(f1_list), color="red", linestyle="--", label=f'Mean={np.mean(f1_list):.3f}')
        ax1.axvline(np.median(f1_list), color="green", linestyle="--", label=f'Median={np.median(f1_list):.3f}')
        ax1.set_title("F1 Score Distribution")
        ax1.set_xlabel("F1 Score")
        ax1.set_ylabel("Query Count")
        ax1.legend()
        ax1.set_xlim(0, 1)

    # 2. Precision vs Recall 散点图
    ax2 = axes[0, 1]
    precisions = stats.get("precision_list", [])
    recalls = stats.get("recall_list", [])
    if precisions and recalls:
        ax2.scatter(recalls, precisions, alpha=0.5, c="steelblue", edgecolors="white", s=30)
        ax2.axhline(stats["micro_precision"], color="red", linestyle="--", alpha=0.7,
                    label=f'Micro P={stats["micro_precision"]:.3f}')
        ax2.axvline(stats["micro_recall"], color="green", linestyle="--", alpha=0.7,
                    label=f'Micro R={stats["micro_recall"]:.3f}')
        ax2.set_title("Precision vs Recall (per query)")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.legend()
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)

    # 3. @k 曲线
    ax3 = axes[1, 0]
    at_k = stats.get("at_k", {})
    if at_k:
        ks = sorted(at_k.keys())
        p_ks = [at_k[k]["precision"] for k in ks]
        r_ks = [at_k[k]["recall"] for k in ks]
        f1_ks = [at_k[k]["f1"] for k in ks]
        ax3.plot(ks, p_ks, "o-", label="Precision@k", color="steelblue")
        ax3.plot(ks, r_ks, "s-", label="Recall@k", color="green")
        ax3.plot(ks, f1_ks, "^-", label="F1@k", color="red")
        ax3.set_title("Metrics@k")
        ax3.set_xlabel("k")
        ax3.set_ylabel("Score")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(ks)

    # 4. 汇总指标柱状图
    ax4 = axes[1, 1]
    metrics_names = ["Micro\nPrecision", "Micro\nRecall", "Micro\nF1",
                     "Macro\nPrecision", "Macro\nRecall", "Macro\nF1"]
    metrics_values = [
        stats["micro_precision"], stats["micro_recall"], stats["micro_f1"],
        stats["macro_precision"], stats["macro_recall"], stats["macro_f1"],
    ]
    colors = ["steelblue", "steelblue", "steelblue", "coral", "coral", "coral"]
    bars = ax4.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor="white")
    for bar, val in zip(bars, metrics_values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax4.set_title("Summary Metrics")
    ax4.set_ylabel("Score")
    ax4.set_ylim(0, 1.1)
    ax4.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="0.5 threshold")
    ax4.legend()

    plt.tight_layout()
    img_path = os.path.join(img_dir, f"evaluation_{benchmark_name}.png")
    plt.savefig(img_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[图表] 已保存到 {img_path}")


# ====================================================================
#  主入口
# ====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SPAR 学术搜索系统评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 全新搜索 + 评估（AutoScholarQuery，50条样本）
  python evaluate.py --benchmark AutoScholarQuery --mode full --sample 50

  # 全新搜索 + 评估（OwnBenchmark）
  python evaluate.py --benchmark OwnBenchmark --mode full --sample 20

  # 评估已有运行结果（不重新搜索）
  python evaluate.py --benchmark AutoScholarQuery --mode load \\
    --result_dir ./gen_result/AutoScholarQuery_50_.../

  # 评估默认 run_spr_agent.py 的输出
  python evaluate.py --benchmark AutoScholarQuery --mode load \\
    --result_dir ./gen_result/AutoScholarQuery_2000_.../

  # 快捷运行（默认参数，50条样本）
  python evaluate.py
        """,
    )
    parser.add_argument("--benchmark", type=str, default="AutoScholarQuery",
                        choices=["AutoScholarQuery", "OwnBenchmark"],
                        help="Benchmark 数据集名称 (default: AutoScholarQuery)")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "load"],
                        help="运行模式: full=搜索+评估, load=仅评估已有结果 (default: full)")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="load 模式下的结果目录路径 (run_spr_agent.py 的输出)")
    parser.add_argument("--sample", type=int, default=None,
                        help="采样数量 (默认全部)")
    parser.add_argument("--depth", type=int, default=2,
                        help="搜索树深度 (default: 2)")
    parser.add_argument("--score_thresh", type=float, default=0.5,
                        help="相关性分数阈值 (default: 0.5)")
    parser.add_argument("--relevance_docs", type=int, default=10,
                        help="每查询目标相关论文数 (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认自动生成)")
    parser.add_argument("--no_chart", action="store_true",
                        help="跳过生成图表")
    parser.add_argument("--title_match", action="store_true",
                        help="启用标题模糊匹配增强（ID 匹配的基础上）")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  SPAR 学术搜索系统评估")
    print("=" * 70)
    print(f"  数据集:   {args.benchmark}")
    print(f"  模式:     {'全新搜索+评估' if args.mode == 'full' else '加载已有结果'}")
    print(f"  采样:     {'全部' if args.sample is None else f'{args.sample} 条'}")
    print("=" * 70)

    # === 运行评估 ===
    start_time = time.time()

    if args.mode == "full":
        per_query_results, output_dir = run_search_and_evaluate(
            benchmark_name=args.benchmark,
            sample_num=args.sample,
            max_depth=args.depth,
            score_thresh=args.score_thresh,
            relevance_doc_num=args.relevance_docs,
            output_dir=args.output,
        )
    else:  # load mode
        if not args.result_dir:
            print("[错误] load 模式需要 --result_dir 参数")
            sys.exit(1)
        if not os.path.isdir(args.result_dir):
            print(f"[错误] 结果目录不存在: {args.result_dir}")
            sys.exit(1)
        per_query_results, output_dir = evaluate_existing_results(
            benchmark_name=args.benchmark,
            result_dir=args.result_dir,
            sample_num=args.sample,
        )

    if not per_query_results:
        print("[错误] 没有有效的评估结果")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n[评估完成] 耗时: {elapsed:.1f}s")

    # === 汇总与报告 ===
    stats = aggregate_metrics(per_query_results)

    # 控制台报告
    print_report(stats, f"SPAR Evaluation on {args.benchmark}")

    # === 保存 Excel ===
    excel_path = os.path.join(output_dir, f"evaluation_{args.benchmark}.xlsx")
    try:
        save_results_to_excel(stats, excel_path, args.benchmark)
    except Exception as e:
        print(f"[Excel 保存失败] {e}")

    # === 绘制图表 ===
    if not args.no_chart:
        try:
            draw_metric_distributions(stats, output_dir, args.benchmark)
        except Exception as e:
            print(f"[绘图失败] {e}")

    # === 输出 JSON 报告 ===
    report_json_path = os.path.join(output_dir, f"evaluation_{args.benchmark}.json")
    # 构建可序列化的报告
    serializable_stats = {
        "benchmark": args.benchmark,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "num_queries": stats["num_queries"],
        "micro_precision": stats["micro_precision"],
        "micro_recall": stats["micro_recall"],
        "micro_f1": stats["micro_f1"],
        "macro_precision": stats["macro_precision"],
        "macro_recall": stats["macro_recall"],
        "macro_f1": stats["macro_f1"],
        "total_tp": stats["total_tp"],
        "total_fp": stats["total_fp"],
        "total_fn": stats["total_fn"],
        "combined_precision": stats.get("combined_precision", stats["micro_precision"]),
        "combined_recall": stats.get("combined_recall", stats["micro_recall"]),
        "combined_f1": stats.get("combined_f1", stats["micro_f1"]),
        "at_k": {str(k): v for k, v in stats.get("at_k", {}).items()},
        "config": {
            "depth": args.depth,
            "score_thresh": args.score_thresh,
            "relevance_docs": args.relevance_docs,
            "sample": args.sample,
            "mode": args.mode,
        },
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    print(f"[报告] JSON 报告已保存到 {report_json_path}")

    print(f"\n📁 所有输出保存在: {output_dir}")
    print("=" * 70)

    # 返回主要指标供调用
    return stats


if __name__ == "__main__":
    main()
