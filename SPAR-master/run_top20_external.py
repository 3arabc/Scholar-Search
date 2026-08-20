#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Descriptions] : 临时驱动脚本：用当前 SPAR 系统跑外部数据集前 20 题
#                  数据集: C:\Users\lenovo\Desktop\人工智能创新大赛\Datasets\PasaDataset\AutoScholarQuery\test.jsonl
#                  复用 evaluate.py 的评估/汇总/报告逻辑，但按顺序取前 20 条（非随机采样）
# ==================================================================

import json
import os
import sys
import time
import re
import traceback
from datetime import datetime

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from global_config import SEARCH_ROUTES
from pipeline_spar import AcademicSearchTree
from utils import get_md5

# 复用 evaluate.py 的评估函数
from evaluate import (
    extract_predicted_ids,
    evaluate_query,
    aggregate_metrics,
    print_report,
    save_results_to_excel,
    draw_metric_distributions,
)

DATASET_PATH = r"C:\Users\lenovo\Desktop\人工智能创新大赛\Datasets\PasaDataset\AutoScholarQuery\test.jsonl"
N_TOP = 20

# ==================================================================
# 运行日志输出到 eval_log/ 目录（避免堆在项目根目录）
# ==================================================================
LOG_DIR = os.path.join(BASE_DIR, "eval_log")
os.makedirs(LOG_DIR, exist_ok=True)


def setup_run_log():
    """将 print / logger 输出重定向到 eval_log/run_topN_<时间戳>.log"""
    import logging

    log_path = os.path.join(
        LOG_DIR,
        f"run_top{N_TOP}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    stream = open(log_path, "w", encoding="utf-8")
    sys.stdout = stream
    sys.stderr = stream
    # 同步更新已绑定 sys.stdout 的 logger StreamHandler（log.py 中创建）
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler):
            h.setStream(stream)
    print(f"[运行日志] {log_path}")
    return log_path

# 与 evaluate.py 的搜索参数保持一致
MAX_DEPTH = 2
SCORE_THRESH = 0.5
RELEVANCE_DOC_NUM = 10
FILTER_THRESHOLD = 0.4  # 预测结果过滤阈值（与 evaluate.py 一致）

# 标题归一化（用于匹配，与 evaluate.py 一致）
from evaluate import normalize_title, title_similarity, extract_arxiv_id


def load_top_n_records(dataset_path: str, n: int):
    """从外部数据集文件按顺序读取前 n 条，构造与 evaluate.load_benchmark 相同格式的记录"""
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            question = data.get("question", data.get("query", ""))
            answer_titles = data.get("answer", [])
            answer_ids = set()
            raw_ids = data.get("answer_arxiv_id", [])
            for aid in raw_ids:
                aid = str(aid).strip()
                if aid:
                    extracted = extract_arxiv_id(aid)
                    clean_id = extracted if extracted else aid
                    answer_ids.add(clean_id)
            qid = data.get("qid", f"q_{len(records)}")
            records.append({
                "question": question,
                "answer_titles": answer_titles,
                "answer_ids": answer_ids,
                "gold_id_to_title": {},
                "qid": qid,
                "raw": data,
            })
    print(f"[数据集] 外部文件 {dataset_path}")
    print(f"[数据集] 前 {n} 条查询已加载（共 {len(records)} 条）")
    return records


def run():
    start_time = time.time()
    routes_str = "-".join(SEARCH_ROUTES)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        BASE_DIR, "gen_result",
        f"AutoScholarQuery_first{N_TOP}_{routes_str}_depth{MAX_DEPTH}_sim{SCORE_THRESH}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集（前 N 条）
    records = load_top_n_records(DATASET_PATH, N_TOP)

    # 创建搜索 agent
    search_agent = AcademicSearchTree(
        max_depth=MAX_DEPTH,
        max_docs=RELEVANCE_DOC_NUM,
        similarity_threshold=SCORE_THRESH,
    )

    # 扫描已有结果（断点续跑缓存）
    existing_results = {}
    for fname in os.listdir(output_dir):
        if fname.endswith(".json") and len(fname) > 32:
            existing_results[fname[:32]] = os.path.join(output_dir, fname)

    per_query_results = []
    for i, record in enumerate(records):
        question = record["question"]
        q_hash = get_md5(question)
        result_data = None

        cached = existing_results.get(q_hash)
        if cached and os.path.exists(cached):
            with open(cached, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            print(f"\n  [使用缓存] {i+1}/{len(records)}: {question[:60]}...")
        else:
            print(f"\n  [搜索] {i+1}/{len(records)}: {question[:60]}...")
            t0 = time.time()
            try:
                sorted_docs = search_agent.search(question, end_date="")
                result_data = search_agent.root.convert_to_dict()
                dest_file = os.path.join(output_dir, f"{q_hash}.json")
                with open(dest_file, "w", encoding="utf-8") as fw:
                    json.dump(result_data, fw, indent=2, ensure_ascii=False)
                print(f"  [已保存] {dest_file} (耗时 {time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  [错误] {question[:60]} 搜索失败: {e}")
                traceback.print_exc()
                continue

        if not result_data:
            print(f"  [跳过] {question[:60]} 无结果数据")
            continue

        # 提取预测结果 + 过滤（与 evaluate.py 相同逻辑）
        all_docs, pred_ids = extract_predicted_ids(result_data)
        pred_titles = [d.get("title", "") for d in all_docs if d.get("title")]
        filtered_pred_ids = set()
        filtered_pred_titles = []
        for doc in all_docs:
            score = doc.get("rerank_score_bge", doc.get("sim_score", 0.0))
            if score >= FILTER_THRESHOLD:
                doc_id = doc.get("paper_id", "")
                if doc_id:
                    filtered_pred_ids.add(doc_id)
                    title = doc.get("title", "")
                    if title:
                        filtered_pred_titles.append(title)

        # 评估
        eval_result = evaluate_query(
            pred_ids=filtered_pred_ids,
            gold_ids=record["answer_ids"],
            pred_titles=filtered_pred_titles,
            gold_titles=record["answer_titles"],
            gold_id_to_title=record.get("gold_id_to_title", {}),
        )

        per_query_results.append({
            "qid": record["qid"],
            "question": question,
            "num_gold": record["answer_ids"],
            "num_pred": filtered_pred_ids,
            "eval": eval_result,
        })

        score_str = (
            f"P={eval_result['precision']:.3f} R={eval_result['recall']:.3f} "
            f"F1={eval_result['f1']:.3f}  "
            f"TP={eval_result['tp']} FP={eval_result['fp']} FN={eval_result['fn']}"
        )
        print(f"  [结果] {score_str}")

    elapsed = time.time() - start_time
    print(f"\n[评估完成] 共 {len(per_query_results)} 条有效结果，耗时 {elapsed:.1f}s")

    if not per_query_results:
        print("[错误] 没有有效的评估结果")
        sys.exit(1)

    # 汇总与报告
    stats = aggregate_metrics(per_query_results)
    print_report(stats, f"SPAR Evaluation on AutoScholarQuery (前{N_TOP}条, 外部数据集)")

    # 保存 Excel
    excel_path = os.path.join(output_dir, f"evaluation_AutoScholarQuery_first{N_TOP}.xlsx")
    try:
        save_results_to_excel(stats, excel_path, "AutoScholarQuery")
    except Exception as e:
        print(f"[Excel 保存失败] {e}")

    # 绘图
    try:
        draw_metric_distributions(stats, output_dir, "AutoScholarQuery")
    except Exception as e:
        print(f"[绘图失败] {e}")

    # JSON 报告
    serializable_stats = {
        "benchmark": "AutoScholarQuery",
        "dataset_path": DATASET_PATH,
        "first_n": N_TOP,
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
        "micro_precision_ids": stats.get("micro_precision_ids", 0),
        "micro_recall_ids": stats.get("micro_recall_ids", 0),
        "micro_f1_ids": stats.get("micro_f1_ids", 0),
        "at_k": {str(k): v for k, v in stats.get("at_k", {}).items()},
        "config": {
            "depth": MAX_DEPTH,
            "score_thresh": SCORE_THRESH,
            "relevance_docs": RELEVANCE_DOC_NUM,
            "filter_threshold": FILTER_THRESHOLD,
            "sample": "first_20",
            "mode": "full",
        },
    }
    report_json_path = os.path.join(output_dir, f"evaluation_AutoScholarQuery_first{N_TOP}.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    print(f"[报告] JSON 报告已保存到 {report_json_path}")

    print(f"\n[输出] 所有输出保存在: {output_dir}")
    return stats


if __name__ == "__main__":
    setup_run_log()
    run()
