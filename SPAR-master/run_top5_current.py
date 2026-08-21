#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Descriptions] : 用当前系统跑外部数据集前 5 题（按顺序，非随机）
#                  复用 evaluate.py 的评估逻辑
# ==================================================================

import json
import os
import sys
import time
import traceback
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from global_config import SEARCH_ROUTES
from pipeline_spar import AcademicSearchTree
from utils import get_md5
from evaluate import (
    extract_predicted_ids,
    evaluate_query,
    aggregate_metrics,
    print_report,
    save_results_to_excel,
    draw_metric_distributions,
    extract_arxiv_id,
)

DATASET_PATH = r"C:\Users\lenovo\Desktop\人工智能创新大赛\Datasets\PasaDataset\AutoScholarQuery\test.jsonl"
N_TOP = 20
MAX_DEPTH = 2
SCORE_THRESH = 0.5
RELEVANCE_DOC_NUM = 10
FILTER_THRESHOLD = 0.4

LOG_DIR = os.path.join(BASE_DIR, "eval_log")
os.makedirs(LOG_DIR, exist_ok=True)


def setup_run_log():
    """将 print / logger 输出重定向到 eval_log/"""
    import logging

    log_path = os.path.join(
        LOG_DIR, f"run_top{N_TOP}_current_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    stream = open(log_path, "w", encoding="utf-8")
    sys.stdout = stream
    sys.stderr = stream
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler):
            h.setStream(stream)
    print(f"[运行日志] {log_path}")


def load_top_n_records(dataset_path: str, n: int):
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
            for aid in data.get("answer_arxiv_id", []):
                aid = str(aid).strip()
                if aid:
                    extracted = extract_arxiv_id(aid)
                    answer_ids.add(extracted if extracted else aid)
            records.append({
                "question": question,
                "answer_titles": answer_titles,
                "answer_ids": answer_ids,
                "gold_id_to_title": {},
                "qid": data.get("qid", f"q_{i}"),
                "raw": data,
            })
    print(f"[数据集] 前 {n} 条查询已加载（共 {len(records)} 条）")
    return records


def run():
    start_time = time.time()
    routes_str = "-".join(SEARCH_ROUTES)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        BASE_DIR, "gen_result",
        f"AutoScholarQuery_first{N_TOP}_current_{routes_str}_depth{MAX_DEPTH}_sim{SCORE_THRESH}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    records = load_top_n_records(DATASET_PATH, N_TOP)
    search_agent = AcademicSearchTree(
        max_depth=MAX_DEPTH,
        max_docs=RELEVANCE_DOC_NUM,
        similarity_threshold=SCORE_THRESH,
    )

    per_query_results = []
    for i, record in enumerate(records):
        question = record["question"]
        q_hash = get_md5(question)
        print(f"\n  [搜索] {i+1}/{len(records)}: {question[:60]}...")
        t0 = time.time()
        try:
            sorted_docs = search_agent.search(question, end_date="")
            result_data = search_agent.root.convert_to_dict()
            dest_file = os.path.join(output_dir, f"{q_hash}.json")
            with open(dest_file, "w", encoding="utf-8") as fw:
                json.dump(result_data, fw, indent=2, ensure_ascii=False)
            print(f"  [已保存] (耗时 {time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  [错误] {question[:60]} 搜索失败: {e}")
            traceback.print_exc()
            continue

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
        print(
            f"  [结果] P={eval_result['precision']:.3f} R={eval_result['recall']:.3f} "
            f"F1={eval_result['f1']:.3f}  "
            f"TP={eval_result['tp']} FP={eval_result['fp']} FN={eval_result['fn']}"
        )

    elapsed = time.time() - start_time
    print(f"\n[评估完成] 共 {len(per_query_results)} 条有效结果，耗时 {elapsed:.1f}s")

    stats = aggregate_metrics(per_query_results)
    print_report(stats, f"SPAR Evaluation (前{N_TOP}条, 当前系统)")

    excel_path = os.path.join(output_dir, f"evaluation_first{N_TOP}.xlsx")
    try:
        save_results_to_excel(stats, excel_path, "AutoScholarQuery")
    except Exception as e:
        print(f"[Excel 保存失败] {e}")

    report_json_path = os.path.join(output_dir, f"evaluation_first{N_TOP}.json")
    serializable = {
        "benchmark": "AutoScholarQuery",
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
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"[报告] 已保存到 {report_json_path}")
    print(f"[输出] 所有输出保存在: {output_dir}")


if __name__ == "__main__":
    setup_run_log()
    run()
