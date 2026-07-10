# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Descriptions] : Evaluation metrics calculator for SPAR results
# ==================================================================

import os
import json
import glob
import argparse
from tqdm import tqdm

def keep_letters(s):
    """移除一切非字母字符并转为纯小写"""
    letters = [c for c in s if c.isalpha()]
    result = ''.join(letters)
    return result.lower()

def cal_micro(pred_set, label_set):
    """计算 TP, FP, FN"""
    if len(label_set) == 0:
        return 0, 0, 0

    if len(pred_set) == 0:
        return 0, 0, len(label_set)

    tp = len(pred_set & label_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)
    return tp, fp, fn

def main():
    parser = argparse.ArgumentParser(description="Evaluate F1 performance for SPAR search results.")
    parser.add_argument('--output_folder', type=str, default=None, 
                        help="Path to the directory containing prediction JSON files. Defaults to the latest directory in ./gen_result/")
    args = parser.parse_args()

    output_folder = args.output_folder

    # 如果未指定，自动检测 gen_result 下最新的结果文件夹
    if not output_folder:
        gen_result_dir = "./gen_result"
        if os.path.exists(gen_result_dir):
            subdirs = [os.path.join(gen_result_dir, d) for d in os.listdir(gen_result_dir) 
                       if os.path.isdir(os.path.join(gen_result_dir, d))]
            if subdirs:
                # 按照修改时间排序，获取最新修改的结果文件夹
                output_folder = max(subdirs, key=os.path.getmtime)
                print(f"Auto-detected latest result folder: {output_folder}")
            else:
                print("Error: No result folders found in ./gen_result/. Please run run_spr_agent.py first.")
                return
        else:
            print("Error: ./gen_result/ directory does not exist. Please run run_spr_agent.py first.")
            return

    pred_files = glob.glob(os.path.join(output_folder, "*.json"))
    if not pred_files:
        print(f"Error: No JSON prediction files found in {output_folder}")
        return

    print(f"Evaluating {len(pred_files)} prediction files...")
    
    crawler_recalls = []
    precisions = []
    recalls = []
    
    for pred_file in tqdm(pred_files, desc="Calculating metrics"):
        try:
            with open(pred_file, "r", encoding="utf-8") as f:
                paper_root = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {pred_file}: {str(e)}")
            continue

        crawled_paper_set = set()
        selected_paper_set = set()
        
        # 提取标准答案 (Ground Truth)
        extra_data = paper_root.get("extra", {})
        answers = extra_data.get("answer", [])
        answer_paper_set = set([keep_letters(paper) for paper in answers])
        
        if not answer_paper_set:
            continue

        # 收集被检索并打分推荐的论文 (直接使用 root 节点 extra 字典中的全局缓存 searched_docs)
        searched_docs = extra_data.get("searched_docs", {})
        for doc in searched_docs.values():
            title = doc.get("title", "")
            if not title:
                continue
            normalized_title = keep_letters(title)
            
            sim_score = doc.get("sim_score", 0.0)
            if sim_score > 0.5:
                selected_paper_set.add(normalized_title)
            crawled_paper_set.add(normalized_title)

        crawled_res = cal_micro(crawled_paper_set, answer_paper_set)
        selected_res = cal_micro(selected_paper_set, answer_paper_set)

        # 收集当前题目的 Precision & Recall
        crawler_recall = crawled_res[0] / (crawled_res[0] + crawled_res[2] if (crawled_res[0] + crawled_res[2]) > 0 else 1e-9)
        precision = selected_res[0] / (selected_res[0] + selected_res[1] if (selected_res[0] + selected_res[1]) > 0 else 1e-9)
        recall = selected_res[0] / (selected_res[0] + selected_res[2] if (selected_res[0] + selected_res[2]) > 0 else 1e-9)

        crawler_recalls.append(crawler_recall)
        precisions.append(precision)
        recalls.append(recall)

    if not precisions:
        print("Error: No valid metrics could be calculated.")
        return

    avg_crawler_recall = sum(crawler_recalls) / len(crawler_recalls)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    
    # 自动计算最终 F1 得分
    if avg_precision + avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0.0

    print("\n" + "="*50)
    print("               Evaluation Results Summary             ")
    print("="*50)
    print(f"Total Evaluated Questions : {len(precisions)}")
    print(f"Average Precision         : {avg_precision:.4f} ({avg_precision*100:.2f}%)")
    print(f"Average Recall            : {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    print(f"Average Crawler Recall    : {avg_crawler_recall:.4f} ({avg_crawler_recall*100:.2f}%)")
    print("-"*50)
    print(f"Final F1 Score            : {f1_score:.4f} ({f1_score*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
