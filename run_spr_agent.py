# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

import json
from datetime import datetime, timedelta

import json
from tqdm import tqdm
import os
import traceback
import random
from global_config import (
    LLM_MODEL_NAME,
    DO_REFERENCE_SEARCH,
    DO_FUSION_JUDGE,
    FUSION_TEMPLATE,
    SEARCH_ROUTES,
)
import sys
import glob
from utils import get_md5
import shutil
from pipeline_spar import AcademicSearchTree

file_lst = [
    "./global_config.py",
    "./instruction.py",
    "./run_spr_agent.py",
    "./search_engine.py",
    "./api_web.py",
    "./pipeline_spar.py"
]

sample_num = 50  # 评测选用的题目总量。例如跑 50 题测试则设为 50，跑 10 题设为 10
score_thresh = 0.5  # 文献推荐的相似度及格线。只有当大模型评估相似度分数 >= 该分值时，才把该文献列入最终的 Relevant 推荐列表中
max_depth = 2  # 学术树最大下探检索层数。设为 2 代表最大为两轮自适应下探检索（第一层探索完评估不足则下探第二层）
relevance_doc_num = 10  # 最终输出列表所包含的最大文献篇数限制（赛题一般限制最多提交 10 篇最相关论文）

benchmark_map = {
    "AutoScholarQuery": {
        "src_file": "./benchmark/AutoScholarQuery_test.jsonl",
        "select_file": f"./benchmark/AutoScholarQuery_test_select_{sample_num}.jsonl",
    },
    "OwnBenchmark": {
        "src_file": "./benchmark/spar_bench.jsonl",
        "select_file": f"./benchmark/spar_bench_select_{sample_num}.jsonl"
    },
}

# =============================================================================
# 【数据集选择与命令行传参】
# 运行时必须通过命令行传入数据集名称作为第一个参数。例如：
#   python run_spr_agent.py OwnBenchmark
#   python run_spr_agent.py AutoScholarQuery
# =============================================================================
if len(sys.argv) < 2:
    print("错误: 缺少命令行参数。使用示例:")
    print("  python run_spr_agent.py OwnBenchmark")
    print("  python run_spr_agent.py AutoScholarQuery")
    sys.exit(1)

benchmark_name = sys.argv[1]

src_file = benchmark_map[benchmark_name]["src_file"]
select_file = benchmark_map[benchmark_name]["select_file"]
print(f"select_file: {select_file}")


output_folder = f"./gen_result/{benchmark_name}_{sample_num}_msearch_{'-'.join(SEARCH_ROUTES)}_depth{max_depth}_do_reference_{DO_REFERENCE_SEARCH}_query_judge_{DO_FUSION_JUDGE}_fusion_{FUSION_TEMPLATE}_no_enddate_no_autocorrect_pasa_score_{score_thresh}"  # 加上query fusion

print(f"output_folder: {output_folder}")

os.makedirs(output_folder, exist_ok=True)


search_agent = AcademicSearchTree(
    max_depth=max_depth, max_docs=relevance_doc_num, similarity_threshold=score_thresh
)


for one in file_lst:
    shutil.copy2(one, output_folder)

already = {}
for one in glob.glob(f"{output_folder}/*.json"):  # 【断点续传/秒级重入机制】
    with open(one, "r", encoding="utf-8") as fr:
        info = json.load(fr)
    question = info["search_query"]
    already[question] = one  # 扫描已生成输出的 json，如果文件已经落地，则程序下次启动时自动跳过这道题，免去重复打分费用

with open(src_file, "r", encoding="utf-8") as f:
    if src_file.endswith(".jsonl"):
        lines = f.readlines()  # 读取数据集的行
        random.seed(123)  # 使用固定随机种子保证评测子集的可重复性
        random.shuffle(lines)  # 将题目打乱
        lines = lines[:sample_num]  # 切片挑选出前 sample_num 题运行测试
    elif src_file.endswith(".json"):
        lines = json.load(f)
    print(f"lines: {len(lines)}")

    with open(select_file,"w", encoding="utf-8") as fw:
        for one in lines:
            fw.write(one.strip() + "\n")


    for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Processing lines"):
        try:
            if isinstance(line, str):
                data = json.loads(line)
                question = data["question"]
            elif isinstance(line, dict):
                data = line
                question = data["query"]
            else:
                data = {}
                question = line

            end_date = ""
            if question in already:
                print(f"pass: {already[question]}")
                continue

            dest_name = get_md5(question)
            dest_file = os.path.join(output_folder, f"{dest_name}.json")

            sorted_docs = search_agent.search(question, end_date=end_date)

            if "answer" in data:
                search_agent.root.extra["answer"] = data["answer"]

            elif "data_result_add_score" in data:
                search_agent.root.extra["answer"] = [
                    one["title"] for one in data["data_result_add_score"]
                ]

            if output_folder != "":
                res = search_agent.root.convert_to_dict()
                with open(dest_file, "w") as fw:
                    json.dump(res, fw, indent=2)

            try:
                print("draw search tree")
                search_agent.visualize_tree(f"{output_folder}/{dest_name}")
            except:
                traceback.print_exc()
                pass

            # break

        except:
            traceback.print_exc()
            pass

print(f"output_folder: {output_folder}")
