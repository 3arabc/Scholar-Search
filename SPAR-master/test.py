# coding:utf-8
# @FileName: test.py
# @Author  : BLC
# @Time    : 2026/6/24 20:45
# @Project : SPAR-master
# @Function:
# test_openalex.py
# test.py
import requests

url = "https://google.serper.dev/search"
headers = {"X-API-KEY": "f5d0629ef3fbfc8ea4fccc95ef6249825f4a3485"}

# 尝试不同的搜索词
test_queries = [
    "Retrieval-Augmented Generation arxiv",           # 不加 site:，直接加 arxiv
    "RAG arxiv",                                       # 缩写
    '"Retrieval-Augmented Generation" arxiv',         # 带引号
    "Retrieval-Augmented Generation paper arxiv"      # 加 paper
]

for q in test_queries:
    params = {"q": q, "num": 10}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    organic_count = len(data.get("organic", []))
    print(f"搜索词: {q}")
    print(f"  organic 数量: {organic_count}")
    if organic_count > 0:
        print(f"  第一条结果: {data['organic'][0].get('link', '')}")
    print()