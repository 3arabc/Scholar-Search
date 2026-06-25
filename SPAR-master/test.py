# coding:utf-8
# @FileName: test.py
# @Author  : BLC
# @Time    : 2026/6/24 20:45
# @Project : SPAR-master
# @Function:
import requests
url = "https://google.serper.dev/search"
headers = {"X-API-KEY": "f5d0629ef3fbfc8ea4fccc95ef6249825f4a3485"}
params = {"q": "test", "num": 1}
response = requests.get(url, headers=headers, params=params)
print(response.status_code)
print(response.text[:200])