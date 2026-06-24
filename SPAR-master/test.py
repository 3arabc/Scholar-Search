# coding:utf-8
# @FileName: test.py
# @Author  : BLC
# @Time    : 2026/6/24 20:45
# @Project : SPAR-master
# @Function:
import requests

# 正确的API地址和你的密钥
url = "https://api.siliconflow.cn/v1/models"
headers = {
    "Authorization": "Bearer sk-hpanrlwpblvsroryefetlzyxzajksaodnrdnasojxiapxwpl" # 请替换为你的密钥
}

try:
    response = requests.get(url, headers=headers)
    print(f"状态码: {response.status_code}") # 若看到 200，即表示成功

    if response.status_code == 200:
        print("✅ API地址和密钥有效！")
        # print(response.json()) # 可取消注释查看模型列表
    else:
        print(f"❌ 请求失败，请检查密钥。返回信息: {response.text}")
except Exception as e:
    print(f"❌ 连接错误: {e}")