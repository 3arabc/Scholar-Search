import os
import requests
import numpy as np
from typing import List, Union

class BGEM3EmbeddingAgent:
    def __init__(self, api_key: str = None, model: str = "BAAI/bge-m3"):
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("请提供 SiliconFlow API Key 或设置环境变量 SILICONFLOW_API_KEY")
        self.model = model
        self.url = "https://api.siliconflow.cn/v1/embeddings"

    def _get_embedding(self, text: str) -> List[float]:
        """内部方法：获取单个文本的向量"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": text
        }
        response = requests.post(self.url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(t) for t in texts]

    def get_score(self, query: str, documents: List[str], **kwargs) -> List[float]:
        """
        计算 query 与每个 document 的余弦相似度。
        这里 documents 是已拼接的字符串列表（由调用方生成），
        但我们可以要求调用方传入结构化数据，或在内部重构。
        为了最小改动，我们在调用方（search_engine.py）构造文本时优化。
        """
        q_vec = np.array(self._get_embedding(query))
        doc_vecs = [np.array(self._get_embedding(doc)) for doc in documents]
        scores = []
        for d_vec in doc_vecs:
            cos_sim = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
            scores.append(float(cos_sim))
        return scores