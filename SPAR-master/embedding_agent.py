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

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量获取文本向量（每次请求多篇，减少 HTTP 往返）。
        OpenAI 兼容接口支持 input 数组，返回的 data 按 index 排序对齐。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"model": self.model, "input": batch}
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            embeddings.extend([d["embedding"] for d in data])
        return embeddings

    def get_score(self, query: str, documents: List[str], **kwargs) -> List[float]:
        """
        计算 query 与每个 document 的余弦相似度。
        query 单独一次请求，documents 批量请求（batch_size 可控，默认 32）。
        """
        batch_size = kwargs.get("batch_size", 32)
        q_vec = np.array(self._get_embedding(query))
        doc_vecs = [np.array(v) for v in self.embed_documents(documents, batch_size=batch_size)]
        scores = []
        for d_vec in doc_vecs:
            cos_sim = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
            scores.append(float(cos_sim))
        return scores