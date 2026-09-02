# -*- coding: utf-8 -*-

from llm.qwen_client import QwenGateway, normalize_qwen_model


DEFAULT_RERANK_MODEL = "qwen3-rerank"
DEFAULT_RERANK_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
DEFAULT_RERANK_TIMEOUT = 60
DEFAULT_RERANK_ATTEMPTS = 2
DEFAULT_RERANK_INSTRUCT = (
    "Given an ecommerce image-editing request, rank passages only when they provide directly "
    "actionable facts needed for the requested edit. Treat generic topical overlap, low-information "
    "text, and passages that conflict with explicit user instructions as irrelevant."
)


class QwenReranker:
    def __init__(self, values, gateway=None):
        self.values = values
        self.gateway = gateway or QwenGateway(values)
        self.model = normalize_qwen_model(
            values.get("QWEN_RERANK_MODEL"),
            DEFAULT_RERANK_MODEL,
            "重排序",
        )
        self.endpoint = str(
            values.get("QWEN_RERANK_ENDPOINT") or DEFAULT_RERANK_ENDPOINT
        ).strip()
        self.timeout = int(values.get("QWEN_RERANK_TIMEOUT") or DEFAULT_RERANK_TIMEOUT)
        self.attempts = min(
            max(int(values.get("QWEN_RERANK_ATTEMPTS") or DEFAULT_RERANK_ATTEMPTS), 1),
            3,
        )
        self.instruct = str(
            values.get("QWEN_RERANK_INSTRUCT") or DEFAULT_RERANK_INSTRUCT
        ).strip()

    def rerank(self, query, candidates, top_n=3):
        candidates = list(candidates or [])
        if not candidates:
            return []
        documents = [format_candidate_document(item) for item in candidates]
        response = self.gateway._post_json(
            self.endpoint,
            {
                "model": self.model,
                "query": str(query or "").strip(),
                "documents": documents,
                "top_n": min(max(int(top_n), 1), len(documents)),
                "instruct": self.instruct,
            },
            timeout=self.timeout,
            error_label="千问重排序模型",
            attempts=self.attempts,
        )
        ranked = []
        seen_indexes = set()
        for result in response.get("results", []):
            index = int(result.get("index", -1))
            if index < 0 or index >= len(candidates) or index in seen_indexes:
                continue
            seen_indexes.add(index)
            rerank_score = float(result.get("relevance_score") or 0.0)
            candidate = dict(candidates[index])
            candidate["vector_score"] = float(candidate.get("score") or 0.0)
            candidate["rerank_score"] = rerank_score
            candidate["score"] = rerank_score
            ranked.append(candidate)
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked


def format_candidate_document(candidate):
    return (
        f"标题：{candidate.get('title') or '未命名资料'}\n"
        f"分类：{candidate.get('category') or '未分类'}\n"
        f"类型：{candidate.get('chunk_type') or 'semantic'}\n"
        f"内容：{str(candidate.get('content') or '')[:3000]}"
    )
