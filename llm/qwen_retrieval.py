# -*- coding: utf-8 -*-

from llm.qwen_client import QwenGateway, extract_chat_content, load_json_object, normalize_qwen_model


DEFAULT_EMBEDDING_MODEL = "qwen3-vl-embedding"
DEFAULT_EMBEDDING_DIMENSION = 1024


class QwenRetrievalGateway:
    def __init__(self, values, gateway=None):
        self.values = values
        self.gateway = gateway or QwenGateway(values)
        self.embedding_model = normalize_qwen_model(
            values.get("QWEN_EMBEDDING_MODEL"),
            DEFAULT_EMBEDDING_MODEL,
            "向量化",
        )
        self.dimension = int(values.get("QWEN_EMBEDDING_DIMENSION") or DEFAULT_EMBEDDING_DIMENSION)
        if self.dimension not in {256, 512, 768, 1024, 1536, 2048, 2560}:
            raise RuntimeError("QWEN_EMBEDDING_DIMENSION 不是 qwen3-vl-embedding 支持的维度。")
        self.endpoint = str(
            values.get("QWEN_EMBEDDING_ENDPOINT")
            or f"{self.gateway.api_base_url}/services/embeddings/multimodal-embedding/multimodal-embedding"
        ).strip()
        self.request_timeout = int(values.get("QWEN_EMBEDDING_TIMEOUT") or 120)
        self.request_attempts = min(max(int(values.get("QWEN_EMBEDDING_ATTEMPTS") or 2), 1), 3)

    def embed_text(self, text):
        return self._embed([{"text": str(text or "").strip()}])

    def embed_image(self, image_data_url, description=""):
        contents = [{"image": image_data_url}]
        if str(description or "").strip():
            contents.append({"text": str(description).strip()})
        return self._embed(contents)

    def _embed(self, contents):
        if not contents or not any(next(iter(item.values()), "") for item in contents):
            raise RuntimeError("千问向量模型缺少有效输入。")
        response = self.gateway._post_json(
            self.endpoint,
            {
                "model": self.embedding_model,
                "input": {"contents": contents},
                "parameters": {
                    "enable_fusion": True,
                    "dimension": self.dimension,
                    "instruct": "Retrieve relevant ecommerce product knowledge and visual assets.",
                },
            },
            timeout=self.request_timeout,
            error_label="千问向量模型",
            attempts=self.request_attempts,
        )
        embeddings = response.get("output", {}).get("embeddings", [])
        vector = embeddings[0].get("embedding", []) if embeddings else []
        if len(vector) != self.dimension:
            raise RuntimeError(
                f"千问向量模型返回维度异常：期望 {self.dimension}，实际 {len(vector)}。"
            )
        return [float(value) for value in vector]

    def describe_image(self, image_data_url, title="", category="", context=""):
        messages = [
            {
                "role": "system",
                "content": (
                    "你是电商知识库图片分类器，只返回严格 JSON。字段为 title、category、description、tags。"
                    "description 应描述商品主体、颜色、材质、构图、背景、光线和可见文字；"
                    "tags 是最多 10 个中文短标签组成的数组。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"已有标题：{title or '无'}\n"
                            f"已有分类：{category or '无'}\n"
                            f"图片附近的文档文字：{str(context or '')[:1200] or '无'}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ]
        response = self.gateway.chat_completion(
            messages=messages,
            model=self.gateway.vl_model,
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
            timeout=self.request_timeout,
            error_label="千问图片分类模型",
            attempts=self.request_attempts,
        )
        payload = load_json_object(extract_chat_content(response), "千问图片分类模型")
        tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
        return {
            "title": str(payload.get("title") or title or "图片资料").strip()[:255],
            "category": str(payload.get("category") or category or "图片素材").strip()[:100],
            "description": str(payload.get("description") or "").strip()[:3000],
            "tags": [str(item).strip()[:50] for item in tags[:10] if str(item).strip()],
        }
