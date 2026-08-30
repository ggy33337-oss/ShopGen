# -*- coding: utf-8 -*-

import uuid
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import QdrantClient, models

from core.config import read_config
from knowledge.models import KnowledgeContext, KnowledgeImageRecord
from llm.qwen_retrieval import QwenRetrievalGateway


DEFAULT_TEXT_COLLECTION = "ecommerce_kb_text"
DEFAULT_IMAGE_COLLECTION = "ecommerce_kb_image"


class QdrantKnowledgeBase:
    def __init__(self, values=None, client=None, retrieval_gateway=None):
        self.values = values or read_config(".env")
        self.client = client or QdrantClient(
            url=str(self.values.get("QDRANT_URL") or "http://127.0.0.1:6333").rstrip("/"),
            api_key=str(self.values.get("QDRANT_API_KEY") or "") or None,
            timeout=int(self.values.get("QDRANT_TIMEOUT") or 30),
            trust_env=parse_boolean(self.values.get("QDRANT_TRUST_ENV"), default=False),
            check_compatibility=False,
        )
        self.retrieval_gateway = retrieval_gateway
        self.dimension = int(self.values.get("QWEN_EMBEDDING_DIMENSION") or 1024)
        self.text_collection = str(
            self.values.get("QDRANT_TEXT_COLLECTION") or DEFAULT_TEXT_COLLECTION
        ).strip()
        self.image_collection = str(
            self.values.get("QDRANT_IMAGE_COLLECTION") or DEFAULT_IMAGE_COLLECTION
        ).strip()
        self.embedding_workers = min(
            max(int(self.values.get("KNOWLEDGE_EMBEDDING_WORKERS") or 3), 1),
            8,
        )

    def ensure_collections(self):
        for collection_name in (self.text_collection, self.image_collection):
            if self.client.collection_exists(collection_name):
                continue
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            for field_name in ("source_id", "category", "chunk_type"):
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def upsert_text_chunks(self, chunks):
        self.ensure_collections()
        points = self._build_text_points(list(chunks))
        if points:
            self.client.upsert(collection_name=self.text_collection, points=points, wait=True)
        return len(points)

    def upsert_image(
        self,
        source_id,
        image_data_url,
        title,
        category,
        description,
        source_filename,
        tags=None,
    ):
        self.ensure_collections()
        record = KnowledgeImageRecord(
            source_id=source_id,
            image_data_url=image_data_url,
            title=title,
            category=category,
            description=description,
            source_filename=source_filename,
            source_title=title,
            tags=tuple(tags or ()),
        )
        point = self._build_image_point(record)
        self.client.upsert(
            collection_name=self.image_collection,
            points=[point],
            wait=True,
        )
        return str(point.id)

    def replace_source(self, chunks, images):
        self.ensure_collections()
        chunks = list(chunks or [])
        images = list(images or [])
        source_ids = {item.source_id for item in [*chunks, *images] if item.source_id}
        if len(source_ids) != 1:
            raise ValueError("资料替换必须且只能包含一个 source_id。")

        text_points = self._build_text_points(chunks)
        image_points = [self._build_image_point(record) for record in images]
        source_id = next(iter(source_ids))
        self.delete_source(source_id)
        if text_points:
            self.client.upsert(
                collection_name=self.text_collection,
                points=text_points,
                wait=True,
            )
        if image_points:
            self.client.upsert(
                collection_name=self.image_collection,
                points=image_points,
                wait=True,
            )
        return len(text_points), len(image_points)

    def search(self, query, limit=3, category=""):
        query = str(query or "").strip()
        if not query:
            return KnowledgeContext(query="", status="empty_query")
        try:
            matches = self.search_matches(query, limit=limit, category=category)
        except Exception:
            return KnowledgeContext(query=query, status="unavailable")
        if not matches:
            return KnowledgeContext(query=query, status="no_match")
        return KnowledgeContext(
            query=query,
            status="matched",
            context_text=build_context_text(matches),
            matches=tuple(matches),
        )

    def search_matches(self, query, limit=5, category=""):
        self.ensure_collections()
        query_vector = self._get_retrieval_gateway().embed_text(query)
        query_filter = build_category_filter(category)
        candidates = []
        candidate_limit = max(int(limit) * 3, 6)
        for collection_name in (self.text_collection, self.image_collection):
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=candidate_limit,
                with_payload=True,
            )
            for point in response.points:
                payload = dict(point.payload or {})
                candidates.append(
                    {
                        "source_id": str(payload.get("source_id") or ""),
                        "chunk_id": str(payload.get("chunk_id") or point.id),
                        "title": str(payload.get("title") or "未命名资料"),
                        "content": str(payload.get("content") or ""),
                        "chunk_type": str(payload.get("chunk_type") or "semantic"),
                        "category": str(payload.get("category") or ""),
                        "source_filename": str(payload.get("source_filename") or ""),
                        "source_title": str(payload.get("source_title") or ""),
                        "page_number": int(payload.get("page_number") or 0),
                        "section": str(payload.get("section") or ""),
                        "score": float(point.score),
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return deduplicate_matches(candidates, int(limit))

    def delete_source(self, source_id):
        self.ensure_collections()
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=str(source_id)),
                    )
                ]
            )
        )
        for collection_name in (self.text_collection, self.image_collection):
            self.client.delete(
                collection_name=collection_name,
                points_selector=selector,
                wait=True,
            )

    def list_sources(self, limit=100):
        self.ensure_collections()
        sources = {}
        for collection_name in (self.text_collection, self.image_collection):
            offset = None
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = dict(point.payload or {})
                    source_id = str(payload.get("source_id") or "")
                    if not source_id:
                        continue
                    item = sources.setdefault(
                        source_id,
                        {
                            "source_id": source_id,
                            "title": str(
                                payload.get("source_title")
                                or payload.get("title")
                                or "未命名资料"
                            ),
                            "category": str(payload.get("category") or ""),
                            "source_filename": str(payload.get("source_filename") or ""),
                            "vector_count": 0,
                        },
                    )
                    item["vector_count"] += 1
                if next_offset is None:
                    break
                offset = next_offset
        return list(sources.values())[: int(limit)]

    def _get_retrieval_gateway(self):
        if self.retrieval_gateway is None:
            self.retrieval_gateway = QwenRetrievalGateway(self.values)
            self.dimension = self.retrieval_gateway.dimension
        return self.retrieval_gateway

    def _build_text_points(self, chunks):
        if not chunks:
            return []
        gateway = self._get_retrieval_gateway()
        if self.embedding_workers == 1 or len(chunks) == 1:
            vectors = [gateway.embed_text(chunk.content) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=self.embedding_workers) as executor:
                vectors = list(executor.map(gateway.embed_text, (chunk.content for chunk in chunks)))
        return [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "source_id": chunk.source_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "source_title": chunk.source_title or chunk.title,
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type,
                    "category": chunk.category,
                    "source_filename": chunk.source_filename,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

    def _build_image_point(self, record):
        point_id = str(uuid.uuid4())
        vector = self._get_retrieval_gateway().embed_image(
            record.image_data_url,
            record.description,
        )
        return models.PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "source_id": record.source_id,
                "chunk_id": point_id,
                "title": record.title,
                "source_title": record.source_title or record.title,
                "content": record.description or f"图片资料：{record.title}",
                "chunk_type": "image",
                "category": record.category,
                "source_filename": record.source_filename,
                "page_number": record.page_number,
                "section": record.section,
                "tags": list(record.tags),
            },
        )


def build_category_filter(category):
    category = str(category or "").strip()
    if not category:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value=category),
            )
        ]
    )


def parse_boolean(value, default=False):
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def deduplicate_matches(candidates, limit):
    matches = []
    seen = set()
    for candidate in candidates:
        identity = (candidate["source_id"], candidate["chunk_type"], candidate["content"])
        if identity in seen:
            continue
        seen.add(identity)
        matches.append(candidate)
        if len(matches) >= limit:
            break
    return matches


def build_context_text(matches):
    parts = []
    for index, item in enumerate(matches, start=1):
        parts.append(
            f"[知识库匹配 {index}]\n"
            f"标题：{item['title']}\n"
            f"分类：{item['category'] or '未分类'}\n"
            f"类型：{item['chunk_type']}\n"
            f"章节：{item.get('section') or '未标注'}\n"
            f"页码：{item.get('page_number') or '未标注'}\n"
            f"内容：{item['content'][:2000]}"
        )
    return "\n\n".join(parts)
