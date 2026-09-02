# -*- coding: utf-8 -*-

import argparse
import json

from qdrant_client import models

from core.config import read_config
from knowledge.qdrant_knowledge_base import QdrantKnowledgeBase
from knowledge.quality import is_meaningful_knowledge_text


def find_invalid_points(knowledge_base):
    invalid = []
    for collection_name in (
        knowledge_base.text_collection,
        knowledge_base.image_collection,
    ):
        offset = None
        while True:
            points, offset = knowledge_base.client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                if is_meaningful_knowledge_text(
                    payload.get("content"),
                    payload.get("title"),
                ):
                    continue
                invalid.append(
                    {
                        "collection": collection_name,
                        "point_id": str(point.id),
                        "source_id": str(payload.get("source_id") or ""),
                        "title": str(payload.get("title") or ""),
                        "content": str(payload.get("content") or "")[:300],
                    }
                )
            if offset is None:
                break
    return invalid


def delete_invalid_points(knowledge_base, invalid):
    deleted = 0
    for collection_name in (
        knowledge_base.text_collection,
        knowledge_base.image_collection,
    ):
        point_ids = [
            item["point_id"]
            for item in invalid
            if item["collection"] == collection_name
        ]
        if not point_ids:
            continue
        knowledge_base.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )
        deleted += len(point_ids)
    return deleted


def main():
    parser = argparse.ArgumentParser(description="审计并清理知识库低信息向量")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际删除；默认只输出审计结果",
    )
    args = parser.parse_args()
    knowledge_base = QdrantKnowledgeBase(read_config(".env"))
    invalid = find_invalid_points(knowledge_base)
    deleted = delete_invalid_points(knowledge_base, invalid) if args.apply else 0
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "invalid_count": len(invalid),
                "deleted_count": deleted,
                "points": invalid,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
