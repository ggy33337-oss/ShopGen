# -*- coding: utf-8 -*-

import unittest

from llm.qwen_reranker import QwenReranker


class CapturingGateway:
    def __init__(self):
        self.request = {}

    def _post_json(self, endpoint, payload, timeout, error_label, attempts=1):
        self.request = {
            "endpoint": endpoint,
            "payload": payload,
            "timeout": timeout,
            "error_label": error_label,
            "attempts": attempts,
        }
        return {
            "results": [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.72},
            ]
        }


class QwenRerankerTests(unittest.TestCase):
    def test_maps_rerank_scores_back_to_candidates(self):
        gateway = CapturingGateway()
        reranker = QwenReranker({}, gateway=gateway)
        candidates = [
            {
                "chunk_id": "first",
                "title": "物流说明",
                "content": "支持顺丰发货。",
                "chunk_type": "paragraph",
                "score": 0.88,
            },
            {
                "chunk_id": "second",
                "title": "品牌视觉规范",
                "content": "品牌标准蓝为 #0057B8。",
                "chunk_type": "paragraph",
                "score": 0.76,
            },
        ]

        ranked = reranker.rerank("背景改成品牌蓝色", candidates, top_n=2)

        self.assertEqual("second", ranked[0]["chunk_id"])
        self.assertEqual(0.91, ranked[0]["rerank_score"])
        self.assertEqual(0.76, ranked[0]["vector_score"])
        self.assertEqual(0.91, ranked[0]["score"])
        self.assertEqual("qwen3-rerank", gateway.request["payload"]["model"])
        self.assertEqual(2, gateway.request["payload"]["top_n"])
        self.assertIn("品牌标准蓝", gateway.request["payload"]["documents"][1])


if __name__ == "__main__":
    unittest.main()
