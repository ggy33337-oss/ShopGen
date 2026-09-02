# -*- coding: utf-8 -*-

import unittest

from runtime.gen_searcher_chain import (
    GenSearcherImageChain,
    _bounded_float,
    _extract_entity_hint,
    _matches_entity,
    _normalize_visual_requirements,
)


class GenSearcherSelectionTests(unittest.TestCase):
    def test_entity_hint_preserves_the_exact_college_name(self):
        self.assertEqual(
            "南昌航空大学科技学院",
            _extract_entity_hint("生成南昌航空大学科技学院招生海报"),
        )

    def test_entity_filter_rejects_parent_university_results(self):
        entity = "南昌航空大学科技学院"
        self.assertTrue(
            _matches_entity(
                {"title": "南昌航空大学科技学院校徽", "link": "https://baike.baidu.com/item/x"},
                entity,
            )
        )
        self.assertFalse(
            _matches_entity(
                {"title": "校园风光_学校概况_南昌航空大学", "link": "https://www.nchu.edu.cn/xxgk/xyfg"},
                entity,
            )
        )

    def test_bounded_float_always_receives_explicit_bounds(self):
        self.assertEqual(0.0, _bounded_float("not-a-score", 0.0, 0.0, 1.0))
        self.assertEqual(1.0, _bounded_float(2, 0.0, 0.0, 1.0))

    def test_same_visual_layer_keeps_strongest_reference(self):
        chain = GenSearcherImageChain({"GEN_SEARCHER_REFERENCE_LIMIT": "5"})
        image_map = {
            "IMG_001": {
                "img_id": "IMG_001",
                "visual_role": "representative_scene",
                "local_path": "old.jpg",
                "quality_score": 0.30,
                "information_score": 0.30,
                "retrieved_at": "2026-08-01T00:00:00+00:00",
            },
            "IMG_002": {
                "img_id": "IMG_002",
                "visual_role": "representative_scene",
                "local_path": "new.jpg",
                "quality_score": 0.90,
                "information_score": 0.85,
                "retrieved_at": "2026-09-01T00:00:00+00:00",
            },
            "IMG_003": {
                "img_id": "IMG_003",
                "visual_role": "identity",
                "local_path": "logo.png",
                "quality_score": 0.70,
                "information_score": 0.60,
                "retrieved_at": "2026-09-01T00:00:00+00:00",
            },
        }
        selected = chain._select_references(
            {"reference_images": [{"img_id": "IMG_001"}, {"img_id": "IMG_003"}]},
            image_map,
            _normalize_visual_requirements(
                [{"visual_role": "identity", "required": True, "query": "school logo"}]
            ),
        )
        self.assertEqual(["IMG_003", "IMG_002"], [item["img_id"] for item in selected])
        self.assertEqual(
            ["identity", "representative_scene"],
            [item["visual_role"] for item in selected],
        )

    def test_visual_requirements_do_not_force_unrequested_layers(self):
        requirements = _normalize_visual_requirements(
            [
                {"visual_role": "logo", "required": "false", "query": "logo"},
                {"visual_role": "campus", "required": True, "query": "campus"},
            ]
        )
        self.assertEqual(["identity", "representative_scene"], [item["visual_role"] for item in requirements])
        self.assertFalse(requirements[0]["required"])
        self.assertTrue(requirements[1]["required"])


if __name__ == "__main__":
    unittest.main()
