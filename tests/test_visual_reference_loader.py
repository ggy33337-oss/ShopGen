# -*- coding: utf-8 -*-

import base64
import io
import unittest

from PIL import Image

from services.visual_reference_loader import prepare_visual_planner_image


def image_data_url(size=(1122, 1402), image_format="PNG"):
    image = Image.new("RGB", size, (180, 40, 40))
    output = io.BytesIO()
    image.save(output, format=image_format)
    media_type = "image/png" if image_format == "PNG" else "image/jpeg"
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


class PlannerReferenceImageTests(unittest.TestCase):
    def test_transcodes_large_image_to_configured_jpeg_thumbnail(self):
        original = image_data_url()

        planner_url, details = prepare_visual_planner_image(
            {
                "QWEN_VL_MAX_IMAGE_SIDE": "768",
                "QWEN_VL_IMAGE_QUALITY": "80",
            },
            original,
        )

        self.assertTrue(details["prepared"])
        self.assertTrue(planner_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual([1122, 1402], details["original_size"])
        self.assertEqual([615, 768], details["planner_size"])
        self.assertLess(details["planner_bytes"], details["original_bytes"])

    def test_invalid_data_url_is_preserved_for_fake_or_non_image_references(self):
        original = "data:image/png;base64,aW1hZ2U="

        planner_url, details = prepare_visual_planner_image({}, original)

        self.assertEqual(original, planner_url)
        self.assertFalse(details["prepared"])


if __name__ == "__main__":
    unittest.main()
