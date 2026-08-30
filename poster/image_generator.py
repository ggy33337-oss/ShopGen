# -*- coding: utf-8 -*-

from llm.qwen_client import QwenGateway
from poster.models.generation import FinalPromptResult, ImageGenerationResult


def extract_image_url(response):
    if not isinstance(response, dict):
        return ""

    choices = response.get("output", {}).get("choices", [])
    for choice in choices:
        content = choice.get("message", {}).get("content", [])
        for item in content:
            image_url = item.get("image")
            if image_url:
                return str(image_url).strip()
    return ""


def generate_poster_image(values, final_prompt: FinalPromptResult):
    image_url = QwenGateway(values).generate_image(final_prompt.final_prompt)
    return ImageGenerationResult(image_url=image_url)
