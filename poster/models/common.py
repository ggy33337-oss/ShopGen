# -*- coding: utf-8 -*-

from typing import Any


MAX_TEXT_LENGTH = 12000
MAX_SHORT_TEXT_LENGTH = 400
MAX_PROMPT_LENGTH = 6000


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    items = []
    for item in value:
        text = normalize_text(item)
        if text:
            items.append(text[:MAX_SHORT_TEXT_LENGTH])
    return items
