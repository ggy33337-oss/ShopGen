# -*- coding: utf-8 -*-

import json


class StrictJsonError(ValueError):
    pass


def load_strict_json_object(raw_text, layer_name):
    text = str(raw_text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise StrictJsonError(f"{layer_name} 未返回严格 JSON。") from e

    if not isinstance(data, dict):
        raise StrictJsonError(f"{layer_name} JSON 根节点必须是对象。")
    return data
