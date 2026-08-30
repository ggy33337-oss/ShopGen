# -*- coding: utf-8 -*-

import os
from pathlib import Path


def read_config(path=".env"):
    values = dict(os.environ)
    config_path = Path(path)
    if not config_path.exists():
        return values

    text = config_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        values[key.strip()] = value
    return values
