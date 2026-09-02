# -*- coding: utf-8 -*-

import re


PURE_INDEX_PATTERN = re.compile(
    r"^(?:第\s*)?[0-9一二三四五六七八九十百千]+"
    r"(?:\.[0-9]+)*(?:\s*[页章节部分])?$",
    flags=re.IGNORECASE,
)
SEMANTIC_CHARACTER_PATTERN = re.compile(r"[\u3400-\u9fffA-Za-z]")
VISIBLE_CHARACTER_PATTERN = re.compile(r"\S")
METADATA_PREFIXES = ("资料标题：", "章节：", "页码：", "内容类型：")
CORRUPTION_MARKERS = ("\ufffd", "锟斤拷", "烫烫烫", "屯屯屯")


def is_meaningful_knowledge_text(text, title=""):
    signal = extract_information_signal(text)
    title_signal = extract_information_signal(title)
    combined = " ".join(part for part in (title_signal, signal) if part).strip()
    if not combined:
        return False
    if any(marker in combined for marker in CORRUPTION_MARKERS):
        return False

    compact = "".join(VISIBLE_CHARACTER_PATTERN.findall(combined))
    if not compact or PURE_INDEX_PATTERN.fullmatch(compact):
        return False

    semantic_characters = SEMANTIC_CHARACTER_PATTERN.findall(compact)
    if len(semantic_characters) < 2:
        return False
    unique_semantic_characters = {item.casefold() for item in semantic_characters}
    if len(unique_semantic_characters) < 2:
        return False
    if (
        len(compact) > 20
        and len(semantic_characters) < 8
        and len(semantic_characters) / len(compact) < 0.2
    ):
        return False
    return True


def extract_information_signal(text):
    lines = []
    for raw_line in str(text or "").splitlines():
        line = " ".join(raw_line.replace("\x00", " ").strip().split())
        if not line or line.startswith(METADATA_PREFIXES):
            continue
        if re.fullmatch(r"\|?(?:\s*:?-+:?\s*\|)+", line):
            continue
        lines.append(line.strip("|-—_ "))
    return " ".join(part for part in lines if part).strip()
