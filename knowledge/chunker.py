# -*- coding: utf-8 -*-

import re
import uuid

from knowledge.models import KnowledgeChunk, KnowledgeElement
from knowledge.quality import is_meaningful_knowledge_text


DEFAULT_SAFETY_MAX_TOKENS = 900
DEFAULT_SHORT_PARAGRAPH_TOKENS = 160
DEFAULT_MERGED_PARAGRAPH_TOKENS = 480
TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?；;])")
CLAUSE_PATTERN = re.compile(r"(?<=[，、,：:])")
INDEXED_HEADING_PATTERN = re.compile(
    r"^(?:第[一二三四五六七八九十百千0-9]+[编章节部分单元课]|"
    r"[一二三四五六七八九十]+[、.．]|"
    r"[0-9]+(?:\.[0-9]+)+[、.．]?)"
)


def estimate_tokens(text):
    return len(TOKEN_PATTERN.findall(str(text or "")))


def build_knowledge_chunks(
    source_id,
    title,
    elements,
    category="",
    source_filename="",
    safety_max_tokens=DEFAULT_SAFETY_MAX_TOKENS,
):
    source_title = normalize_text(title) or "未命名资料"
    chunks = [
        create_chunk(
            source_id=source_id,
            source_title=source_title,
            title=source_title,
            content=f"资料标题：{source_title}",
            chunk_type="title",
            category=category,
            source_filename=source_filename,
        )
    ]
    pending = []
    pending_page = 0
    pending_section = ""
    seen_headings = set()

    def flush_pending():
        nonlocal pending, pending_page, pending_section
        if not pending:
            return
        body = "\n\n".join(item.text for item in pending)
        chunk_type = "paragraph" if len(pending) == 1 else "semantic"
        chunks.append(
            create_contextual_chunk(
                source_id,
                source_title,
                body,
                chunk_type,
                category,
                source_filename,
                pending_page,
                pending_section,
            )
        )
        pending = []
        pending_page = 0
        pending_section = ""

    for element in elements:
        if (
            element.element_type == "image"
            or not normalize_text(element.text)
            or not is_meaningful_knowledge_text(element.text)
        ):
            continue
        if element.element_type == "heading":
            flush_pending()
            normalized_heading = normalize_text(element.text)
            if (
                should_index_heading(element)
                and normalized_heading not in seen_headings
            ):
                seen_headings.add(normalized_heading)
                chunks.append(
                    create_contextual_chunk(
                        source_id,
                        source_title,
                        element.text,
                        "heading",
                        category,
                        source_filename,
                        element.page_number,
                        element.section or element.text,
                    )
                )
            continue
        if element.element_type == "table":
            flush_pending()
            for table_part in split_markdown_table(element.text, safety_max_tokens):
                chunks.append(
                    create_contextual_chunk(
                        source_id,
                        source_title,
                        table_part,
                        "table",
                        category,
                        source_filename,
                        element.page_number,
                        element.section,
                    )
                )
            continue

        semantic_parts = split_semantic_unit(element.text, safety_max_tokens)
        if len(semantic_parts) > 1:
            flush_pending()
            for part in semantic_parts:
                chunks.append(
                    create_contextual_chunk(
                        source_id,
                        source_title,
                        part,
                        "semantic",
                        category,
                        source_filename,
                        element.page_number,
                        element.section,
                    )
                )
            continue

        paragraph = KnowledgeElement(
            element_type="paragraph",
            text=semantic_parts[0],
            page_number=element.page_number,
            section=element.section,
        )
        tokens = estimate_tokens(paragraph.text)
        same_context = (
            not pending
            or (pending_page == paragraph.page_number and pending_section == paragraph.section)
        )
        if not same_context:
            flush_pending()
        if tokens >= DEFAULT_SHORT_PARAGRAPH_TOKENS:
            flush_pending()
            pending = [paragraph]
            pending_page = paragraph.page_number
            pending_section = paragraph.section
            flush_pending()
            continue
        candidate = "\n\n".join([*(item.text for item in pending), paragraph.text])
        if pending and estimate_tokens(candidate) > DEFAULT_MERGED_PARAGRAPH_TOKENS:
            flush_pending()
        pending.append(paragraph)
        pending_page = paragraph.page_number
        pending_section = paragraph.section
        if estimate_tokens("\n\n".join(item.text for item in pending)) >= DEFAULT_MERGED_PARAGRAPH_TOKENS:
            flush_pending()
    flush_pending()
    return chunks


def split_knowledge_text(
    source_id,
    title,
    text,
    category="",
    source_filename="",
    max_tokens=DEFAULT_SAFETY_MAX_TOKENS,
    overlap_tokens=0,
):
    del overlap_tokens
    elements = []
    current_section = ""
    for raw_line in str(text or "").splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        if line.startswith("#"):
            current_section = line.lstrip("# ").strip()
            elements.append(
                KnowledgeElement(element_type="heading", text=current_section, section=current_section)
            )
        else:
            elements.append(
                KnowledgeElement(element_type="paragraph", text=line, section=current_section)
            )
    return build_knowledge_chunks(
        source_id,
        title,
        elements,
        category=category,
        source_filename=source_filename,
        safety_max_tokens=max_tokens,
    )


def split_semantic_unit(text, safety_max_tokens):
    normalized = normalize_text(text)
    if not normalized:
        return []
    if estimate_tokens(normalized) <= safety_max_tokens:
        return [normalized]
    sentences = split_complete_units(normalized, SENTENCE_PATTERN)
    parts = pack_complete_units(sentences, safety_max_tokens)
    result = []
    for part in parts:
        if estimate_tokens(part) <= safety_max_tokens:
            result.append(part)
            continue
        clauses = split_complete_units(part, CLAUSE_PATTERN)
        result.extend(pack_complete_units(clauses, safety_max_tokens, allow_hard_split=True))
    return result


def split_complete_units(text, pattern):
    return [item.strip() for item in pattern.split(text) if item.strip()] or [text]


def pack_complete_units(units, safety_max_tokens, allow_hard_split=False):
    parts = []
    current = ""
    for unit in units:
        if estimate_tokens(unit) > safety_max_tokens and allow_hard_split:
            if current:
                parts.append(current)
                current = ""
            parts.extend(hard_split(unit, safety_max_tokens))
            continue
        candidate = f"{current}{unit}" if current else unit
        if current and estimate_tokens(candidate) > safety_max_tokens:
            parts.append(current)
            current = unit
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def hard_split(text, safety_max_tokens):
    parts = []
    remaining = text
    while estimate_tokens(remaining) > safety_max_tokens:
        ratio = safety_max_tokens / max(estimate_tokens(remaining), 1)
        cut = max(1, int(len(remaining) * ratio))
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def split_markdown_table(text, safety_max_tokens):
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if estimate_tokens("\n".join(lines)) <= safety_max_tokens or len(lines) <= 3:
        return ["\n".join(lines)] if lines else []
    header = lines[:2]
    parts = []
    current_rows = []
    for row in lines[2:]:
        candidate = "\n".join([*header, *current_rows, row])
        if current_rows and estimate_tokens(candidate) > safety_max_tokens:
            parts.append("\n".join([*header, *current_rows]))
            current_rows = [row]
        else:
            current_rows.append(row)
    if current_rows:
        parts.append("\n".join([*header, *current_rows]))
    return parts


def should_index_heading(element):
    heading = normalize_text(element.text)
    return is_meaningful_knowledge_text(heading) and (
        int(element.page_number or 0) == 0
        or bool(INDEXED_HEADING_PATTERN.match(heading))
    )


def create_contextual_chunk(
    source_id,
    source_title,
    body,
    chunk_type,
    category,
    source_filename,
    page_number,
    section,
):
    context_lines = [f"资料标题：{source_title}"]
    if section:
        context_lines.append(f"章节：{section}")
    if page_number:
        context_lines.append(f"页码：{page_number}")
    context_lines.append(f"内容类型：{chunk_type}")
    content = "\n".join([*context_lines, normalize_text_or_table(body)])
    return create_chunk(
        source_id=source_id,
        source_title=source_title,
        title=section or source_title,
        content=content,
        chunk_type=chunk_type,
        category=category,
        source_filename=source_filename,
        page_number=page_number,
        section=section,
    )


def create_chunk(
    source_id,
    source_title,
    title,
    content,
    chunk_type,
    category,
    source_filename,
    page_number=0,
    section="",
):
    return KnowledgeChunk(
        chunk_id=uuid.uuid4().hex,
        source_id=source_id,
        title=normalize_text(title) or source_title,
        content=content.strip(),
        chunk_type=chunk_type,
        category=normalize_text(category),
        source_filename=source_filename,
        source_title=source_title,
        page_number=int(page_number or 0),
        section=normalize_text(section),
    )


def normalize_text(value):
    return " ".join(str(value or "").strip().split())


def normalize_text_or_table(value):
    text = str(value or "").strip()
    if text.startswith("|"):
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return normalize_text(text)
