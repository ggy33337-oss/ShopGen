# -*- coding: utf-8 -*-

import base64
import hashlib
import re
from io import BytesIO
from pathlib import Path

from knowledge.models import KnowledgeElement, ParsedKnowledgeFile


SUPPORTED_KNOWLEDGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".docx"}
DEFAULT_MAX_DOCUMENT_IMAGES = 6
DEFAULT_MAX_DOCUMENT_PAGES = 300
DEFAULT_MAX_TEXT_CHARACTERS = 500000
DEFAULT_MAX_IMAGE_EDGE = 1280
DEFAULT_MAX_IMAGE_BYTES = 900 * 1024
HEADING_PATTERN = re.compile(
    r"^(?:第[一二三四五六七八九十百千0-9]+[编章节部分单元课].*|"
    r"[一二三四五六七八九十]+[、.．].+|"
    r"[0-9]+(?:\.[0-9]+)*[、.．]\s*.+)$"
)


def parse_knowledge_file(
    filename,
    content_type,
    content,
    max_images=DEFAULT_MAX_DOCUMENT_IMAGES,
    max_pages=DEFAULT_MAX_DOCUMENT_PAGES,
    max_text_characters=DEFAULT_MAX_TEXT_CHARACTERS,
):
    del content_type
    if not content:
        return None
    extension = Path(str(filename or "")).suffix.lower()
    if extension not in SUPPORTED_KNOWLEDGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_KNOWLEDGE_EXTENSIONS))
        raise ValueError(f"不支持的文件类型：{extension or '未知'}。当前支持：{supported}")
    if extension in {".png", ".jpg", ".jpeg"}:
        data_url = normalize_image_data_url(content)
        if not data_url:
            raise ValueError("图片资料无法解析。")
        return ParsedKnowledgeFile(
            filename=filename,
            extension=extension,
            elements=(KnowledgeElement(element_type="image", image_data_url=data_url),),
        )
    if extension == ".pdf":
        elements = parse_pdf_elements(content, max_images, max_pages, max_text_characters)
    else:
        elements = parse_docx_elements(content, max_images, max_text_characters)
    return ParsedKnowledgeFile(filename=filename, extension=extension, elements=tuple(elements))


def parse_pdf_elements(content, max_images, max_pages, max_text_characters):
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("缺少 PDF 结构化解析依赖 PyMuPDF，请先安装 requirements.txt。") from exc

    elements = []
    seen_images = set()
    current_section = ""
    text_characters = 0
    with pymupdf.open(stream=content, filetype="pdf") as document:
        toc_by_page = build_toc_by_page(document.get_toc())
        page_limit = min(document.page_count, max_pages)
        for page_index in range(page_limit):
            page = document[page_index]
            page_number = page_index + 1
            if toc_by_page.get(page_number):
                current_section = toc_by_page[page_number]
                elements.append(
                    KnowledgeElement(
                        element_type="heading",
                        text=current_section,
                        page_number=page_number,
                        section=current_section,
                    )
                )

            page_dict = page.get_text("dict", sort=True)
            body_font_size = estimate_body_font_size(page_dict)
            tables = extract_pdf_tables(page)
            table_rects = [item[0] for item in tables]
            page_items = []

            for block in page_dict.get("blocks", []):
                block_type = int(block.get("type", -1))
                bbox = tuple(block.get("bbox") or (0, 0, 0, 0))
                if block_type == 0:
                    if is_page_margin_text(bbox, page.rect):
                        continue
                    if overlaps_table(bbox, table_rects):
                        continue
                    text = extract_text_from_pdf_block(block)
                    if not text:
                        continue
                    font_size = max_font_size(block)
                    element_type = (
                        "heading"
                        if looks_like_pdf_heading(text, font_size, body_font_size)
                        else "paragraph"
                    )
                    page_items.append((bbox[1], bbox[0], element_type, text, "", bbox))
                elif block_type == 1 and len(seen_images) < max_images:
                    image_bytes = block.get("image") or b""
                    digest = hashlib.sha256(image_bytes).hexdigest() if image_bytes else ""
                    if not is_meaningful_image(block, page.rect) or not digest or digest in seen_images:
                        continue
                    data_url = normalize_image_data_url(image_bytes)
                    if not data_url:
                        continue
                    seen_images.add(digest)
                    page_items.append((bbox[1], bbox[0], "image", "", data_url, bbox))

            for table_rect, table_text in tables:
                page_items.append((table_rect[1], table_rect[0], "table", table_text, "", table_rect))
            page_items.sort(key=lambda item: (item[0], item[1]))

            text_items = [item for item in page_items if item[2] in {"heading", "paragraph", "table"}]
            for _, _, element_type, text, data_url, bbox in page_items:
                if element_type == "image":
                    elements.append(
                        KnowledgeElement(
                            element_type="image",
                            page_number=page_number,
                            section=current_section,
                            image_data_url=data_url,
                            context=nearby_pdf_text(text_items, bbox),
                        )
                    )
                    continue
                if text_characters >= max_text_characters:
                    continue
                text = text[: max_text_characters - text_characters]
                if not text:
                    continue
                if element_type == "heading":
                    current_section = text
                elements.append(
                    KnowledgeElement(
                        element_type=element_type,
                        text=text,
                        page_number=page_number,
                        section=current_section,
                    )
                )
                text_characters += len(text)
    return elements


def parse_docx_elements(content, max_images, max_text_characters):
    try:
        from docx import Document
        from docx.table import Table
    except ImportError as exc:
        raise RuntimeError("缺少 DOCX 结构化解析依赖 python-docx，请先安装 requirements.txt。") from exc

    document = Document(BytesIO(content))
    elements = []
    seen_images = set()
    current_section = ""
    recent_text = []
    text_characters = 0
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            text = table_rows_to_markdown(
                [[normalize_text(cell.text) for cell in row.cells] for row in block.rows]
            )
            if text and text_characters < max_text_characters:
                text = text[: max_text_characters - text_characters]
                elements.append(
                    KnowledgeElement(element_type="table", text=text, section=current_section)
                )
                text_characters += len(text)
                recent_text.append(text[:500])
            continue

        text = normalize_text(block.text)
        style_name = str(getattr(getattr(block, "style", None), "name", "") or "")
        element_type = "heading" if is_docx_heading(style_name, text) else "paragraph"
        if text and text_characters < max_text_characters:
            text = text[: max_text_characters - text_characters]
            if element_type == "heading":
                current_section = text
            elements.append(
                KnowledgeElement(element_type=element_type, text=text, section=current_section)
            )
            text_characters += len(text)
            recent_text.append(text[:500])

        for relation_id in block._p.xpath(".//a:blip/@r:embed"):
            if len(seen_images) >= max_images or relation_id in seen_images:
                continue
            image_part = document.part.related_parts.get(relation_id)
            if not image_part:
                continue
            data_url = normalize_image_data_url(image_part.blob)
            if not data_url:
                continue
            seen_images.add(relation_id)
            elements.append(
                KnowledgeElement(
                    element_type="image",
                    section=current_section,
                    image_data_url=data_url,
                    context="\n".join(recent_text[-2:])[:1000],
                )
            )
    return elements


def extract_pdf_tables(page):
    tables = []
    try:
        finder = page.find_tables()
    except Exception:
        return tables
    for table in finder.tables:
        text = table_rows_to_markdown(table.extract())
        if text:
            tables.append((tuple(table.bbox), text))
    return tables


def table_rows_to_markdown(rows):
    normalized_rows = []
    width = 0
    for row in rows or []:
        cells = [normalize_table_cell(cell) for cell in (row or [])]
        if not any(cells):
            continue
        width = max(width, len(cells))
        normalized_rows.append(cells)
    if not normalized_rows or width == 0:
        return ""
    padded = [row + [""] * (width - len(row)) for row in normalized_rows]
    lines = ["| " + " | ".join(padded[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return "\n".join(lines)


def normalize_image_data_url(content):
    try:
        from PIL import Image, ImageOps

        image = ImageOps.exif_transpose(Image.open(BytesIO(content)))
        image.load()
        if (
            len(content) <= DEFAULT_MAX_IMAGE_BYTES
            and max(image.size) <= DEFAULT_MAX_IMAGE_EDGE
            and image.format in {"PNG", "JPEG"}
        ):
            media_type = "image/png" if image.format == "PNG" else "image/jpeg"
            return build_data_url(media_type, content)
        image.thumbnail((DEFAULT_MAX_IMAGE_EDGE, DEFAULT_MAX_IMAGE_EDGE))
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, "white")
            background.paste(image.convert("RGB"), mask=image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
        output = BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True)
        return build_data_url("image/jpeg", output.getvalue())
    except Exception:
        return ""


def build_data_url(media_type, content):
    return f"data:{media_type};base64,{base64.b64encode(content).decode('ascii')}"


def build_toc_by_page(toc):
    result = {}
    stack = []
    for item in toc or []:
        if len(item) < 3:
            continue
        level, title, page_number = int(item[0]), normalize_text(item[1]), int(item[2])
        if not title or page_number <= 0:
            continue
        stack = stack[: max(level - 1, 0)]
        stack.append(title)
        result[page_number] = " / ".join(stack)
    return result


def extract_text_from_pdf_block(block):
    lines = []
    for line in block.get("lines", []):
        text = "".join(str(span.get("text") or "") for span in line.get("spans", []))
        text = normalize_text(text)
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def estimate_body_font_size(page_dict):
    samples = []
    for block in page_dict.get("blocks", []):
        if int(block.get("type", -1)) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = normalize_text(span.get("text"))
                size = float(span.get("size") or 0)
                if text and size > 0:
                    samples.extend([size] * min(len(text), 50))
    if not samples:
        return 10.0
    samples.sort()
    return samples[len(samples) // 2]


def max_font_size(block):
    sizes = [
        float(span.get("size") or 0)
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]
    return max(sizes, default=0.0)


def looks_like_pdf_heading(text, font_size, body_font_size):
    compact = normalize_text(text)
    if not compact or len(compact) > 72 or compact.endswith(("。", "！", "？", ".", "：", ":")):
        return False
    if HEADING_PATTERN.match(compact):
        return True
    chinese_count = len(re.findall(r"[\u3400-\u9fff]", compact))
    visible_count = len(re.sub(r"\s+", "", compact))
    chinese_ratio = chinese_count / max(visible_count, 1)
    return font_size >= body_font_size * 1.45 and chinese_ratio >= 0.5


def is_docx_heading(style_name, text):
    normalized_style = str(style_name or "").strip().lower()
    return bool(text) and (
        normalized_style.startswith("heading")
        or normalized_style.startswith("标题")
        or bool(HEADING_PATTERN.match(text))
    )


def overlaps_table(bbox, table_rects):
    x0, y0, x1, y1 = bbox
    area = max((x1 - x0) * (y1 - y0), 1)
    for tx0, ty0, tx1, ty1 in table_rects:
        intersection = max(0, min(x1, tx1) - max(x0, tx0)) * max(0, min(y1, ty1) - max(y0, ty0))
        if intersection / area >= 0.5:
            return True
    return False


def is_page_margin_text(bbox, page_rect):
    _, y0, _, y1 = bbox
    top_margin = max(float(page_rect.height) * 0.035, 24)
    bottom_margin = float(page_rect.height) - max(float(page_rect.height) * 0.035, 24)
    return y1 <= top_margin or y0 >= bottom_margin


def is_meaningful_image(block, page_rect):
    width = int(block.get("width") or 0)
    height = int(block.get("height") or 0)
    x0, y0, x1, y1 = tuple(block.get("bbox") or (0, 0, 0, 0))
    displayed_area = max(0, x1 - x0) * max(0, y1 - y0)
    page_area = max(float(page_rect.width * page_rect.height), 1)
    return width >= 160 and height >= 120 and displayed_area / page_area >= 0.015


def nearby_pdf_text(text_items, image_bbox):
    image_y = image_bbox[1]
    ranked = sorted(
        text_items,
        key=lambda item: min(abs(item[5][3] - image_y), abs(item[5][1] - image_bbox[3])),
    )
    return "\n".join(item[3] for item in ranked[:2])[:1000]


def normalize_table_cell(value):
    return normalize_text(value).replace("|", "\\|")


def normalize_text(value):
    return " ".join(str(value or "").strip().split())
