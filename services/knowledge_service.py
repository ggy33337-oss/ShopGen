# -*- coding: utf-8 -*-

import base64
import hashlib
import uuid
from pathlib import Path

from core.config import read_config
from knowledge.chunker import build_knowledge_chunks
from knowledge.document_parser import parse_knowledge_file
from knowledge.models import KnowledgeImageRecord, KnowledgeUploadResult
from knowledge.qdrant_knowledge_base import QdrantKnowledgeBase


DEFAULT_MAX_KNOWLEDGE_FILE_MB = 100


class KnowledgeService:
    def __init__(self, values=None, knowledge_base=None):
        self.values = values or read_config(".env")
        self.knowledge_base = knowledge_base or QdrantKnowledgeBase(self.values)
        self.storage_root = Path(
            self.values.get("KNOWLEDGE_STORAGE_DIR") or "data/knowledge"
        ).resolve()
        self.max_file_mb = parse_positive_integer(
            self.values.get("KNOWLEDGE_MAX_FILE_MB"),
            DEFAULT_MAX_KNOWLEDGE_FILE_MB,
        )
        self.max_file_bytes = self.max_file_mb * 1024 * 1024
        self.max_document_images = parse_positive_integer(
            self.values.get("KNOWLEDGE_MAX_DOCUMENT_IMAGES"),
            6,
        )
        self.max_document_pages = parse_positive_integer(
            self.values.get("KNOWLEDGE_MAX_DOCUMENT_PAGES"),
            300,
        )
        self.max_text_characters = parse_positive_integer(
            self.values.get("KNOWLEDGE_MAX_TEXT_CHARACTERS"),
            500000,
        )

    def upload(self, filename, content_type, content, title="", category=""):
        if not content:
            raise ValueError("知识资料文件不能为空。")
        if len(content) > self.max_file_bytes:
            raise ValueError(f"知识资料文件不能超过 {self.max_file_mb} MB。")
        parsed = parse_knowledge_file(
            filename,
            content_type,
            content,
            max_images=self.max_document_images,
            max_pages=self.max_document_pages,
            max_text_characters=self.max_text_characters,
        )
        if not parsed:
            raise ValueError("知识资料解析失败。")

        source_id = hashlib.sha256(content).hexdigest()[:32]
        normalized_title = " ".join(str(title or Path(filename).stem or "未命名资料").split())[:255]
        normalized_category = " ".join(str(category or "").split())[:100]
        self._save_original(source_id, parsed.extension, content)

        chunks = build_knowledge_chunks(
            source_id=source_id,
            title=normalized_title,
            elements=parsed.text_elements,
            category=normalized_category or ("图片素材" if parsed.is_image else "文档资料"),
            source_filename=filename,
        )
        image_records = []
        for index, element in enumerate(parsed.image_elements):
            image_title = (
                normalized_title
                if len(parsed.image_elements) == 1
                else f"{normalized_title} - 图片 {index + 1}"
            )
            image_category = normalized_category or "图片素材"
            analysis = self.knowledge_base._get_retrieval_gateway().describe_image(
                element.image_data_url,
                title=image_title,
                category=image_category,
                context=element.context,
            )
            description_parts = [analysis["description"]]
            if element.section:
                description_parts.append(f"所在章节：{element.section}")
            if element.page_number:
                description_parts.append(f"所在页码：{element.page_number}")
            if element.context:
                description_parts.append(f"邻近文字：{element.context[:1000]}")
            description = "\n".join(part for part in description_parts if part)
            self._save_data_url(source_id, index, element.image_data_url)
            image_records.append(
                KnowledgeImageRecord(
                    source_id=source_id,
                    image_data_url=element.image_data_url,
                    title=analysis["title"],
                    category=analysis["category"],
                    description=description,
                    source_filename=filename,
                    source_title=normalized_title,
                    page_number=element.page_number,
                    section=element.section,
                    tags=tuple(analysis["tags"]),
                )
            )

        if not chunks and not image_records:
            raise ValueError("知识资料中没有可向量化的标题、段落、表格或图片。")
        try:
            text_count, image_count = self.knowledge_base.replace_source(
                chunks,
                image_records,
            )
        except Exception as exc:
            raise RuntimeError(f"知识资料向量化失败：{exc}") from exc
        return KnowledgeUploadResult(
            source_id=source_id,
            title=normalized_title,
            category=normalized_category,
            text_vector_count=text_count,
            image_vector_count=image_count,
        )

    def search(self, query, limit=5, category=""):
        try:
            return self.knowledge_base.search_matches(query, limit=limit, category=category)
        except Exception as exc:
            raise RuntimeError(f"知识库检索失败：{exc}") from exc

    def list_sources(self, limit=100):
        try:
            return self.knowledge_base.list_sources(limit=limit)
        except Exception as exc:
            raise RuntimeError(f"知识库读取失败：{exc}") from exc

    def _save_original(self, source_id, extension, content):
        directory = self._source_directory(source_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"original{extension}").write_bytes(content)

    def _save_data_url(self, source_id, index, data_url):
        header, separator, encoded = str(data_url).partition(",")
        if separator != "," or ";base64" not in header:
            return
        extension = ".jpg" if "jpeg" in header else ".png"
        content = base64.b64decode(encoded, validate=True)
        directory = self._source_directory(source_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"image-{index + 1}{extension}").write_bytes(content)

    def _source_directory(self, source_id):
        safe_source_id = str(source_id or "").strip().lower()
        if len(safe_source_id) != 32 or any(char not in "0123456789abcdef" for char in safe_source_id):
            safe_source_id = uuid.uuid4().hex
        return self.storage_root / safe_source_id


def parse_positive_integer(value, default):
    try:
        parsed = int(str(value or "").strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default
