# -*- coding: utf-8 -*-

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeElement:
    element_type: str
    text: str = ""
    page_number: int = 0
    section: str = ""
    image_data_url: str = ""
    context: str = ""


@dataclass(frozen=True)
class ParsedKnowledgeFile:
    filename: str
    extension: str
    elements: tuple[KnowledgeElement, ...] = ()

    @property
    def is_image(self):
        return self.extension in {".png", ".jpg", ".jpeg"}

    @property
    def text_elements(self):
        return tuple(item for item in self.elements if item.element_type != "image")

    @property
    def image_elements(self):
        return tuple(item for item in self.elements if item.element_type == "image")


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source_id: str
    title: str
    content: str
    chunk_type: str
    category: str = ""
    source_filename: str = ""
    source_title: str = ""
    page_number: int = 0
    section: str = ""


@dataclass(frozen=True)
class KnowledgeImageRecord:
    source_id: str
    image_data_url: str
    title: str
    category: str
    description: str
    source_filename: str
    source_title: str = ""
    page_number: int = 0
    section: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeContext:
    query: str
    status: str
    context_text: str = ""
    matches: tuple[dict, ...] = ()


@dataclass(frozen=True)
class KnowledgeUploadResult:
    source_id: str
    title: str
    category: str
    text_vector_count: int
    image_vector_count: int
