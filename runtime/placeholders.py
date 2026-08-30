# -*- coding: utf-8 -*-

from dataclasses import dataclass

from knowledge.models import KnowledgeContext


@dataclass(frozen=True)
class PlaceholderChainResult:
    reply_text: str
    status: str = "placeholder"


class PlaceholderTextToImageChain:
    """无参考图的检索增强生图链路占位。"""

    def run(self, user_input: str, knowledge: KnowledgeContext) -> PlaceholderChainResult:
        del user_input, knowledge
        return PlaceholderChainResult(
            reply_text=(
                "当前请求需要走无参考图的链路三。该生成链路目前为占位实现，"
                "暂不调用生图模型；上传一张参考图后可走链路二，或基于会话中的历史图走链路一。"
            )
        )
