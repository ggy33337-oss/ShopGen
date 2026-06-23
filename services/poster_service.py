import time

from core.config import read_config
from infra.conversation_store import (
    append_messages,
    append_visual_history,
    normalize_conversation_id,
)
from infra.logger import write_log
from poster.copywriting_agent import generate_copywriting
from poster.file_parser import parse_uploaded_file
from poster.image_generator import generate_poster_image
from poster.models.request import PosterGenerationRequest
from poster.multimodal_analyzer import ANALYZER_MODEL_FALLBACK, analyze_multimodal
from poster.prompt_builder import build_final_prompt
from poster.response_builder import build_response
from poster.schema_builder import build_poster_schema


class PosterService:
    def __init__(self, values):
        self.values = values

    def generate(self, request: PosterGenerationRequest):
        start_time = time.perf_counter()
        conversation_id = normalize_conversation_id(request.conversation_id)

        analyzer_result = self.run_multimodal_analyzer(request)
        poster_schema = self.build_poster_schema(request, analyzer_result)
        copywriting_result = self.run_copywriting_agent(poster_schema)
        final_prompt = self.run_prompt_builder(poster_schema, copywriting_result)
        image_result = self.run_image_generator(final_prompt)

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        response = self.run_response_builder(
            poster_schema=poster_schema,
            copywriting_result=copywriting_result,
            image_result=image_result,
            latency_ms=latency_ms,
            conversation_id=conversation_id,
        )

        self.write_conversation_memory(
            conversation_id=conversation_id,
            request=request,
            response=response,
            final_prompt=final_prompt,
            image_url=image_result.image_url,
        )
        self.write_success_log(
            conversation_id=conversation_id,
            request=request,
            poster_schema=poster_schema,
            copywriting_result=copywriting_result,
            final_prompt=final_prompt,
            image_url=image_result.image_url,
            latency_ms=latency_ms,
        )
        return response

    def run_multimodal_analyzer(self, request: PosterGenerationRequest):
        uploaded_content = self.parse_uploaded_content(request)
        return analyze_multimodal(
            self.values,
            request.user_text,
            uploaded_content,
        )

    def build_poster_schema(self, request: PosterGenerationRequest, analyzer_result):
        return build_poster_schema(
            user_text=request.user_text,
            analyzer_result=analyzer_result,
            poster_type=request.poster_type,
            campaign=request.campaign,
            target_audience=request.target_audience,
        )

    def run_copywriting_agent(self, poster_schema):
        return generate_copywriting(self.values, poster_schema)

    def run_prompt_builder(self, poster_schema, copywriting_result):
        return build_final_prompt(poster_schema, copywriting_result)

    def run_image_generator(self, final_prompt):
        return generate_poster_image(self.values, final_prompt)

    def run_response_builder(
        self,
        poster_schema,
        copywriting_result,
        image_result,
        latency_ms,
        conversation_id,
    ):
        return build_response(
            poster_schema=poster_schema,
            copywriting_result=copywriting_result,
            image_result=image_result,
            generation_time=f"{latency_ms} ms",
            conversation_id=conversation_id,
        )

    def parse_uploaded_content(self, request: PosterGenerationRequest):
        if not request.file_content:
            return None
        return parse_uploaded_file(
            request.file_name,
            request.file_content_type,
            request.file_content,
        )

    def write_success_log(
        self,
        conversation_id,
        request,
        poster_schema,
        copywriting_result,
        final_prompt,
        image_url,
        latency_ms,
    ):
        write_log({
            "pipeline": "poster_schema",
            "conversation_id": conversation_id,
            "user_input": request.user_text,
            "poster_schema": poster_schema.model_dump(),
            "copywriting": copywriting_result.model_dump(),
            "final_prompt": final_prompt.final_prompt,
            "image_url": image_url,
            "latency_ms": latency_ms,
            "analyzer_model": self.values.get("POSTER_ANALYZER_MODEL_NAME") or ANALYZER_MODEL_FALLBACK,
            "copywriting_model": self.values.get("POSTER_COPYWRITING_MODEL_NAME") or self.values.get("MODEL_NAME", ""),
            "image_model": self.values.get("IMAGE_MODEL_NAME", ""),
        })

    def write_conversation_memory(
        self,
        conversation_id,
        request,
        response,
        final_prompt,
        image_url,
    ):
        assistant_text = self.build_assistant_text(response)
        user_text = request.user_text
        if request.file_name:
            user_text = f"{user_text}\n附件：{request.file_name}"

        if image_url:
            append_visual_history(conversation_id, {
                "image_url": image_url,
                "user_input": request.user_text,
                "assistant_text": assistant_text,
                "image_prompt": final_prompt.final_prompt,
            })

        append_messages(conversation_id, [
            {"role": "user", "content": user_text},
            {
                "role": "assistant",
                "content": assistant_text,
                "image_url": image_url,
            },
        ])

    def build_assistant_text(self, response):
        copywriting = response.copywriting
        lines = [
            f"主标题：{copywriting.headline}" if copywriting.headline else "",
            f"副标题：{copywriting.subheadline}" if copywriting.subheadline else "",
            f"行动语：{copywriting.cta}" if copywriting.cta else "",
        ]
        return "\n".join(line for line in lines if line) or "已生成海报方案。"


def generate_poster(request: PosterGenerationRequest):
    return PosterService(read_config(".env")).generate(request)


def generate_schema_driven_poster(
    user_text,
    conversation_id="default",
    file_name=None,
    file_content_type="",
    file_content=None,
    poster_type="商业海报",
    campaign="",
    target_audience="",
):
    request = PosterGenerationRequest(
        user_text=user_text,
        conversation_id=conversation_id,
        file_name=file_name,
        file_content_type=file_content_type,
        file_content=file_content,
        poster_type=poster_type,
        campaign=campaign,
        target_audience=target_audience,
    )
    return generate_poster(request)
