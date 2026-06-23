from poster.models.copywriting import CopywritingResult
from poster.models.generation import ImageGenerationResult
from poster.models.poster_schema import PosterSchema
from poster.models.response import PosterPayload


def build_response(
    poster_schema: PosterSchema,
    copywriting_result: CopywritingResult,
    image_result: ImageGenerationResult,
    generation_time,
    conversation_id="default",
):
    return PosterPayload(
        copywriting=copywriting_result.copywriting,
        poster=image_result,
        metadata={
            "style": poster_schema.style,
            "layout": poster_schema.layout,
            "generation_time": generation_time,
        },
        conversation_id=conversation_id,
    )
