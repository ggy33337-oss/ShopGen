import json

from poster.models.copywriting import CopywritingResult
from poster.models.generation import FinalPromptResult
from poster.models.poster_schema import PosterSchema


def build_final_prompt(poster_schema: PosterSchema, copywriting_result: CopywritingResult):
    schema_json = json.dumps(poster_schema.model_dump(), ensure_ascii=False)
    image_prompt_json = json.dumps(copywriting_result.image_prompt.model_dump(), ensure_ascii=False)
    copywriting_json = json.dumps(copywriting_result.copywriting.model_dump(), ensure_ascii=False)

    final_prompt = (
        "Create a polished e-commerce poster image.\n"
        f"User original requirement and campaign context: {poster_schema.campaign}\n"
        f"Poster Schema: {schema_json}\n"
        f"Structured image prompt: {image_prompt_json}\n"
        f"Poster copy to include when readable text is appropriate: {copywriting_json}\n"
        "Design requirements: strong product focus, commercial composition, clean hierarchy, "
        "coherent color palette, premium lighting, no distorted product, no messy typography, "
        "no watermark, no extra logos unless requested."
    )
    return FinalPromptResult(final_prompt=final_prompt)
