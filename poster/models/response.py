from pydantic import BaseModel

from poster.models.copywriting import Copywriting
from poster.models.generation import ImageGenerationResult


class PosterPayload(BaseModel):
    copywriting: Copywriting
    poster: ImageGenerationResult
    metadata: dict[str, str]
    conversation_id: str = "default"
