from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.dtos.poster import PosterResponseDTO
from poster.models.request import PosterGenerationRequest
from services.poster_service import generate_poster


router = APIRouter(prefix="/api/poster", tags=["poster"])


@router.post("/generate", response_model=PosterResponseDTO)
async def generate_poster_route(
    message: str = Form(..., min_length=1, max_length=1000),
    conversation_id: str = Form(default="default", max_length=64),
    poster_type: str = Form(default="商业海报", max_length=80),
    campaign: str = Form(default="", max_length=1000),
    target_audience: str = Form(default="", max_length=120),
    file: UploadFile | None = File(default=None),
):
    request = PosterGenerationRequest(
        user_text=message,
        conversation_id=conversation_id,
        poster_type=poster_type,
        campaign=campaign,
        target_audience=target_audience,
        **await read_upload_file(file),
    )

    try:
        return generate_poster(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def read_upload_file(file: UploadFile | None):
    if not file or not file.filename:
        return {
            "file_name": None,
            "file_content_type": "",
            "file_content": None,
        }

    return {
        "file_name": file.filename,
        "file_content_type": file.content_type or "",
        "file_content": await file.read(),
    }
