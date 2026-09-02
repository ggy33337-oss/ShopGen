# -*- coding: utf-8 -*-

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes.chat_routes import router as chat_router
from api.routes.conversation_routes import router as conversation_router
from api.routes.image_routes import router as image_router
from api.routes.generation_log_routes import router as generation_log_router
from api.routes.knowledge_routes import router as knowledge_router
from api.routes.poster_routes import router as poster_router
from infra.database import dispose_async_engine


@asynccontextmanager
async def lifespan(_app):
    yield
    await dispose_async_engine()


app = FastAPI(title="电商文案与图片生成助手", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(chat_router)
app.include_router(conversation_router)
app.include_router(image_router)
app.include_router(generation_log_router)
app.include_router(knowledge_router)
app.include_router(poster_router)


@app.get("/")
def index():
    return FileResponse("static/index.html")
