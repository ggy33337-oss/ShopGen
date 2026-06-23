from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes.chat_routes import router as chat_router
from api.routes.conversation_routes import router as conversation_router
from api.routes.poster_routes import router as poster_router


app = FastAPI(title="电商文案与图片生成助手")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(chat_router)
app.include_router(conversation_router)
app.include_router(poster_router)


@app.get("/")
def index():
    return FileResponse("static/index.html")
