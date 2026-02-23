from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from app.routers import users, children, reports
from app.services.file_service import get_file_from_gridfs
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="insoz_api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # или ["http://localhost:5173"] если хочешь ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(children.router)
app.include_router(reports.router)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.get("/pictures/{picture_id}")
async def get_picture(picture_id: str):
    file_data = await get_file_from_gridfs(picture_id)
    if file_data is None:
        raise HTTPException(status_code=404, detail="Файл не найден")

    # Возвращаем бинарные данные с правильным заголовком
    return Response(content=file_data, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
