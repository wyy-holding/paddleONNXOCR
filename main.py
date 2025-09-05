from api import app
from api.ocr_services import ocrRouter

app.include_router(ocrRouter)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
