from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from paddleONNXOCR.predict.predict_system import PredictSystem

ocr_predict_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_predict_system
    ocr_predict_system = PredictSystem()
    await ocr_predict_system.__aenter__()
    yield
    await ocr_predict_system.__aexit__(None, None, None)


app = FastAPI(
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


async def get_ocr_predict_system():
    return ocr_predict_system
