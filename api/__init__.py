from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from paddleONNXOCR import PredictSystem, ImageTableToHTML

ocr_predict_system = None
ocr_image_table_html = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_predict_system, ocr_image_table_html
    ocr_predict_system = PredictSystem()
    await ocr_predict_system.__aenter__()
    ocr_image_table_html = ImageTableToHTML()
    await ocr_image_table_html.__aenter__()
    yield
    await ocr_predict_system.__aexit__(None, None, None)
    await ocr_image_table_html.__aexit__(None, None, None)


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


async def get_ocr_image_table_html():
    return ocr_image_table_html
