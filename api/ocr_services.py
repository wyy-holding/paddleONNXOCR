from fastapi import APIRouter
from api import get_ocr_predict_system
from api.models import OCRInput, OCRResponse, OCRResultData
from paddleONNXOCR.predict.predict_system import OCRResult


class OCRServices:

    @classmethod
    async def universalOcr(cls, user_input: OCRInput):
        ocr_response: OCRResponse = OCRResponse()
        ocr_system = await get_ocr_predict_system()
        ocr_response.data = await ocr_system.predict_batch(user_input.content)
        return ocr_response


ocrRouter = APIRouter(tags=["OCR"])
# 通用ocr
ocrRouter.add_api_route(
    path="/universalOcr",
    endpoint=OCRServices.universalOcr,
    methods=["POST"],
    response_model=OCRResponse
)
