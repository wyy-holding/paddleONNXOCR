from typing import List, AsyncGenerator
from fastapi import APIRouter
from api import get_ocr_predict_system
from api.models import OCRInput, OCRResponse, OCRResultData
from paddleONNXOCR.predict.predict_system import OCRResults


class OCRServices:
    @staticmethod
    async def ocr_predict(content: List[str]) -> AsyncGenerator[OCRResults, None]:
        ocr_system = await get_ocr_predict_system()
        async for result in ocr_system.predict_batch(content):
            yield result

    @staticmethod
    async def universalRapidOcr(user_input: OCRInput):
        ocr_response: OCRResponse = OCRResponse()
        async for ocr_result in OCRServices.ocr_predict(user_input.content):
            ocr_response.data.append(
                OCRResultData(text=ocr_result.text, data=ocr_result.results)
            )
        return ocr_response


ocrRouter = APIRouter(tags=["OCR"])
# 通用ocr
ocrRouter.add_api_route(
    path="/universalOcr",
    endpoint=OCRServices.universalRapidOcr,
    methods=["POST"],
    response_model=OCRResponse
)
