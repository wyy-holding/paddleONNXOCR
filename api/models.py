from typing import List, Dict, Any
from pydantic import BaseModel, Field

from paddleONNXOCR.predict.predict_system import OCRChunkResult


class BaseResponse(BaseModel):
    success: bool = Field(default=True, description="接口成功还是失败")
    code: int = Field(default=200, description="状态码")
    data: List[Dict[str, Any]] = Field(default=[], description="识别结果列表")
    message: str = Field(default="", description="提示信息")


class OCRResultData(BaseModel):
    text: str | None = Field(default=None, description="ocr识别结果纯文本内容")
    data: List[OCRChunkResult] = Field(default="", description="识别结果列表")


class OCRInput(BaseModel):
    content: List[str] = Field(..., description="需要识别的内容，在线图片地址和base64图片")


class OCRResponse(BaseResponse):
    data: List[OCRResultData] = Field(default=[], description="ocr列表识别结果")
