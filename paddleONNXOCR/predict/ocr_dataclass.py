import json
import numpy
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class OCRChunkResult:
    """OCR结果块"""
    text: str = ""
    confidence: float = 0.0
    box: List[List[float]] | None = None
    angle: str = "0_degree"
    angle_confidence: float = 1.0

    async def to_json(
            self,
            ensure_ascii=False,
            indent=4
    ):
        """
        OCR结果块转json
        """
        return json.dumps(asdict(self), ensure_ascii=ensure_ascii, indent=indent)

    async def to_dict(self):
        """
        OCR结果块转dict
        """
        return asdict(self)


@dataclass
class OCRResult:
    """OCR完整结果"""
    text: str = ""
    json: str = ""
    results: List[OCRChunkResult] | None = None
    image: numpy.ndarray | None = None


@dataclass
class PdfPageResult:
    """pdf结果"""
    page_index: int = 0
    text: str = ""
    ocr_data: List[OCRResult] | None = None
