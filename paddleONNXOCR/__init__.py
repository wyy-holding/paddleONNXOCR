from paddleONNXOCR.predict.predict_system import PredictSystem
from paddleONNXOCR.models_enum import *
from paddleONNXOCR.predict.predict_cls import TextLineOrientationDetector
from paddleONNXOCR.predict.predict_det import TextDetector
from paddleONNXOCR.predict.predict_rec import OCRRecognizer
from paddleONNXOCR.predict.predict_doc_cls import DocumentOrientationDetector
from paddleONNXOCR.predict.predict_uvdoc import DocumentRectifier

__author__ = 'wyy-holding'
__version__ = '0.0.5'
__all__ = [
    "PredictSystem",
    "TextLineOrientationDetector",
    "TextDetector",
    "OCRRecognizer",
    "DocumentOrientationDetector",
    "DocumentRectifier",
    "TextLineModels",
    "DetModels",
    "RecModels",
    "TableModels",
    "ImageModels"
]
