from paddleONNXOCR.predict.predict_system import PredictSystem
from paddleONNXOCR.models_enum import *
from paddleONNXOCR.predict.predict_cls import TextLineOrientationDetector
from paddleONNXOCR.predict.predict_det import TextDetector
from paddleONNXOCR.predict.predict_rec import OCRRecognizer
from paddleONNXOCR.predict.predict_doc_cls import DocumentOrientationDetector
from paddleONNXOCR.predict.predict_uvdoc import DocumentRectifier
from paddleONNXOCR.predict.predict_layout import LayoutDetection
from paddleONNXOCR.predict.predict_table_cell import TableCellDetector
from paddleONNXOCR.predict.predict_table_cls import TableClassifier

__author__ = 'wyy-holding'
__version__ = '0.0.9'
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
    "ImageModels",
    "LayoutDetection",
    "TableCellDetector",
    "TableClassifier"
]
