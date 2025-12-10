import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy
from typing import List, Union, Optional

import onnxruntime
from PIL import Image
from paddleONNXOCR.predict.predict_uvdoc import DocumentRectifier
from paddleONNXOCR.image_loader import ImageLoader
from paddleONNXOCR.models_enum import TableModels, ImageModels
from paddleONNXOCR.predict.ocr_dataclass import TableHtml
from paddleONNXOCR.predict.predict_doc_cls import DocumentOrientationDetector
from paddleONNXOCR.predict.predict_system import PredictSystem
from paddleONNXOCR.predict.predict_table_cls import TableClassifier
from paddleONNXOCR.predict.predict_table_cell import TableCellDetector
from paddleONNXOCR.predict.predict_table import TableDetector
from paddleONNXOCR.utils import TableHTMLGenerator
from paddleONNXOCR.image_rotate import ImageRotate


class ImageTableToHTML:
    """检测图片中的表格并转换为HTML - 组合多个PredictBase子类"""

    def __init__(
            self,
            table_detector: TableDetector = None,
            ocr_system: PredictSystem = None,
            table_classifier: TableClassifier = None,
            table_cell_detector_wired: TableCellDetector = None,
            table_cell_detector_wireless: TableCellDetector = None,
            doc_cls_model: DocumentOrientationDetector = None,
            uvdoc_model: DocumentRectifier = None,
            uvdoc_model_name: ImageModels = ImageModels.UVDOC,
            uvdoc_model_path: str | None = None,
            model_local_dir: str = "models",
            providers: Optional[List[str]] = None,
            session_options: Optional[onnxruntime.SessionOptions] = None,
            executor: Optional[ThreadPoolExecutor] = None,
            # 如果需要自定义表格检测模型
            table_detection_model_path: str = None,
            score_threshold: float = 0.5,
            use_deskew: bool = False,
            use_doc_cls: bool = True,
            use_uvdoc: bool = False,
    ):
        self.table_detector = table_detector
        self.ocr_system = ocr_system
        self.table_classifier = table_classifier
        self.table_cell_detector_wired = table_cell_detector_wired
        self.table_cell_detector_wireless = table_cell_detector_wireless
        self.doc_cls_model = doc_cls_model
        self.uvdoc_model = uvdoc_model
        self.uvdoc_model_name = uvdoc_model_name
        self.uvdoc_model_path = uvdoc_model_path
        self.model_local_dir = model_local_dir
        self.providers = providers
        self.session_options = session_options
        self.executor = executor
        self.table_detection_model_path = table_detection_model_path
        self.score_threshold = score_threshold
        self.use_deskew = use_deskew
        self.use_doc_cls = use_doc_cls
        self.use_uvdoc = use_uvdoc

    async def __aenter__(self):
        if self.doc_cls_model is None:
            self.doc_cls_model = DocumentOrientationDetector()
            await self.doc_cls_model.__aenter__()
        if self.use_uvdoc:
            if self.uvdoc_model is None:
                self.uvdoc_model = DocumentRectifier(
                    model_name=self.uvdoc_model_name,
                    model_path=self.uvdoc_model_path,
                    model_local_dir=self.model_local_dir,
                    providers=self.providers,
                    session_options=self.session_options,
                    executor=self.executor
                )
            await self.uvdoc_model.__aenter__()
        if self.table_detector is None:
            self.table_detector = TableDetector(
                model_path=self.table_detection_model_path,
                threshold=self.score_threshold
            )
            await self.table_detector.__aenter__()
        if self.ocr_system is None:
            self.ocr_system = PredictSystem(use_doc_cls=False, use_angle_cls=False)
            await self.ocr_system.__aenter__()
        if self.table_classifier is None:
            self.table_classifier = TableClassifier()
            await self.table_classifier.__aenter__()
        if self.table_cell_detector_wired is None:
            self.table_cell_detector_wired = TableCellDetector(
                model_name=TableModels.WIRED_TABLE
            )
            await self.table_cell_detector_wired.__aenter__()
        if self.table_cell_detector_wireless is None:
            self.table_cell_detector_wireless = TableCellDetector(
                model_name=TableModels.WIRELESS_TABLE
            )
            await self.table_cell_detector_wireless.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        models = [
            self.doc_cls_model,
            self.table_detector,
            self.ocr_system,
            self.table_classifier,
            self.table_cell_detector_wired,
            self.table_cell_detector_wireless
        ]
        for model in models:
            if model:
                await model.__aexit__(exc_type, exc_val, exc_tb)

    async def _convert_table_to_html(self, table_image: numpy.ndarray) -> str:
        """将表格图片转换为HTML"""
        cls_result = await self.table_classifier.predict(table_image)
        table_type = cls_result['class_name']
        detector = (
            self.table_cell_detector_wired
            if table_type == 'wired_table'
            else self.table_cell_detector_wireless
        )
        cell_result = await detector.predict(table_image)
        ocr_result = await self.ocr_system.predict(table_image)
        html_generator = TableHTMLGenerator(
            cell_detection_results=cell_result,
            ocr_result=ocr_result,
            iou_threshold=0.7
        )
        html_output = html_generator.generate_html()
        return html_output

    async def ocr_table(
            self,
            image: Union[str, numpy.ndarray, Image.Image]
    ):
        numpy_image: numpy.ndarray = await ImageLoader.load_image(image)
        if self.use_deskew:
            numpy_image = await ImageRotate.deskew_rotate_image(numpy_image)
        if self.use_doc_cls:
            numpy_image = await ImageRotate.doc_cls_rotate_image(numpy_image, self.doc_cls_model)
        if self.use_uvdoc:
            numpy_image = await self.uvdoc_model.predict(numpy_image)
        detection_result = await self.table_detector.predict(numpy_image)
        table_boxes = detection_result['boxes']
        results = []
        for box_info in table_boxes:
            x1, y1, x2, y2 = box_info['bbox']
            table_patch = numpy_image[y1:y2, x1:x2].copy()
            if table_patch.size == 0:
                continue
            html_output = await self._convert_table_to_html(table_patch)
            results.append(
                TableHtml(
                    bbox=(x1, y1, x2, y2),
                    score=box_info['score'],
                    html=html_output
                )
            )
        return results

    async def predict(
            self,
            image: Union[str, numpy.ndarray, Image.Image]
    ) -> List[TableHtml]:
        return await self.ocr_table(image)

    async def predict_batch(
            self,
            images: List[Union[str, numpy.ndarray, Image.Image]],
            max_concurrent: int = os.cpu_count()
    ) -> List[List[TableHtml]]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(image):
            async with semaphore:
                return await self.predict(image)

        tasks = [process_with_semaphore(img) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

#
# async def main():
#     image = cv2.imread("1.jpg")
#     if image is None:
#         raise FileNotFoundError("无法读取图像")
#     async with ImageTableToHTML() as processor:
#         results = await processor.predict(image)
#         for i, result in enumerate(results):
#             print(f"\n=== 表格 {i + 1} ===")
#             print(f"位置: {result['bbox']}")
#             print(f"置信度: {result['score']:.3f}")
#             print(f"HTML:\n{result['html']}")
#
#
# if __name__ == '__main__':
#     import asyncio
#
#     asyncio.run(main())
