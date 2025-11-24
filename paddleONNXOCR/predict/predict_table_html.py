import os

import cv2
import numpy
from typing import List

from paddleONNXOCR.models_enum import TableModels
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
            # 如果需要自定义表格检测模型
            table_detection_model_path: str = None,
            score_threshold: float = 0.5
    ):
        self.table_detector = table_detector
        self.ocr_system = ocr_system
        self.table_classifier = table_classifier
        self.table_cell_detector_wired = table_cell_detector_wired
        self.table_cell_detector_wireless = table_cell_detector_wireless
        self.doc_cls_model = doc_cls_model
        self.table_detection_model_path = table_detection_model_path
        self.score_threshold = score_threshold

    async def __aenter__(self):
        if self.doc_cls_model is None:
            self.doc_cls_model = DocumentOrientationDetector()
            await self.doc_cls_model.__aenter__()
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

    async def predict(
            self,
            image: numpy.ndarray
    ) -> List[dict]:
        """
        处理图像，检测表格并转换为HTML

        参数:
            image: 输入图像

        返回:
            List[dict]: 每个元素包含:
                - 'bbox': (x1, y1, x2, y2) 表格位置
                - 'html': str 表格的HTML代码
        """
        image = await ImageRotate.doc_cls_rotate_image(image, self.doc_cls_model)
        detection_result = await self.table_detector.predict(image)
        table_boxes = detection_result['boxes']
        results = []
        for box_info in table_boxes:
            x1, y1, x2, y2 = box_info['bbox']
            table_patch = image[y1:y2, x1:x2].copy()
            if table_patch.size == 0:
                continue
            html_output = await self._convert_table_to_html(table_patch)
            results.append({
                'bbox': (x1, y1, x2, y2),
                'score': box_info['score'],
                'html': html_output
            })
        return results

    async def predict_batch(
            self,
            images: List[numpy.ndarray],
            max_concurrent: int = os.cpu_count()
    ) -> List[List[dict]]:
        """
        批量处理多张图像

        参数:
            images: 图像列表
            max_concurrent: 最大并发数

        返回:
            List[List[dict]]: 每张图像的表格检测结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(image):
            async with semaphore:
                return await self.predict(image)

        tasks = [process_with_semaphore(img) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results


async def main():
    image = cv2.imread("1.jpg")
    if image is None:
        raise FileNotFoundError("无法读取图像")
    async with ImageTableToHTML() as processor:
        results = await processor.predict(image)
        for i, result in enumerate(results):
            print(f"\n=== 表格 {i + 1} ===")
            print(f"位置: {result['bbox']}")
            print(f"置信度: {result['score']:.3f}")
            print(f"HTML:\n{result['html']}")


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
