import asyncio
import os
import sys
import cv2
import json
import numpy
import onnxruntime
import psutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple, Union, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from paddleONNXOCR.image_loader import ImageLoader
from paddleONNXOCR.image_rotate import ImageRotate
from paddleONNXOCR.models_enum import *
from paddleONNXOCR.predict.predict_cls import TextLineOrientationDetector
from paddleONNXOCR.predict.predict_det import TextDetector
from paddleONNXOCR.predict.predict_rec import OCRRecognizer
from paddleONNXOCR.predict.predict_doc_cls import DocumentOrientationDetector
from paddleONNXOCR.predict.predict_uvdoc import DocumentRectifier
from paddleONNXOCR.utils import TextBoxSorter
from paddleONNXOCR.predict.ocr_dataclass import OCRChunkResult, OCRResult


class PredictSystem:
    def __init__(
            self,
            det_model_name: DetModels = DetModels.MOBILE,
            cls_model_name: TextLineModels = TextLineModels.L_CNET_X0_25,
            rec_model_name: RecModels = RecModels.MOBILE,
            doc_cls_model_name: ImageModels = ImageModels.L_CNet_x1_0,
            uvdoc_model_name: ImageModels = ImageModels.UVDOC,
            det_model_path: str | None = None,
            cls_model_path: str | None = None,
            rec_model_path: str | None = None,
            doc_cls_model_path: str | None = None,
            uvdoc_model_path: str | None = None,
            model_local_dir: str = "models",
            det_model: TextDetector = None,
            cls_model: TextLineOrientationDetector = None,
            rec_model: OCRRecognizer = None,
            doc_cls_model: DocumentOrientationDetector = None,
            uvdoc_model: DocumentRectifier = None,
            charset_path: str = str(Path(__file__).parent.parent / "static/ppocrv5_dict.txt"),
            use_angle_cls: bool = True,
            use_deskew: bool = True,
            use_uvdoc: bool = False,
            use_doc_cls: bool = True,
            cls_thresh: float = 0.5,
            drop_score: float = 0.3,
            sort_boxes: bool = True,  # 是否对检测框排序
            providers: Optional[List[str]] = None,
            session_options: Optional[onnxruntime.SessionOptions] = None,
            executor: Optional[ThreadPoolExecutor] = None,
            # 检测参数
            det_db_thresh: float = 0.3,
            det_db_box_thresh: float = 0.6,
            det_db_unclip_ratio: float = 1.5,
            det_max_candidates: int = 1000,
            # 识别参数
            rec_batch_num: int = 6,
            rec_image_shape: str = "3,48,320",
    ):
        self.det_model_name = det_model_name
        self.cls_model_name = cls_model_name
        self.rec_model_name = rec_model_name
        self.doc_cls_model_name = doc_cls_model_name
        self.uvdoc_model_name = uvdoc_model_name
        self.det_model_path = det_model_path
        self.cls_model_path = cls_model_path
        self.rec_model_path = rec_model_path
        self.doc_cls_model_path = doc_cls_model_path
        self.uvdoc_model_path = uvdoc_model_path
        self.model_local_dir = model_local_dir
        self.use_angle_cls = use_angle_cls
        self.use_deskew = use_deskew
        self.use_uvdoc = use_uvdoc
        self.use_doc_cls = use_doc_cls
        self.cls_thresh = cls_thresh
        self.drop_score = drop_score
        self.sort_boxes = sort_boxes
        self.det_model = det_model
        self.cls_model = cls_model
        self.rec_model = rec_model
        self.doc_cls_model = doc_cls_model
        self.uvdoc_model = uvdoc_model
        self.charset_path = charset_path
        self.providers = providers
        self.session_options = session_options
        self.executor = executor
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.det_max_candidates = det_max_candidates
        self.rec_batch_num = rec_batch_num
        self.rec_image_shape = rec_image_shape
        self._font_path = None

    async def _init_det_model(self):
        if self.det_model is None:
            self.det_model = TextDetector(
                model_name=self.det_model_name,
                model_path=self.det_model_path,
                model_local_dir=self.model_local_dir,
                providers=self.providers,
                session_options=self.session_options,
                executor=self.executor,
                thresh=self.det_db_thresh,
                box_thresh=self.det_db_box_thresh,
                unclip_ratio=self.det_db_unclip_ratio,
                max_candidates=self.det_max_candidates,
            )
        await self.det_model.__aenter__()

    async def _init_cls_model(self):
        if self.use_angle_cls:
            if self.cls_model is None:
                self.cls_model = TextLineOrientationDetector(
                    model_name=self.cls_model_name,
                    model_path=self.cls_model_path,
                    model_local_dir=self.model_local_dir,
                    providers=self.providers,
                    session_options=self.session_options,
                    executor=self.executor,
                )
            await self.cls_model.__aenter__()
        else:
            self.cls_model = None

    async def _init_rec_model(self):
        if self.rec_model is None:
            self.rec_model = OCRRecognizer(
                model_name=self.rec_model_name,
                model_path=self.rec_model_path,
                model_local_dir=self.model_local_dir,
                providers=self.providers,
                session_options=self.session_options,
                executor=self.executor,
                charset_path=self.charset_path,
                rec_batch_num=self.rec_batch_num,
                rec_image_shape=self.rec_image_shape,
            )
        await self.rec_model.__aenter__()

    async def _init_doc_cls_model(self):
        if self.use_doc_cls:
            if self.doc_cls_model is None:
                self.doc_cls_model = DocumentOrientationDetector(
                    model_name=self.doc_cls_model_name,
                    model_path=self.doc_cls_model_path,
                    model_local_dir=self.model_local_dir,
                    providers=self.providers,
                    session_options=self.session_options,
                    executor=self.executor
                )
            await self.doc_cls_model.__aenter__()

    async def _init_uvdoc_model(self):
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

    async def _init_models(self):
        await self._init_det_model()
        await self._init_cls_model()
        await self._init_rec_model()
        await self._init_doc_cls_model()
        await self._init_uvdoc_model()

    async def _release_models(self, exc_type, exc_val, exc_tb):
        await self.det_model.__aexit__(exc_type, exc_val, exc_tb)
        if self.cls_model:
            await self.cls_model.__aexit__(exc_type, exc_val, exc_tb)
        await self.rec_model.__aexit__(exc_type, exc_val, exc_tb)
        if self.doc_cls_model:
            await self.doc_cls_model.__aexit__(exc_type, exc_val, exc_tb)
        if self.uvdoc_model:
            await self.uvdoc_model.__aexit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._init_models()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self._release_models(exc_type, exc_val, exc_tb)

    async def _get_rotate_crop_image(
            self,
            img: numpy.ndarray,
            points: numpy.ndarray
    ) -> numpy.ndarray:
        """
        根据检测框裁剪并矫正图像
        :param img: 图像
        :param points: 交点坐标
        :return: 裁剪后图像
        """
        points = numpy.array(points, dtype=numpy.float32)
        img_crop_width = int(
            max(
                numpy.linalg.norm(points[0] - points[1]),
                numpy.linalg.norm(points[2] - points[3])
            )
        )
        img_crop_height = int(
            max(
                numpy.linalg.norm(points[0] - points[3]),
                numpy.linalg.norm(points[1] - points[2])
            )
        )

        pts_std = numpy.array([
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height]
        ], dtype=numpy.float32)

        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC
        )
        dst_h, dst_w = dst_img.shape[:2]
        if dst_h * 1.0 / dst_w >= 1.5:
            dst_img = numpy.rot90(dst_img, 1)

        return dst_img

    async def _classify_text(
            self,
            img_list: List[numpy.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        批量文本方向分类
        :param img_list: 图像列表
        :return: 分类结果列表
        """
        if not self.use_angle_cls or self.cls_model is None:
            return [("0_degree", 1.0) for _ in img_list]

        # 批量预测
        cls_results = await self.cls_model.predict_batch(img_list)

        results = []
        for result in cls_results:
            if "error" in result:
                results.append(("0_degree", 1.0))
            else:
                results.append((result["class_name"], result["confidence"]))
        return results

    async def _recognize_text(
            self,
            img_list: List[numpy.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        批量文本识别
        :param img_list: 图像列表
        :return:  识别结果列表
        """
        rec_results = await self.rec_model.predict_batch(img_list)
        results = []
        for result in rec_results:
            if "error" in result:
                results.append(("", 0.0))
            else:
                results.append((result["text"], result["score"]))
        return results

    async def predict(
            self,
            image: Union[str, numpy.ndarray, Image.Image],
    ) -> OCRResult:
        """
        对单张图像进行OCR
        :param image: 输入图像（路径、numpy数组或PIL图像）
        :return: OCR结果列表和可选的裁剪图像列表
        """
        numpy_image: numpy.ndarray = await ImageLoader.load_image(image)
        if self.use_deskew:
            numpy_image = await ImageRotate.deskew_rotate_image(numpy_image)
        if self.use_uvdoc:
            numpy_image = await self.uvdoc_model.predict(numpy_image)
        if self.use_doc_cls:
            numpy_image = await ImageRotate.doc_cls_rotate_image(numpy_image, self.doc_cls_model)
        det_result = await self.det_model.predict(numpy_image)
        if det_result["num_boxes"] == 0:
            return OCRResult(image=numpy_image)
        boxes = det_result["boxes"]
        scores = det_result["scores"]
        if self.sort_boxes:
            boxes, scores = TextBoxSorter.sort_boxes_in_reading_order(
                boxes, scores, "auto"
            )
        # 裁剪文本区域
        img_crop_list = [
            await self._get_rotate_crop_image(numpy_image, numpy.array(box))
            for box in boxes
        ]
        # 方向分类
        cls_results = await self._classify_text(img_crop_list)
        # 根据分类结果旋转图像
        for idx, (angle, confidence) in enumerate(cls_results):
            if angle == "180_degree" and confidence > self.cls_thresh:
                img_crop_list[idx] = cv2.rotate(img_crop_list[idx], cv2.ROTATE_180)
        # 文本识别
        rec_results = await self._recognize_text(img_crop_list)
        ocr_results = []
        for idx, ((text, score), box, (angle, angle_conf)) in enumerate(
                zip(rec_results, boxes, cls_results)
        ):
            if score >= self.drop_score:
                result = OCRChunkResult(
                    text=text,
                    confidence=score,
                    box=box,
                    angle=angle,
                    angle_confidence=angle_conf
                )
                ocr_results.append(result)
        return OCRResult(
            await self.get_text(ocr_results),
            await self.get_json(ocr_results),
            ocr_results,
            numpy_image.copy()
        )

    async def predict_batch(
            self,
            images: List[Union[str, numpy.ndarray, Image.Image]],
            max_concurrent: int = psutil.cpu_count()
    ) -> AsyncGenerator[OCRResult, None]:
        """
        批量处理多张图像
        :param images: 图像列表
        :param max_concurrent:  最大并发数
        :return: 每张图像的OCR结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(image):
            async with semaphore:
                try:
                    ocr_result = await self.predict(image)
                    return ocr_result
                except Exception as e:
                    raise Exception(e)

        tasks = [process_with_semaphore(img) for img in images]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    async def visualize_results(
            self,
            ocr_result: OCRResult
    ) -> numpy.ndarray:
        """
        画出对比图
        @:param ocr_result:ocr识别结果
        :return: 对比图，左边原图，右边识别结果
        """
        if ocr_result.image is None or ocr_result.results is None:
            raise ValueError("没有可用的OCR结果。请先调用 predict 方法。")
        font_path = self._font_path
        height, width = ocr_result.image.shape[:2]
        white_img = numpy.ones((height, width, 3), dtype=numpy.uint8) * 255
        white_img_pil = Image.fromarray(cv2.cvtColor(white_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(white_img_pil)
        for idx, result in enumerate(ocr_result.results):
            box = numpy.array(result.box, dtype=numpy.int32)
            # 计算框的尺寸
            box_width = numpy.linalg.norm(box[1] - box[0])
            box_height = numpy.linalg.norm(box[3] - box[0])
            # 判断是否为竖排文本（高度明显大于宽度）
            is_vertical = box_height >= box_width * 1.1
            # 根据文本方向调整字体大小
            if is_vertical:
                font_size = max(10, int(box_width * 0.85))
            else:
                font_size = max(10, int(box_height * 0.85))

            # 加载字体
            try:
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, size=font_size)
                else:
                    if sys.platform == "win32":
                        font = ImageFont.truetype("msyh.ttc", size=font_size)
                    elif sys.platform == "darwin":
                        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", size=font_size)
                    else:
                        font = ImageFont.truetype(
                            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size=font_size
                        )
            except:
                font = ImageFont.load_default()
            text = result.text
            if is_vertical:
                text_x, text_y = box[0][0], box[0][1]
                # 计算平均字符高度
                total_height = box_height
                char_count = len(text)
                if char_count > 0:
                    char_spacing = total_height / char_count
                else:
                    char_spacing = font_size * 1.2
                char_y = text_y
                for i, char in enumerate(text):
                    # 获取单个字符的尺寸
                    bbox = draw.textbbox((0, 0), char, font=font)
                    char_width = bbox[2] - bbox[0]
                    # 确保不超出画布边界
                    if char_y + font_size > height:
                        break
                    # 居中对齐字符
                    char_x = text_x + (box_width - char_width) // 2
                    char_x = max(0, min(char_x, width - char_width))
                    # 绘制字符
                    draw.text((char_x, char_y), char, font=font, fill=(255, 0, 0))
                    # 使用计算的平均间距
                    char_y = text_y + int((i + 1) * char_spacing)

            else:
                # 横排文本处理（原有逻辑）
                text_x, text_y = box[0][0], box[0][1]
                # 获取文本尺寸
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                # 边界检查
                text_x = max(0, min(text_x, width - text_width))
                text_y = max(0, min(text_y, height - text_height))
                # 绘制文字
                draw.text((text_x, text_y), text, font=font, fill=(255, 0, 0))

        white_img = cv2.cvtColor(numpy.array(white_img_pil), cv2.COLOR_RGB2BGR)
        vis_img = numpy.hstack([ocr_result.image, white_img])
        return vis_img

    async def get_text(
            self,
            ocr_result: List[OCRChunkResult],
            separator: str = " "
    ) -> str:
        """
        将OCR结果拼接成完整文本
        @:param ocr_result:ocr识别结果列表
        :param separator: 文本分隔符
        :return:  拼接后的完整文本
        """
        return separator.join([result.text for result in ocr_result])

    async def get_json(
            self,
            ocr_result: List[OCRChunkResult],
            ensure_ascii=False,
            indent=4
    ) -> str:
        """
        获取完整json数据
        :return: 完整json数据
        """
        data = []
        for result in ocr_result:
            data.append(await result.to_dict())
        return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
