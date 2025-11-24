import asyncio
import cv2
import numpy
import onnxruntime
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from paddleONNXOCR import (
    ImageModels
)
from paddleONNXOCR.predict.predict_base import PredictBase


class TableDetector(PredictBase):
    """表格区域检测器 - 继承自PredictBase"""

    def __init__(
            self,
            model_name: ImageModels = ImageModels.TABLE_LAYOUT_PICODET,
            model_path: str | None = None,
            model_local_dir: str = "models",
            providers: list = None,
            session_options: onnxruntime.SessionOptions = None,
            executor: ThreadPoolExecutor | None = None,
            threshold: float = 0.5,
            target_size: Tuple[int, int] = (800, 608)
    ):
        """
        :param model_name: 模型名称枚举
        :param model_path: 模型路径
        :param model_local_dir: 模型下载到本地路径
        :param providers: onnx providers
        :param session_options: onnxruntime.SessionOptions对象
        :param executor: 线程池
        :param threshold: 检测置信度阈值
        :param target_size: 目标尺寸 (height, width)
        """
        self.threshold = threshold
        self.target_h, self.target_w = target_size
        super().__init__(model_name, model_path, model_local_dir, providers, session_options, executor)

    def _preprocess_sync(self, image: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, Tuple[int, int]]:
        """预处理图像 - 参考LayoutDetection的实现"""
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        # 转换为RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img_rgb.shape[:2]

        # 缩放到目标尺寸
        img_resized = cv2.resize(img_rgb, (self.target_w, self.target_h))

        # 归一化
        img_normalized = img_resized.astype(numpy.float32) / 255.0
        mean = numpy.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = numpy.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_normalized = (img_normalized - mean) / std

        # 转换为CHW格式并添加batch维度
        img_input = img_normalized.transpose(2, 0, 1)[numpy.newaxis, :].astype(numpy.float32)

        # 计算缩放因子
        scale_factor = numpy.array([[self.target_h / ori_h, self.target_w / ori_w]], dtype=numpy.float32)

        return img_input, scale_factor, (ori_w, ori_h)

    def _run_inference_sync(self, blob: Tuple) -> Dict[str, Any]:
        """同步推理"""
        img_input, scale_factor, (ori_w, ori_h) = blob

        # 构造输入字典
        input_dict = {
            'image': img_input,
            'scale_factor': scale_factor
        }

        # 执行推理
        outputs = self.session.run(self.output_names, input_dict)
        boxes_with_scores = outputs[0]
        num_boxes = int(outputs[1][0])

        # 后处理 - 解析检测框
        detected_boxes = []
        for i in range(num_boxes):
            if i >= len(boxes_with_scores):
                break
            box_info = boxes_with_scores[i]
            score = float(box_info[1])
            if score <= self.threshold:
                continue
            x1, y1, x2, y2 = box_info[2:6]
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(ori_w, int(round(x2)))
            y2 = min(ori_h, int(round(y2)))
            if x2 > x1 and y2 > y1:
                detected_boxes.append({
                    'bbox': (x1, y1, x2, y2),
                    'score': score
                })

        return {'boxes': detected_boxes}

    async def _run_inference(self, preprocess_result: Tuple) -> Dict[str, Any]:
        """异步推理包装"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._run_inference_sync,
            preprocess_result
        )
