import asyncio
import cv2
import numpy
import onnxruntime
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from paddleONNXOCR.models_enum import TableModels
from paddleONNXOCR.predict.predict_base import PredictBase


class TableClassifier(PredictBase):
    """表格分类器（基于 ONNX 模型）（ONNX Runtime版，预处理/后处理对齐Paddle配置）"""

    # 默认参数，可以被构造函数覆盖
    DEFAULT_INPUT_SIZE = 224
    DEFAULT_RESIZE_SHORT = 256
    NUM_CLASSES = 2
    CLASS_NAMES = ["wired_table", "wireless_table"]

    MEAN = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
    STD = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)
    SCALE = 1.0 / 255.0  # 配置文件里的 scale

    def __init__(
            self,
            model_name: TableModels = TableModels.L_CNet_x1_0,
            model_path: str | None = None,
            model_local_dir: str = "models",
            providers: list = None,
            session_options: onnxruntime.SessionOptions = None,
            executor: ThreadPoolExecutor | None = None,
            input_size: int = 224,
            resize_short: int = 256
    ):
        """
        :param model_name: 模型名称
        :param model_path: 模型路径
        :param model_local_dir: 模型下载到本地路径
        :param providers: onnx providers，默认由于PaddleONNOCRXUtils.get_available_providers选择
        :param session_options: onnxruntime.SessionOptions对象
        :param executor: 线程池
        :param input_size: 输入图像尺寸（用于中心裁剪）
        :param resize_short: 短边缩放尺寸
        """
        super().__init__(model_name, model_path, model_local_dir, providers, session_options, executor)

        # 可配置参数
        self.input_size = input_size
        self.resize_short = resize_short

    def _preprocess_sync(self, image: numpy.ndarray) -> numpy.ndarray:
        """同步预处理图像"""
        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # step1: 按短边缩放图像
        image = asyncio.run(self._resize_image(image, resize_short=self.resize_short))
        # step2: 中心裁剪图像（带填充处理）
        image = asyncio.run(self._crop_image(image, size=self.input_size))
        # step3: 归一化 (和 config 完全一致)
        image = image.astype(numpy.float32) * self.SCALE  # 等价于 /255
        image = (image - self.MEAN) / self.STD
        # step4: HWC → CHW
        image = numpy.transpose(image, (2, 0, 1))
        # step5: batch 维度
        blob = numpy.expand_dims(image, axis=0).astype(numpy.float32)
        return blob

    async def _resize_image(self, image: numpy.ndarray, resize_short: int = 256) -> numpy.ndarray:
        """按短边缩放图像"""
        h, w = image.shape[:2]
        if h < w:
            new_h = resize_short
            new_w = int(w * resize_short / h)
        else:
            new_w = resize_short
            new_h = int(h * resize_short / w)

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized_image

    async def _crop_image(self, image: numpy.ndarray, size: int = 224) -> numpy.ndarray:
        """中心裁剪图像（带填充处理）"""
        h, w = image.shape[:2]
        start_y = max(0, (h - size) // 2)
        start_x = max(0, (w - size) // 2)
        end_y = min(h, start_y + size)
        end_x = min(w, start_x + size)
        cropped_image = image[start_y:end_y, start_x:end_x]
        # 如果裁剪后尺寸不足，进行填充
        if cropped_image.shape[0] != size or cropped_image.shape[1] != size:
            pad_h = size - cropped_image.shape[0]
            pad_w = size - cropped_image.shape[1]
            cropped_image = numpy.pad(
                cropped_image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=0
            )
        return cropped_image

    def _run_inference_sync(self, blob: numpy.ndarray) -> Dict[str, Any]:
        """
        运行模型推理（模型已包含 softmax）
        """
        probs = self.session.run(self.output_names, {self.input_names[0]: blob})[0][0]  # 直接就是概率分布
        pred_id = int(numpy.argmax(probs))
        pred_prob = float(probs[pred_id])
        return {
            "class_id": pred_id,
            "class_name": self.CLASS_NAMES[pred_id],
            "probability": pred_prob,
            "probabilities": probs.tolist()
        }

    async def _run_inference(self, blob: numpy.ndarray) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._run_inference_sync,
            blob
        )
