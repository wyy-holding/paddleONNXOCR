import cv2
import numpy
import asyncio
import onnxruntime
from typing import Dict, Any
from paddleONNXOCR.models_enum import TextLineModels
from paddleONNXOCR.predict.predict_base import PredictBase
from concurrent.futures import ThreadPoolExecutor


class TextLineOrientationDetector(PredictBase):
    """文本行方向检测器 (ONNX Runtime版, 按配置文件对齐预处理)"""
    CLASS_NAMES = ('0_degree', '180_degree')
    INPUT_SIZE = (80, 160)

    def __init__(
            self,
            model_name: TextLineModels = TextLineModels.L_CNET_X0_25,
            model_path: str | None = None,
            model_local_dir: str = "models",
            providers: list = None,
            session_options: onnxruntime.SessionOptions = None,
            executor: ThreadPoolExecutor | None = None
    ):
        """
        :param model_name: 模型名称
        :param model_path: 模型路径
        :param model_local_dir: 模型下载到本地路径
        :param providers: onnx providers，默认由于PaddleONNOCRXUtils.get_available_providers选择
        :param session_options: onnxruntime.SessionOptions对象
        :param executor: 线程池
        """
        super().__init__(model_name, model_path, model_local_dir, providers, session_options, executor)

    def _preprocess_sync(
            self,
            image: numpy.ndarray
    ) -> numpy.ndarray:
        """
        预处理 按配置文件 Resize(160,80) -> Normalize -> ToCHW
        :param image: 图像数据
        :return: 处理后图像
        """
        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        image = image.astype(numpy.float32) * 0.00392156862745098
        mean = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
        std = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)
        image = (image - mean) / std
        image = numpy.transpose(image, (2, 0, 1))
        image = numpy.expand_dims(image, axis=0).astype(numpy.float32)
        return image

    def run_inference(
            self,
            blob: numpy.ndarray
    ) -> numpy.ndarray:
        return self._run_inference_sync(blob)[0]

    async def _run_inference(
            self,
            blob: numpy.ndarray
    ) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            probs = await loop.run_in_executor(self.executor, self.run_inference, blob)
            pred_id = int(numpy.argmax(probs))
            confidence = float(numpy.max(probs))
            return {
                "class_name": self.CLASS_NAMES[pred_id],
                "confidence": confidence,  # 就是 PaddleOCR 的 0.999
                "probability": confidence,  # 等价字段
                "class_id": pred_id,
                "probabilities": probs.tolist()  # 直接输出概率列表
            }
        except Exception as e:
            raise RuntimeError(f"ONNX 行方向推理失败: {e}")
