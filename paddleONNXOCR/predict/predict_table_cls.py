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

    INPUT_SIZE = 224
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

    def _preprocess_sync(self, image: numpy.ndarray) -> numpy.ndarray:
        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # step1: Resize 短边=256，保持宽高比
        h, w = image.shape[:2]
        if h < w:
            new_h, new_w = self.INPUT_SIZE + 32, int(w * (self.INPUT_SIZE + 32) / h)
        else:
            new_h, new_w = int(h * (self.INPUT_SIZE + 32) / w), self.INPUT_SIZE + 32
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # step2: 中心裁剪 224 x 224
        h, w = image.shape[:2]
        start_h = (h - self.INPUT_SIZE) // 2
        start_w = (w - self.INPUT_SIZE) // 2
        image = image[start_h:start_h + self.INPUT_SIZE, start_w:start_w + self.INPUT_SIZE, :]

        # step3: 归一化 (和 config 完全一致)
        image = image.astype(numpy.float32) * self.SCALE  # 等价于 /255
        image = (image - self.MEAN) / self.STD

        # step4: HWC → CHW
        image = numpy.transpose(image, (2, 0, 1))

        # step5: batch 维度
        blob = numpy.expand_dims(image, axis=0).astype(numpy.float32)
        return blob

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
