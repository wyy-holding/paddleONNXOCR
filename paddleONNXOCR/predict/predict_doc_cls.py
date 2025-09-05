import asyncio
import cv2
import numpy
import onnxruntime
from typing import Optional, Dict, Any
from paddleONNXOCR.models_enum import ImageModels
from paddleONNXOCR.predict.predict_base import PredictBase
from concurrent.futures import ThreadPoolExecutor


class DocumentOrientationDetector(PredictBase):
    """图像方向检测器（ONNX Runtime版，预处理/后处理对齐Paddle配置）"""

    # 类常量：和配置文件里的label_list保持一致
    CLASS_NAMES = ('0', '90', '180', '270')
    INPUT_SIZE = 224  # 固定为224x224裁剪

    def __init__(
            self,
            model_name: ImageModels = ImageModels.L_CNet_x1_0,
            model_path: str | None = None,
            model_local_dir: str = "models",
            providers: Optional[list] = None,
            session_options: Optional[onnxruntime.SessionOptions] = None,
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
        """同步预处理函数（符合Paddle配置）"""

        # 1. 确保是RGB格式
        if image.ndim == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. resize short = 256，保持比例
        h, w = image.shape[:2]
        short_side = min(h, w)
        scale = 256.0 / short_side
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 3. 中心裁剪224x224
        h, w = image.shape[:2]
        start_h = (h - self.INPUT_SIZE) // 2
        start_w = (w - self.INPUT_SIZE) // 2
        image = image[start_h:start_h + self.INPUT_SIZE, start_w:start_w + self.INPUT_SIZE]

        # 4. 转换到 [0,1]
        image = image.astype(numpy.float32) / 255.0

        # 5. Normalize (ImageNet mean/std)
        mean = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
        std = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)
        image = (image - mean) / std

        # 6. ToCHWImage -> (3,224,224)
        image = numpy.transpose(image, (2, 0, 1))

        # 7. 增加batch维度
        image = numpy.expand_dims(image, axis=0).astype(numpy.float32)

        return image

    def run_inference(self, blob: numpy.ndarray) -> numpy.ndarray:
        return self._run_inference_sync(blob)[0]

    async def _run_inference(self, blob: numpy.ndarray) -> Dict[str, Any]:
        """异步推理（直接用模型的softmax输出）"""
        try:
            loop = asyncio.get_event_loop()
            probs = await loop.run_in_executor(self.executor, self.run_inference, blob)

            pred_id = int(numpy.argmax(probs))
            probability = float(probs[pred_id])

            return {
                "class_name": self.CLASS_NAMES[pred_id],
                "confidence": probability,  # = 概率
                "probability": probability,  # 保留字段一致性
                "class_id": pred_id,
                "logits": probs.tolist(),  # 实际是softmax输出了
                "probabilities": probs.tolist()
            }
        except Exception as e:
            raise RuntimeError(f"ONNX 推理失败: {e}")
