import asyncio
import cv2
import numpy
import onnxruntime
from concurrent.futures import ThreadPoolExecutor
from paddleONNXOCR.models_enum import ImageModels
from paddleONNXOCR.predict.predict_base import PredictBase


class DocumentRectifier(PredictBase):
    """图像矫正器，支持动态输入尺寸的ONNX模型"""

    def __init__(
            self,
            model_name: ImageModels = ImageModels.UVDOC,
            model_path: str | None = None,
            model_local_dir: str = "models",
            providers: list = None,
            session_options: onnxruntime.SessionOptions = None,
            executor: ThreadPoolExecutor | None = None
    ):
        super().__init__(model_name, model_path, model_local_dir, providers, session_options, executor)

    def _preprocess_sync(self, image: numpy.ndarray) -> numpy.ndarray:
        """
        图像预处理：保持 BGR 格式处理
        """
        if image.ndim == 3:
            if image.shape[2] == 4:  # RGBA -> BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            # 不需要 BGR -> RGB 转换

        img = image.astype(numpy.float32) / 255.0
        chw = numpy.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        chw = numpy.expand_dims(chw, axis=0)  # NCHW
        return chw

    async def _run_inference(self, blob: numpy.ndarray) -> numpy.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._run_inference_sync, blob)

    async def predict_from_array(self, img_array: numpy.ndarray) -> numpy.ndarray:
        blob = await self.preprocess(img_array)
        output = await self._run_inference(blob)  # (N,C,H,W) 图像张量
        if output.ndim == 4:  # NCHW
            out = output[0].transpose(1, 2, 0)
        else:
            raise ValueError("Unexpected output shape from model")
        return numpy.clip(out * 255, 0, 255).astype(numpy.uint8)
