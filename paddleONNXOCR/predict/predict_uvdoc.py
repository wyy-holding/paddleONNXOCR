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
        """
        图像预处理：保持 RGB 格式处理
        """
        # 1. BGR转RGB（与第一份代码一致）
        if image.ndim == 3:
            if image.shape[2] == 4:  # RGBA -> RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # BGR -> RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 2. 归一化到[0,1]
        img = image.astype(numpy.float32) / 255.0
        # 3. 通道转换 HWC -> CHW
        chw = numpy.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        # 4. 添加batch维度 CHW -> BCHW
        chw = numpy.expand_dims(chw, axis=0)  # NCHW
        return chw

    async def _run_inference(self, blob: numpy.ndarray) -> numpy.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._run_inference_sync, blob)

    async def predict_from_array(self, img_array: numpy.ndarray) -> numpy.ndarray:
        blob = await self.preprocess(img_array)
        output = await self._run_inference(blob)  # (N,C,H,W) 图像张量

        # 后处理 - 与第一份代码保持一致
        if output.ndim == 4:  # NCHW
            # 1. 移除batch维度 BCHW -> CHW
            output = output[0]
            # 2. 通道转换 CHW -> HWC
            output = numpy.transpose(output, (1, 2, 0))
            # 3. 反归一化到[0,255]
            output = numpy.clip(output * 255.0, 0, 255).astype(numpy.uint8)
            # 4. RGB转BGR（与第一份代码一致）
            result_image = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            return result_image
        else:
            raise ValueError("Unexpected output shape from model")
