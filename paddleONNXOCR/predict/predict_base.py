import psutil
from typing import Optional, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from paddleONNXOCR.utils import *


class PredictBase:
    """
    推理预测基类
    默认创建系统线程池，max_workers=cpu数量
    """

    def __init__(
            self,
            model_name: TextLineModels | DetModels | RecModels | TableModels | ImageModels,
            model_path: str | None = None,
            model_local_dir: str = "models",
            providers: list = None,
            session_options: Optional[onnxruntime.SessionOptions] = None,
            executor: ThreadPoolExecutor | None = None,
            callable_func: Callable | None = None
    ):
        """
        :param model_path: 模型路径
        :param providers: onnx providers，默认由于PaddleONNOCRXUtils.get_available_providers选择
        :param session_options:
        :param executor:
        :param callable_func:
        """
        self.model_name = model_name.value
        self.model_path = model_path
        self.model_local_dir = model_local_dir
        self.providers = providers
        self.session_options = session_options
        self.callable_func = callable_func
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=psutil.cpu_count()) if executor is None else executor
        self.session = None
        self.input_name = None
        self.output_name = None
        self.output_names = None
        self.input_shape = None
        self.aiohttp_session = None

    async def predict_from_array(
            self,
            img_array: numpy.ndarray
    ) -> Dict[str, Any]:
        """
        numpy.ndarray 格式数据推理
        :param img_array: numpy.ndarray对象
        :return: 推理结果字典
        """
        blob = await self.preprocess(img_array)
        return await self._run_inference(blob)

    def _preprocess_sync(
            self,
            *args,
            **kwargs
    ) -> Any:
        ...

    async def preprocess(
            self,
            image: numpy.ndarray
    ) -> numpy.ndarray:
        """
        预处理
        :param image: numpy.ndarray 图片对象
        :return: 预处理后图片
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._preprocess_sync, image)

    def _run_inference_sync(
            self,
            blob: numpy.ndarray
    ) -> numpy.ndarray:
        """
        同步推理
        :param blob: numpy.ndarray 图片对象
        :return: onnx推理后结果
        """
        return self.session.run([self.output_name], {self.input_name: blob})[0]

    async def _run_inference(self, *args, **kwargs) -> Any:
        ...

    async def predict(
            self,
            image: numpy.ndarray
    ) -> Dict[str, Any] | numpy.ndarray:
        """
        单次推理
        :param image: 待推理输入图片
        :return: 返回推理结果字典或图像
        """
        if isinstance(image, numpy.ndarray):
            return await self.predict_from_array(image)
        else:
            raise TypeError("输入类型不支持。支持: str (路径/URL), np.ndarray, PIL.Image.Image")

    async def predict_batch(
            self,
            img_list: List[numpy.ndarray],
            max_concurrent: int = psutil.cpu_count()
    ) -> tuple[Any]:
        """
        批量推理
        :param img_list: 多个带推理输入列表
        :param max_concurrent: 最大任务数，默认和线程池一样，采用cpu数量
        :return: 多个推理结果元组
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def predict_with_semaphore(image: numpy.ndarray):
            async with semaphore:
                try:
                    return await self.predict(image)
                except Exception as e:
                    return {"error": str(e)}

        tasks = [predict_with_semaphore(image) for image in img_list]
        return await asyncio.gather(*tasks)

    async def download_model(self):
        if self.model_path:
            return
        model_path = Path(f"{self.model_local_dir}/{self.model_name}")
        self.model_path = model_path
        if model_path.exists():
            return
        await UtilsCommon.download_model(self.model_name, self.model_local_dir)

    async def close(self):
        """
        处理结束后关闭aiohttp session和线程池
        :return:
        """
        if self.aiohttp_session:
            await self.aiohttp_session.close()
        self.executor.shutdown(wait=True)

    async def __aenter__(self):
        """
        异步初始化
        :return: 当前类
        """
        await self.download_model()
        if self.session_options is None:
            self.session_options = await PaddleONNOCRXUtils.get_onnx_session_options()
        if self.providers is None:
            self.providers = await PaddleONNOCRXUtils.get_available_providers()
        try:
            self.session = await PaddleONNOCRXUtils.get_onnx_session(self.model_path, self.providers,
                                                                     self.session_options)
        except Exception as e:
            raise e
        input_name, output_name, input_shape = await PaddleONNOCRXUtils.get_onnx_session_params(self.session)
        self.input_name = input_name
        self.output_name = output_name
        self.input_shape = input_shape
        self.aiohttp_session = await PaddleONNOCRXUtils.get_aiohttp_session()
        if self.callable_func is not None:
            await self.callable_func()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
