import asyncio
import math
import os
import cv2
import numpy
import onnxruntime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple, Dict, Any
from paddleONNXOCR.models_enum import RecModels
from paddleONNXOCR.predict.predict_base import PredictBase


class OCRRecognizer(PredictBase):
    """
    基于 ONNXRuntime 的文本识别器，尽量还原 PaddleOCR(Official) 的 rec 推理流程：
      - 预处理：按 batch 计算 max_wh_ratio，resize + padding 到 (C,H,W)，归一化至 [-1,1]
      - 解码：对齐 PaddleOCR 的 CTCLabelDecode（blank=0, idx-1 映射字典）
      - 批处理：宽高比排序 + 分批推理 + 还原原始顺序
    说明：本实现针对 CTC 模型（如 PP-OCRv5_mobile_rec）；其它算法（SRN/SAR/NRTR 等）未实现。
    """

    def __init__(
            self,
            model_name: RecModels = RecModels.MOBILE,
            model_path: str | None = None,
            model_local_dir: str = "models",
            providers: Optional[List[str]] = None,
            session_options: onnxruntime.SessionOptions | None = None,
            executor: ThreadPoolExecutor | None = None,
            charset_path: str = str(Path(__file__).parent.parent / "static/ppocrv5_dict.txt"),
            rec_image_shape: str = "3,48,320",  # 与官网一致，形如 "3,48,320"
            rec_batch_num: int = 6,  # 与官网一致
            use_space_char: bool = True,  # 与官网一致
    ):
        """
        OCR识别器
        :param model_path: ONNX模型路径
        :param providers: ONNX Runtime providers，默认自动选择
        :param session_options: ONNX会话选项
        :param executor: 进程池
        :param charset_path: 字典路径
        :param rec_image_shape: 图像形状
        :param rec_batch_num: 批大小
        :param use_space_char: 是否使用空格字符
        """
        self.charset_path = charset_path
        self.rec_image_shape = rec_image_shape
        self.rec_batch_num = rec_batch_num
        self.use_space_char = use_space_char
        self.imgH = None
        self.imgC = None
        self.imgW = None
        self.blank_idx = None
        self.num_classes = None
        self.charset = None
        super().__init__(model_name, model_path, model_local_dir, providers, session_options, executor, self.init)

    async def init(self):
        # 解析 rec_image_shape
        self.rec_image_shape = [int(v) for v in self.rec_image_shape.split(",")]
        assert len(self.rec_image_shape) == 3, "rec_image_shape 必须是 'C,H,W'"
        self.imgC, self.imgH, self.imgW = self.rec_image_shape

        # 若 onnx 给出了固定宽度（int 且 >0），以 onnx 输入为准；否则采用 rec_image_shape 中的 W
        if isinstance(self.input_shape[3], int) and self.input_shape[3] > 0:
            self.imgW = int(self.input_shape[3])
        # 注意：若 onnx width 是动态的（如 None 或 'None'），则沿用 rec_image_shape 的 W

        # 初始化字典：与官方 CTCLabelDecode 对齐（blank=0，字符索引从 1 开始）
        self.charset = self._load_charset(self.charset_path, self.model_path)
        if self.use_space_char and " " not in self.charset:
            # 与 PaddleOCR 一致，保证空格存在
            self.charset.append(" ")
        # 输出类别应为 len(self.charset)+1（+1 表示 blank）
        self.num_classes = len(self.charset) + 1
        self.blank_idx = 0  # PaddleOCR CTCLabelDecode 的 blank 索引为 0

    def _load_charset(self, charset_path: str, model_path: str) -> List[str]:
        """
        兼容地加载字典：
          - 优先使用传入的 charset_path
          - 若模型目录下存在 inference.yml 并包含 PostProcess.character_dict（字符列表），也可使用
        """
        # 如果模型同目录下有 inference.yml，可以尝试读取（与官网逻辑类似）
        model_dir = os.path.dirname(os.path.abspath(model_path))
        # 默认按文件加载
        if not os.path.exists(charset_path):
            raise FileNotFoundError(f"字典文件不存在: {charset_path}")
        with open(charset_path, "r", encoding="utf-8") as f:
            charset = [line.strip("\n") for line in f.readlines() if line.strip("\n") != ""]
        return charset

    def _resize_norm_img(self, img: numpy.ndarray, max_wh_ratio: float) -> numpy.ndarray:
        """
        与官方 resize_norm_img 对齐：
          - 目标宽 imgW = int(imgH * max_wh_ratio)，若 onnx 输入宽为确定值且 >0 则使用该固定宽
          - 宽度按比例缩放为 resized_w = min(ceil(imgH * ratio), imgW)
          - 归一化到 [-1,1]，NCHW
        """
        assert img.ndim == 3, "输入必须是 HWC"
        if img.shape[2] == 1 and self.imgC == 3:  # 灰度转3通道
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3 and self.imgC == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[:, :, None]

        # 计算目标宽
        imgW = int(self.imgH * max_wh_ratio)
        if isinstance(self.input_shape[3], int) and self.input_shape[3] > 0:
            # onnx 模型固定宽
            imgW = int(self.input_shape[3])

        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = imgW if math.ceil(self.imgH * ratio) > imgW else int(math.ceil(self.imgH * ratio))

        if resized_w <= 0:
            resized_w = 1

        resized_image = cv2.resize(img, (resized_w, self.imgH))
        resized_image = resized_image.astype("float32")
        # to NCHW and [-1,1]
        resized_image = resized_image.transpose(2, 0, 1) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = numpy.zeros((self.imgC, self.imgH, imgW), dtype=numpy.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def _preprocess_sync(self, image: numpy.ndarray) -> numpy.ndarray:
        """同步预处理函数（单图版本）"""
        h, w = image.shape[:2]
        wh_ratio = w / float(h) if h > 0 else 1.0
        default_ratio = self.imgW / float(self.imgH)
        max_wh_ratio = max(default_ratio, wh_ratio)
        blob = self._resize_norm_img(image, max_wh_ratio)
        return numpy.expand_dims(blob, axis=0)  # 添加batch维度

    # =============== 解码（对齐 PaddleOCR CTCLabelDecode） ===============
    def _ctc_greedy_decode_batch(self, preds: numpy.ndarray) -> List[Tuple[str, float]]:
        """
        与 PaddleOCR 的 CTCLabelDecode 对齐：
          - blank 索引为 0
          - 字符索引从 1 开始映射到 self.charset 的 0..len-1
          - 去重 + 去 blank
          - 置信度为字符位置概率的平均值（与官网类似）
        Args:
            preds: [N, T, C]
        Returns:
            List[[text, score], ...]
        """
        assert preds.ndim == 3, f"期望 preds 形状 [N,T,C]，实际 {preds.shape}"
        batch_size, T, C = preds.shape
        rec_results: List[Tuple[str, float]] = []

        # 按时间步取最大类和对应概率
        preds_idx = preds.argmax(axis=-1)  # [N, T]
        preds_prob = preds.max(axis=-1)  # [N, T]

        for b in range(batch_size):
            last_idx = -1
            chars = []
            probs = []
            for t in range(T):
                idx = int(preds_idx[b, t])
                if idx != self.blank_idx and idx != last_idx:
                    # idx-1 对齐字典（官方 CTCLabelDecode 逻辑）
                    if 1 <= idx <= len(self.charset):
                        chars.append(self.charset[idx - 1])
                        probs.append(float(preds_prob[b, t]))
                last_idx = idx

            text = "".join(chars)
            score = float(numpy.mean(probs)) if len(probs) > 0 else 0.0
            rec_results.append((text, score))
        return rec_results

    def _run_inference_sync(self, blob: numpy.ndarray) -> numpy.ndarray:
        """同步推理函数，返回主输出 [N, T, C]"""
        outputs = self.session.run(None, {self.input_name: blob})
        for out in outputs:
            if out.ndim == 3:
                return out
        return outputs[0]

    async def _run_inference(self, blob: numpy.ndarray) -> Dict[str, Any]:
        """异步推理 + 后处理"""
        try:
            loop = asyncio.get_event_loop()
            preds = await loop.run_in_executor(self.executor, self._run_inference_sync, blob)
            rec_res = self._ctc_greedy_decode_batch(preds)[0]  # 单图
            text, score = rec_res
            return {"text": text, "score": score}
        except Exception as e:
            raise RuntimeError(f"ONNX 推理失败: {e}")
