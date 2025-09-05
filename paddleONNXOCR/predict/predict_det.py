import asyncio
import math
import cv2
import numpy
import onnxruntime
import pyclipper
from shapely.geometry import Polygon
from typing import Literal, Dict, Any
from paddleONNXOCR.models_enum import DetModels
from paddleONNXOCR.predict.predict_base import PredictBase
from concurrent.futures import ThreadPoolExecutor


class DBPostProcess:
    def __init__(
            self,
            thresh=0.3,
            box_thresh=0.7,
            max_candidates=1000,
            unclip_ratio=2.0,
            use_dilation=False,
            score_mode="fast",
            box_type="quad",
            min_size=3,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.use_dilation = use_dilation
        self.score_mode = score_mode
        self.box_type = box_type
        self.min_size = min_size
        self.dilation_kernel = None if not use_dilation else numpy.array([[1, 1], [1, 1]], dtype=numpy.uint8)
        assert score_mode in ["fast", "slow"]
        assert box_type in ["quad", "poly"]

    def __call__(self, pred_map, ori_shape, resize_shape):
        """
        :param pred_map: [H, W] 概率图
        :param ori_shape: (h, w), 原图大小
        :param resize_shape: (h, w), 网络输入大小
        :return: [(box, score), ...]
        """

        # 二值化
        segmentation = (pred_map > self.thresh).astype(numpy.uint8)
        if self.dilation_kernel is not None:
            mask = cv2.dilate(segmentation, self.dilation_kernel)
        else:
            mask = segmentation

        boxes = []
        scores = []

        # poly 模式 or quad 模式
        if self.box_type == "poly":
            boxes, scores = self.polygons_from_bitmap(pred_map, mask, ori_shape, resize_shape)
        else:
            boxes, scores = self.boxes_from_bitmap(pred_map, mask, ori_shape, resize_shape)

        return [(b, s) for b, s in zip(boxes, scores)]

    # ---------------- 核心函数 ----------------
    def polygons_from_bitmap(self, pred, bitmap, ori_shape, resize_shape):
        h, w = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(numpy.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes, scores = [], []
        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score(pred, points)
            if score < self.box_thresh:
                continue

            box = self.unclip(points)
            if len(box) == 0:
                continue
            box = numpy.array(box).reshape(-1, 2)

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box = self.map_back(box, ori_shape, resize_shape)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, bitmap, ori_shape, resize_shape):
        h, w = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(numpy.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes, scores = [], []
        for contour in contours[:self.max_candidates]:
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = numpy.array(points)

            score = self.box_score(pred, points)
            if score < self.box_thresh:
                continue

            box = self.unclip(points)
            if len(box) == 0:
                continue
            box = numpy.array(box[0]).reshape(-1, 2)

            box, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue
            box = numpy.array(box)

            box = self.map_back(box, ori_shape, resize_shape)
            boxes.append(box.astype("int32").tolist())
            scores.append(score)
        return boxes, scores

    # ---------------- 工具函数 ----------------
    def unclip(self, box):
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = cv2.boxPoints(bounding_box)
        points = sorted(list(points), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score(self, pred, points):
        if self.score_mode == "fast":
            return self.box_score_fast(pred, points)
        else:
            return self.box_score_slow(pred, points)

    def box_score_fast(self, bitmap, box):
        h, w = bitmap.shape[:2]
        box = numpy.array(box).copy()
        xmin = numpy.clip(numpy.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = numpy.clip(numpy.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = numpy.clip(numpy.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = numpy.clip(numpy.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = numpy.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=numpy.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, [box.astype("int32")], 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, box):
        h, w = bitmap.shape[:2]
        contour = numpy.array(box).reshape((-1, 2))
        xmin = numpy.clip(numpy.min(contour[:, 0]), 0, w - 1)
        xmax = numpy.clip(numpy.max(contour[:, 0]), 0, w - 1)
        ymin = numpy.clip(numpy.min(contour[:, 1]), 0, h - 1)
        ymax = numpy.clip(numpy.max(contour[:, 1]), 0, h - 1)

        mask = numpy.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=numpy.uint8)
        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin
        cv2.fillPoly(mask, [contour.astype("int32")], 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def map_back(self, box, ori_shape, resize_shape):
        ori_h, ori_w = ori_shape
        resize_h, resize_w = resize_shape
        box[:, 0] = numpy.clip(numpy.round(box[:, 0] / resize_w * ori_w), 0, ori_w - 1)
        box[:, 1] = numpy.clip(numpy.round(box[:, 1] / resize_h * ori_h), 0, ori_h - 1)
        return box


class TextDetector(PredictBase):
    def __init__(
            self,
            model_name: DetModels = DetModels.MOBILE,
            model_path: str | None = None,
            model_local_dir: str = "models",
            providers: list = None,
            session_options: onnxruntime.SessionOptions = None,
            executor: ThreadPoolExecutor | None = None,
            thresh: float = 0.3,
            box_thresh: float = 0.6,
            max_candidates: int = 1000,
            unclip_ratio: float = 1.5,
            use_dilation: bool = False,
            score_mode: Literal["fast", "slow"] = "fast",
            box_type: Literal["poly", "quad"] = "quad"
    ):
        self.postprocess_op = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            unclip_ratio=unclip_ratio,
            use_dilation=use_dilation,
            score_mode=score_mode,
            box_type=box_type,
        )
        super().__init__(model_name, model_path, model_local_dir, providers, session_options, executor)

    def _resize_image(self, img, max_side_len=960):
        """DetResizeForTest: 长边=960，保持比例并对齐32"""
        h, w, _ = img.shape
        scale = 1.0
        if max(h, w) > max_side_len:
            scale = float(max_side_len) / float(max(h, w))
        resize_h = int(h * scale)
        resize_w = int(w * scale)
        resize_h = int(math.ceil(resize_h / 32) * 32)
        resize_w = int(math.ceil(resize_w / 32) * 32)
        resized = cv2.resize(img, (resize_w, resize_h))
        return resized, (h, w), (resize_h, resize_w)

    def _preprocess_sync(self, image: numpy.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        resized, ori_shape, resize_shape = self._resize_image(image, max_side_len=960)
        img = resized.astype(numpy.float32) / 255.0
        mean = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
        std = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)
        img = (img - mean) / std
        img = numpy.transpose(img, (2, 0, 1))
        img = numpy.expand_dims(img, axis=0).astype(numpy.float32)
        return img, ori_shape, resize_shape

    async def _run_inference(self, blob, ori_shape, resize_shape) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        preds = await loop.run_in_executor(self.executor, self._run_inference_sync, blob)
        prob_map = preds[0, 0, :, :]  # [1,1,H,W] -> [H,W]
        boxes = self.postprocess_op(prob_map, ori_shape=ori_shape, resize_shape=resize_shape)
        return {
            "num_boxes": len(boxes),
            "boxes": [b for b, s in boxes],
            "scores": [s for b, s in boxes]
        }

    async def predict_from_array(self, img_array: numpy.ndarray) -> Dict[str, Any]:
        blob, ori_shape, resize_shape = await self.preprocess(img_array)
        return await self._run_inference(blob, ori_shape, resize_shape)
