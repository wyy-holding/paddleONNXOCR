import asyncio

import onnxruntime
import numpy
import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Dict, Any
from paddleONNXOCR import ImageModels
from paddleONNXOCR.predict.predict_base import PredictBase


class LayoutDetection(PredictBase):
    def __init__(
            self,
            model_name: ImageModels = ImageModels.DOC_LAYOUT_PLUS_L,
            model_path: str | None = None,
            model_local_dir: str = "models",
            providers: Optional[list] = None,
            session_options: Optional[onnxruntime.SessionOptions] = None,
            executor: ThreadPoolExecutor | None = None,
            threshold: float = 0.3
    ):
        """
        :param model_name: 模型名称
        :param model_path: 模型路径
        :param model_local_dir: 模型下载到本地路径
        :param providers: onnx providers，默认由于PaddleONNOCRXUtils.get_available_providers选择
        :param session_options: onnxruntime.SessionOptions对象
        :param executor: 线程池
        :param threshold: 置信度阈值
        """
        self.threshold = threshold
        self.labels = [
            "paragraph_title", "image", "text", "number", "abstract", "content",
            "figure_table_chart_title", "formula", "table", "reference", "doc_title",
            "footnote", "header", "algorithm", "footer", "seal", "chart",
            "formula_number", "aside_text", "reference_content"
        ]
        self.input_size = None
        super().__init__(model_name, model_path, model_local_dir, providers, session_options, executor)

    async def _get_model_input_size(self) -> Tuple[int, int]:
        """
        从ONNX模型中获取输入尺寸
        Returns:
            (height, width) 输入尺寸
        """
        for input_info in self.session.get_inputs():
            if input_info.name == 'image':
                shape = input_info.shape
                if len(shape) == 4:
                    height = shape[2] if isinstance(shape[2], int) else 480
                    width = shape[3] if isinstance(shape[3], int) else 480
                    return height, width
        # 默认尺寸
        return 480, 480

    def _preprocess_sync(
            self,
            image: numpy.ndarray
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, Tuple[int, int]]:
        """
        预处理输入图像
        Args:
            image_path: 图像文件路径
        Returns:
            处理后的图像数组、im_shape、scale_factor和原始图像尺寸
        """
        # 读取图像并转换为RGB格式
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img_rgb.shape[:2]
        # 使用模型特定的输入尺寸
        target_h, target_w = self.input_size
        resized_img = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        # 计算缩放因子
        scale_w = target_w / ori_w
        scale_h = target_h / ori_h
        # 图像归一化
        normalized_img = resized_img.astype(numpy.float32) / 255.0
        # 转换为CHW格式
        img_chw = normalized_img.transpose(2, 0, 1)
        # 准备模型输入
        img_batch = numpy.expand_dims(img_chw, axis=0)  # (1, 3, H, W)
        im_shape = numpy.array([[target_h, target_w]], dtype=numpy.float32)  # (1, 2) [h, w]
        scale_factor = numpy.array([[scale_h, scale_w]], dtype=numpy.float32)  # (1, 2) [h_scale, w_scale]
        return img_batch, im_shape, scale_factor, (ori_w, ori_h)

    async def postprocess(self, boxes: numpy.ndarray,
                          box_nums: numpy.ndarray,
                          ori_size: Tuple[int, int],
                          layout_nms: bool = True,
                          layout_unclip_ratio: Optional[Tuple[float, float]] = None) -> List[Dict]:
        """
        后处理检测结果
        Args:
            boxes: 检测框 (N, 6)
            box_nums: 框数量
            ori_size: 原始图像尺寸 (w, h)
            layout_nms: 是否使用布局NMS
            layout_unclip_ratio: 边界框扩展比例
        Returns:
            后处理后的检测结果列表
        """
        if len(boxes) == 0:
            return []
        # 阈值过滤
        valid_mask = (boxes[:, 1] > self.threshold) & (boxes[:, 0] >= 0)
        filtered_boxes = boxes[valid_mask]
        if len(filtered_boxes) == 0:
            return []
        # 布局NMS
        if layout_nms:
            keep_indices = await self._layout_nms(filtered_boxes)
            filtered_boxes = filtered_boxes[keep_indices]
        # 边界框扩展
        if layout_unclip_ratio is not None:
            filtered_boxes = self._unclip_boxes(filtered_boxes, layout_unclip_ratio)
        # 转换为最终结果格式
        results = []
        ori_w, ori_h = ori_size
        for box in filtered_boxes:
            cls_id, score, x1, y1, x2, y2 = box
            # 确保坐标在图像范围内
            x1 = max(0, min(ori_w, x1))
            y1 = max(0, min(ori_h, y1))
            x2 = max(0, min(ori_w, x2))
            y2 = max(0, min(ori_h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append({
                'cls_id': int(cls_id),
                'label': self.labels[int(cls_id)],
                'score': float(score),
                'coordinate': [float(x1), float(y1), float(x2), float(y2)]
            })
        return results

    async def _layout_nms(self, boxes: numpy.ndarray, iou_same: float = 0.6, iou_diff: float = 0.98) -> List[int]:
        """
        布局感知的非最大值抑制
        """
        scores = boxes[:, 1]
        indices = numpy.argsort(scores)[::-1]
        selected_indices = []
        while len(indices) > 0:
            current = indices[0]
            current_box = boxes[current]
            current_class = current_box[0]
            current_coords = current_box[2:]

            selected_indices.append(current)
            indices = indices[1:]

            filtered_indices = []
            for i in indices:
                box = boxes[i]
                box_class = box[0]
                box_coords = box[2:]
                iou_value = await self._calculate_iou(current_coords, box_coords)
                threshold = iou_same if current_class == box_class else iou_diff
                if iou_value < threshold:
                    filtered_indices.append(i)
            indices = numpy.array(filtered_indices)
        return selected_indices

    async def _calculate_iou(self, box1: numpy.ndarray, box2: numpy.ndarray) -> float:
        """
        计算两个边界框的IoU
        """
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        # 计算交集坐标
        x1_i = max(x1, x1_p)
        y1_i = max(y1, y1_p)
        x2_i = min(x2, x2_p)
        y2_i = min(y2, y2_p)
        # 计算交集面积
        inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        # 计算并集面积
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    async def _unclip_boxes(self, boxes: numpy.ndarray, unclip_ratio: Tuple[float, float]) -> numpy.ndarray:
        """
        扩展边界框
        """
        width_ratio, height_ratio = unclip_ratio
        expanded_boxes = boxes.copy()
        for i, box in enumerate(boxes):
            cls_id, score, x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            center_x = x1 + width / 2
            center_y = y1 + height / 2
            new_width = width * width_ratio
            new_height = height * height_ratio
            new_x1 = center_x - new_width / 2
            new_y1 = center_y - new_height / 2
            new_x2 = center_x + new_width / 2
            new_y2 = center_y + new_height / 2
            expanded_boxes[i] = [cls_id, score, new_x1, new_y1, new_x2, new_y2]
        return expanded_boxes

    def _run_inference_sync(self, blob: Tuple) -> Any:
        img_batch, im_shape, scale_factor, ori_size = blob
        input_dict = {'image': img_batch}
        if 'scale_factor' in self.input_names:
            input_dict['scale_factor'] = scale_factor
        if 'im_shape' in self.input_names:
            input_dict['im_shape'] = im_shape
        outputs = self.session.run(None, input_dict)
        boxes = outputs[0]  # fetch_name_0: (N, 6) [cls_id, score, x1, y1, x2, y2]
        box_nums = outputs[1]  # fetch_name_1: (batch_size,)
        return asyncio.run(self.postprocess(boxes, box_nums, ori_size, True, None))

    async def _run_inference(self, preprocess_result: Tuple) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._run_inference_sync,
            preprocess_result
        )

    async def __aenter__(self):
        await super().__aenter__()
        self.input_size = await self._get_model_input_size()
        return self
