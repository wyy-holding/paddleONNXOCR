import asyncio
import cv2
import numpy
import onnxruntime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
from paddleONNXOCR import TableModels
from paddleONNXOCR.predict.predict_base import PredictBase


class TableCellDetector(PredictBase):
    def __init__(
            self,
            model_name: TableModels = TableModels.WIRED_TABLE,
            model_path: str | None = None,
            model_local_dir: str = "models",
            providers: list = None,
            session_options: onnxruntime.SessionOptions = None,
            executor: ThreadPoolExecutor | None = None,
            threshold: float = 0.5,
            input_size: int = 640
    ):
        """
        :param model_name: 模型名称
        :param model_path: 模型路径
        :param model_local_dir: 模型下载到本地路径
        :param providers: onnx providers，默认由于PaddleONNOCRXUtils.get_available_providers选择
        :param session_options: onnxruntime.SessionOptions对象
        :param executor: 线程池
        :param threshold: 置信度阈值
        :param input_size: 输入尺寸
        """
        self.threshold = threshold
        self.input_size = input_size
        super().__init__(model_name, model_path, model_local_dir, providers, session_options, executor)

    def _preprocess_sync(self, image: numpy.ndarray) -> tuple:
        original_height, original_width = image.shape[:2]

        # 直接缩放到640x640
        resized_image = cv2.resize(image, (self.input_size, self.input_size))

        # 归一化
        normalized_image = resized_image.astype(numpy.float32) / 255.0

        # HWC转CHW并添加batch维度
        chw_image = normalized_image.transpose(2, 0, 1)
        batch_image = numpy.expand_dims(chw_image, axis=0)

        # 注意：im_shape 可能需要是 [height, width] 格式
        im_shape = numpy.array([[self.input_size, self.input_size]], dtype=numpy.float32)

        # scale_factor 计算方式可能需要调整
        scale_factor = numpy.array([[1.0, 1.0]], dtype=numpy.float32)

        scale_info = {
            'scale_factor_x': self.input_size / original_width,
            'scale_factor_y': self.input_size / original_height,
            'original_width': original_width,
            'original_height': original_height
        }

        return batch_image, im_shape, scale_factor, scale_info

    def _run_inference_sync(self, blob: Tuple) -> Any:
        batch_image, im_shape, scale_factor, scale_info = blob
        # 构造输入字典
        self.onnx_session_run_inputs = {
            'image': batch_image,
            'im_shape': im_shape,
            'scale_factor': scale_factor
        }
        outputs = self.session.run(self.output_names, self.onnx_session_run_inputs)
        # 后处理
        boxes = asyncio.run(self.postprocess(outputs, scale_info))
        return {
            'boxes': boxes
        }

    async def _run_inference(self, preprocess_result: Tuple) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._run_inference_sync,
            preprocess_result
        )

    async def _apply_nms(self, boxes: numpy.ndarray, scores: numpy.ndarray, labels: numpy.ndarray,
                         iou_threshold: float = 0.3) -> tuple:
        """
        应用非极大值抑制

        Args:
            boxes: 边界框 [N, 4]
            scores: 置信度分数 [N]
            labels: 类别标签 [N]
            iou_threshold: IoU阈值

        Returns:
            过滤后的boxes, scores, labels
        """
        if len(boxes) == 0:
            return boxes, scores, labels

            # 计算面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # 按分数排序
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # 计算IoU
            xx1 = numpy.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = numpy.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = numpy.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = numpy.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = numpy.maximum(0, xx2 - xx1)
            h = numpy.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留IoU小于阈值的框
            inds = numpy.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return boxes[keep], scores[keep], labels[keep]

    async def postprocess(self, outputs: List[numpy.ndarray], scale_info: Dict) -> List[Dict]:
        """
        后处理检测结果 - 包含NMS处理

        Args:
            outputs: 模型输出
            scale_info: 缩放信息

        Returns:
            检测结果列表
        """
        if len(outputs) == 2:
            boxes_data = outputs[0]  # [300, 6]
            num_boxes = int(outputs[1][0]) if outputs[1].ndim > 0 else len(boxes_data)
            results = []
            # 遍历每个检测框
            for i in range(min(num_boxes, len(boxes_data))):
                box_data = boxes_data[i]  # 获取第i个检测框的数据 [6]
                # 确保box_data是数组且长度为6
                if hasattr(box_data, '__len__') and len(box_data) >= 6:
                    cls_id = int(box_data[0])
                    score = float(box_data[1])
                    x1, y1, x2, y2 = box_data[2:6]
                else:
                    # 如果box_data不是预期格式，跳过
                    continue
                    # 过滤低置信度检测
                if score < self.threshold:
                    continue

                    # 转换坐标到原始图像 - 移除padding偏移，直接使用缩放比例
                # 使用对应方向的缩放比例
                x1 /= scale_info['scale_factor_x']
                x2 /= scale_info['scale_factor_x']
                y1 /= scale_info['scale_factor_y']
                y2 /= scale_info['scale_factor_y']

                # 限制在图像边界内
                x1 = max(0, min(x1, scale_info['original_width']))
                y1 = max(0, min(y1, scale_info['original_height']))
                x2 = max(0, min(x2, scale_info['original_width']))
                y2 = max(0, min(y2, scale_info['original_height']))

                results.append({
                    'cls_id': cls_id,
                    'label': 'cell',
                    'score': score,
                    'coordinate': [float(x1), float(y1), float(x2), float(y2)]
                })

                # 应用NMS处理
            if len(results) > 0:
                # 提取boxes, scores, labels用于NMS
                boxes = numpy.array([r['coordinate'] for r in results])
                scores = numpy.array([r['score'] for r in results])
                labels = numpy.array([r['cls_id'] for r in results])

                # 应用NMS - 使用官方推荐的阈值0.3
                filtered_boxes, filtered_scores, filtered_labels = await self._apply_nms(
                    boxes, scores, labels, iou_threshold=0.3
                )

                # 重新构建results
                results = []
                for i in range(len(filtered_boxes)):
                    results.append({
                        'cls_id': int(filtered_labels[i]),
                        'label': 'cell',
                        'score': float(filtered_scores[i]),
                        'coordinate': filtered_boxes[i].tolist()
                    })

        else:
            raise ValueError(f"不支持的输出格式，输出数量: {len(outputs)}")
        return results
