import asyncio
import atexit
import io
import pdfplumber
import aiohttp
import fitz
import numpy
import cv2
import base64
import math
import onnxruntime
import filetype
import os
import validators
from pathlib import Path
from urllib.parse import urlparse
from typing import Sequence, Tuple, List, Callable, Awaitable, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from deskew import determine_skew
from modelscope import snapshot_download

from paddleONNXOCR.file_download import file_downloader
from paddleONNXOCR.models_enum import *
from paddleONNXOCR.predict.ocr_dataclass import OCRResult, PdfPageResult, PdfResult


class PaddleONNOCRXUtils:
    @staticmethod
    async def get_available_providers():
        """
        获取onnxruntime可用providers
        :return: 可用providers列表
        """
        available_providers = onnxruntime.get_available_providers()
        preferred = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
            "AzureExecutionProvider"
        ]
        return [p for p in preferred if p in available_providers]

    @staticmethod
    async def get_onnx_session_options():
        """
        获取onnx会话选项
        :return: onnx会话选项
        """
        session_options = onnxruntime.SessionOptions()
        session_options.enable_cpu_mem_arena = False
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4
        return session_options

    @staticmethod
    async def get_onnx_session(
            model_path: str,
            providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
            sess_options: onnxruntime.SessionOptions | None = None
    ) -> onnxruntime.InferenceSession:
        """
        获取onnx会话
        :param model_path: 模型路径
        :param providers: 自定义的providers
        :param sess_options: 自定义onnx会话选项
        :return:
        """
        return onnxruntime.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

    @staticmethod
    async def get_onnx_session_params(
            session: onnxruntime.InferenceSession | None = None
    ) -> Tuple[list, list, list]:
        """
        获取onnx会话参数
        :param session: onnx会话
        :return: input_names, output_names, input_shape
        """
        return [inp.name for inp in session.get_inputs()], [output.name for output in session.get_outputs()], \
            session.get_inputs()[0].shape

    @staticmethod
    async def get_aiohttp_session() -> aiohttp.ClientSession:
        """
        获取aiohttp session会话
        :return: aiohttp session
        """
        timeout = aiohttp.ClientTimeout(total=30)
        http_session = aiohttp.ClientSession(timeout=timeout)
        return http_session

    @staticmethod
    async def rotate_image(image: numpy.ndarray) -> numpy.ndarray:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale, max_angle=180)
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(numpy.sin(angle_radian) * old_height) + abs(numpy.cos(angle_radian) * old_width)
        height = abs(numpy.sin(angle_radian) * old_width) + abs(numpy.cos(angle_radian) * old_height)
        image_center = tuple(numpy.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=(0, 0, 0))

    @staticmethod
    async def sorted_boxes(boxes: list) -> list:
        """
        将检测到的文本框排序：从上到下，从左到右
        :param boxes: list of boxes，每个 box 是 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        :return: 排序后的 boxes
        """
        # 取每个 box 的平均 y 作为排序的主键，平均 x 用来辅助判断
        boxes = sorted(boxes, key=lambda b: (numpy.mean([p[1] for p in b]),
                                             numpy.mean([p[0] for p in b])))

        # 行内再精修：如果在同一行（y 均值差小于一定阈值），按 x 排序
        line_threshold = 10  # 像素阈值，可调
        res = []
        current_line = []
        last_y = None

        for b in boxes:
            cy = numpy.mean([p[1] for p in b])
            if last_y is None or abs(cy - last_y) <= line_threshold:
                current_line.append(b)
                last_y = cy if last_y is None else (last_y + cy) / 2
            else:
                # 当前行结束，左到右排序
                current_line = sorted(current_line, key=lambda b: numpy.mean([p[0] for p in b]))
                res.extend(current_line)
                current_line = [b]
                last_y = cy
        if current_line:
            current_line = sorted(current_line, key=lambda b: numpy.mean([p[0] for p in b]))
            res.extend(current_line)

        return res


class TextBoxSorter:
    """文本框排序工具类"""

    @staticmethod
    def detect_layout(boxes: List[List[List[float]]]) -> str:
        """
        自动检测文本布局方向

        Args:
            boxes: 文本框列表，每个框是4个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            "horizontal" 或 "vertical"
        """
        if len(boxes) < 2:
            return "horizontal"  # 默认返回横向

        # 计算所有框的信息
        box_infos = []
        for box in boxes:
            box_array = numpy.array(box)
            x_min = numpy.min(box_array[:, 0])
            x_max = numpy.max(box_array[:, 0])
            y_min = numpy.min(box_array[:, 1])
            y_max = numpy.max(box_array[:, 1])

            box_infos.append({
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'cx': (x_min + x_max) / 2,
                'cy': (y_min + y_max) / 2,
                'width': x_max - x_min,
                'height': y_max - y_min
            })

        # === 方法1: 整体布局宽高比 ===
        all_x_min = min(box['x_min'] for box in box_infos)
        all_x_max = max(box['x_max'] for box in box_infos)
        all_y_min = min(box['y_min'] for box in box_infos)
        all_y_max = max(box['y_max'] for box in box_infos)

        total_width = all_x_max - all_x_min
        total_height = all_y_max - all_y_min
        layout_ratio = total_width / total_height if total_height > 0 else 1.0

        # 强横向或强纵向可直接返回
        if layout_ratio > 1.6:
            return "horizontal"
        if layout_ratio < 0.6:
            return "vertical"

        # === 方法2: 过滤有效文本框（排除标点、页码等小框）===
        # 根据你的数据，正文宽度通常 > 50，高度 > 15
        valid_boxes = [box for box in box_infos if box['width'] > 50 and box['height'] > 15]
        if not valid_boxes:
            valid_boxes = box_infos  # 退化处理

        # 计算平均宽高比
        aspect_ratios = [box['width'] / box['height'] if box['height'] > 0 else 1.0 for box in valid_boxes]
        avg_aspect_ratio = float(numpy.mean(aspect_ratios))

        wide_boxes = sum(1 for box in valid_boxes if box['width'] > box['height'] * 1.3)
        tall_boxes = sum(1 for box in valid_boxes if box['height'] > box['width'] * 1.3)

        # 基于形状判断
        if avg_aspect_ratio > 1.4 or wide_boxes > tall_boxes * 1.2:
            return "horizontal"
        if avg_aspect_ratio < 0.7 or tall_boxes > wide_boxes * 1.2:
            return "vertical"

        # === 方法3: 位置关系得分（仅当上述不确定时使用）===
        horizontal_score = 0
        vertical_score = 0

        for i in range(len(box_infos)):
            for j in range(i + 1, len(box_infos)):
                b1, b2 = box_infos[i], box_infos[j]
                dx = abs(b1['cx'] - b2['cx'])
                dy = abs(b1['cy'] - b2['cy'])

                # Y轴重叠率（用于判断是否同行）
                y_overlap = max(0, min(b1['y_max'], b2['y_max']) - max(b1['y_min'], b2['y_min']))
                min_h = min(b1['height'], b2['height'])
                y_overlap_ratio = y_overlap / min_h if min_h > 0 else 0

                # X轴重叠率（用于判断是否同列）
                x_overlap = max(0, min(b1['x_max'], b2['x_max']) - max(b1['x_min'], b2['x_min']))
                min_w = min(b1['width'], b2['width'])
                x_overlap_ratio = x_overlap / min_w if min_w > 0 else 0

                if y_overlap_ratio > 0.5 and dx > dy:
                    horizontal_score += 1
                if x_overlap_ratio > 0.5 and dy > dx:
                    vertical_score += 1

        # 仅当得分差异显著时才信任
        if horizontal_score > vertical_score * 1.5:
            return "horizontal"
        if vertical_score > horizontal_score * 1.5:
            return "vertical"

        # === 默认：横向（绝大多数文档为横向）===
        return "horizontal"

    @staticmethod
    def sort_boxes(
            boxes: List[List[List[float]]],
            scores: List[float] | None = None
    ) -> Tuple[List[List[List[float]]], List[float] | None]:
        """
        自动检测布局并对文本框进行排序

        Args:
            boxes: 文本框列表，每个框是4个点的坐标
            scores: 对应的分数列表（可选）

        Returns:
            排序后的文本框和分数
        """
        if len(boxes) == 0:
            return boxes, scores

        # 自动检测布局方向
        layout = TextBoxSorter.detect_layout(boxes)

        # 根据检测结果选择排序方式
        if layout == "vertical":
            return TextBoxSorter._sort_vertical(boxes, scores)
        else:
            return TextBoxSorter._sort_horizontal(boxes, scores)

    @staticmethod
    def _sort_horizontal(
            boxes: List[List[List[float]]],
            scores: List[float] | None = None
    ) -> Tuple[List[List[List[float]]], List[float] | None]:
        """横向排序（从上到下，从左到右）"""
        if len(boxes) == 0:
            return boxes, scores

        # 计算每个框的中心点和边界
        box_infos = []
        for i, box in enumerate(boxes):
            box_array = numpy.array(box)
            # 计算框的边界
            x_min = numpy.min(box_array[:, 0])
            x_max = numpy.max(box_array[:, 0])
            y_min = numpy.min(box_array[:, 1])
            y_max = numpy.max(box_array[:, 1])

            # 计算中心点
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

            box_info = {
                'index': i,
                'box': box,
                'score': scores[i] if scores else None,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'cx': cx,
                'cy': cy,
                'height': y_max - y_min,
                'width': x_max - x_min
            }
            box_infos.append(box_info)

        # 按行分组（基于Y坐标的重叠）
        rows = TextBoxSorter._group_by_rows(box_infos)

        # 对每行内的框按X坐标排序
        sorted_box_infos = []
        for row in rows:
            # 行内按x_min排序
            row.sort(key=lambda x: x['x_min'])
            sorted_box_infos.extend(row)

        # 提取排序后的结果
        sorted_boxes = [info['box'] for info in sorted_box_infos]
        if scores:
            sorted_scores = [info['score'] for info in sorted_box_infos]
            return sorted_boxes, sorted_scores
        return sorted_boxes, None

    @staticmethod
    def _group_by_rows(
            box_infos: List[Dict]
    ) -> List[List[Dict]]:
        if not box_infos:
            return []

        # 按垂直中心 cy 排序
        box_infos.sort(key=lambda x: x['cy'])

        # 估计行高：取所有框高度的中位数或均值
        heights = [box['height'] for box in box_infos]
        median_height = float(numpy.median(heights)) if heights else 20.0
        row_threshold = min(median_height * 0.5, 20.0)
        rows = []
        current_row = [box_infos[0]]
        for box in box_infos[1:]:
            # 如果当前框的 cy 与当前行最后一个框的 cy 差距小于阈值，则归入该行
            last_box = current_row[-1]
            if abs(box['cy'] - last_box['cy']) <= row_threshold:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]

        if current_row:
            rows.append(current_row)

        # 对每一行内部按 x_min 排序（已在 sort_horizontal 中做，但这里也可做）
        for row in rows:
            row.sort(key=lambda x: x['x_min'])

        return rows
    @staticmethod
    def _sort_vertical(
            boxes: List[List[List[float]]],
            scores: List[float] | None = None
    ) -> Tuple[List[List[List[float]]], List[float] | None]:
        """纵向排序（从右到左，从上到下）"""
        if len(boxes) == 0:
            return boxes, scores

        # 计算每个框的信息
        box_infos = []
        for i, box in enumerate(boxes):
            box_array = numpy.array(box)
            x_min = numpy.min(box_array[:, 0])
            x_max = numpy.max(box_array[:, 0])
            y_min = numpy.min(box_array[:, 1])
            y_max = numpy.max(box_array[:, 1])

            box_info = {
                'index': i,
                'box': box,
                'score': scores[i] if scores else None,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'height': y_max - y_min,
                'width': x_max - x_min
            }
            box_infos.append(box_info)

        # 按列分组（从右到左）
        box_infos.sort(key=lambda x: -x['x_max'])  # 按x_max降序

        # 分列
        columns = []
        current_col = [box_infos[0]]
        current_col_x_min = box_infos[0]['x_min']
        current_col_x_max = box_infos[0]['x_max']

        for box_info in box_infos[1:]:
            # 计算X方向的重叠
            x_overlap = min(current_col_x_max, box_info['x_max']) - max(current_col_x_min, box_info['x_min'])
            box_width = box_info['x_max'] - box_info['x_min']

            if x_overlap > box_width * 0.5:
                current_col.append(box_info)
                current_col_x_min = min(current_col_x_min, box_info['x_min'])
                current_col_x_max = max(current_col_x_max, box_info['x_max'])
            else:
                columns.append(current_col)
                current_col = [box_info]
                current_col_x_min = box_info['x_min']
                current_col_x_max = box_info['x_max']

        if current_col:
            columns.append(current_col)

        # 每列内按y排序
        sorted_box_infos = []
        for col in columns:
            col.sort(key=lambda x: x['y_min'])
            sorted_box_infos.extend(col)

        # 提取结果
        sorted_boxes = [info['box'] for info in sorted_box_infos]
        if scores:
            sorted_scores = [info['score'] for info in sorted_box_infos]
            return sorted_boxes, sorted_scores
        return sorted_boxes, None

    @staticmethod
    def sort_boxes_in_reading_order(
            boxes: List[List[List[float]]],
            scores: List[float] | None = None,
            layout: str = "auto"
    ) -> Tuple[List[List[List[float]]], List[float] | None]:
        """
        根据不同的布局模式对文本框进行排序

        Args:
            boxes: 文本框列表
            scores: 分数列表
            layout: 布局模式 - "auto"（自动检测）, "horizontal"（横向）, "vertical"（纵向）

        Returns:
            排序后的文本框和分数
        """
        if layout == "auto":
            return TextBoxSorter.sort_boxes(boxes, scores)
        elif layout == "horizontal":
            return TextBoxSorter._sort_horizontal(boxes, scores)
        elif layout == "vertical":
            return TextBoxSorter._sort_vertical(boxes, scores)
        else:
            return TextBoxSorter.sort_boxes(boxes, scores)


class UtilsCommon:
    @staticmethod
    async def is_base64_image(content: str) -> bool:
        """判断是否为 Base64 图片，若是则返回 True，否则返回 False。"""
        if content.startswith("data:image/"):
            content = content.split(",", 1)[1]
        try:
            decoded = base64.b64decode(content, validate=True)
            Image.open(io.BytesIO(decoded)).convert("RGB")
            return True
        except Exception:
            return False

    @staticmethod
    async def base64_to_numpy_rgb(content: str) -> Union[numpy.ndarray, None]:
        """将 Base64 图像字符串转换为 RGB 格式的 numpy 数组；若失败则返回 None。"""
        if content.startswith("data:image/"):
            content = content.split(",", 1)[1]
        try:
            decoded = base64.b64decode(content, validate=True)
            image = Image.open(io.BytesIO(decoded)).convert("RGB")
            return numpy.array(image)
        except Exception:
            return None

    @staticmethod
    async def download_model(
            model_name: TextLineModels | DetModels | RecModels | TableModels | ImageModels,
            local_dir: str
    ):
        snapshot_download(
            f'wyyHolding/{model_name.value}',
            local_dir=local_dir,
            allow_patterns=["*.onnx"]
        )


class TableHTMLGenerator:
    """
    将表格单元格检测、文本检测和文本识别结果合并生成HTML表格的工具类。
    """

    def __init__(
            self,
            cell_detection_results: Dict,
            ocr_result: OCRResult,
            iou_threshold: float = 0.7,
            alignment_threshold: float = 0.5,
            coordinate_tolerance: int = 8
    ):
        """
        初始化参数

        Args:
            cell_detection_results: 表格单元格检测结果
            ocr_result: 文本检测结果,
            iou_threshold (float): 匹配文本框与单元格时的IoU阈值
            alignment_threshold (float): 判断文本对齐方式的偏移比例阈值
            coordinate_tolerance (int): 坐标聚类时的容差
        """
        self.cell_boxes = []
        self.ocr_boxes = []
        self.ocr_results = []
        self.cell_detection_results = cell_detection_results
        self.ocr_result = ocr_result
        self.iou_threshold = iou_threshold
        self.alignment_threshold = alignment_threshold
        self.coordinate_tolerance = coordinate_tolerance
        self.init_params()

    def init_params(self):
        self.cell_boxes = [box['coordinate'][:4] for box in self.cell_detection_results['boxes']]
        for item in self.ocr_result.results:
            self.ocr_boxes.append(item.box)
            self.ocr_results.append(item.text)

    @staticmethod
    def convert_four_points_to_bbox(four_points: List[List[int]]) -> List[float]:
        """将四个点的格式转换为bbox格式 [x1, y1, x2, y2]"""
        x_coords = [point[0] for point in four_points]
        y_coords = [point[1] for point in four_points]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    def compute_intersection_over_text(self, cell_box: List[float], text_box: List[float]) -> float:
        """计算文本框与单元格的交集占文本框面积的比例（IoT）"""
        x1_1, y1_1, x2_1, y2_1 = map(float, cell_box)
        x1_2, y1_2, x2_2, y2_2 = map(float, text_box)

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        inter_width = max(0, x_right - x_left)
        inter_height = max(0, y_bottom - y_top)
        inter_area = inter_width * inter_height

        text_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        return inter_area / text_area if text_area > 0 else 0

    def calculate_text_alignment(self, cell_box: List[float], text_box: List[float]) -> str:
        """根据文本框在单元格中的位置判断对齐方式"""
        cell_x1, cell_y1, cell_x2, cell_y2 = cell_box
        text_x1, text_y1, text_x2, text_y2 = text_box

        cell_center = (cell_x1 + cell_x2) / 2
        text_center = (text_x1 + text_x2) / 2
        cell_width = cell_x2 - cell_x1

        offset_ratio = (text_center - cell_center) / (cell_width / 2) if cell_width > 0 else 0

        if offset_ratio < -self.alignment_threshold:
            return "left"
        elif offset_ratio > self.alignment_threshold:
            return "right"
        else:
            return "center"

    def sort_table_cells_boxes(self, boxes: List[List[float]]) -> Tuple[List[List[float]], List[int]]:
        """对单元格按行优先排序"""
        boxes_sorted_by_y = sorted(boxes, key=lambda box: box[1])
        rows = []
        current_row = []
        current_y = None
        tolerance = 10  # 行对齐容差

        for box in boxes_sorted_by_y:
            y1 = box[1]
            if current_y is None or abs(y1 - current_y) <= tolerance:
                current_row.append(box)
            else:
                current_row.sort(key=lambda x: x[0])  # 按x排序
                rows.append(current_row)
                current_row = [box]
                current_y = y1

        if current_row:
            current_row.sort(key=lambda x: x[0])
            rows.append(current_row)

        sorted_boxes = []
        flag = [0]
        for row in rows:
            sorted_boxes.extend(row)
            flag.append(flag[-1] + len(row))

        return sorted_boxes, flag

    def match_text_to_cells(self, cell_boxes: List[List[float]], text_boxes: List[List[float]]) -> Dict[int, List[int]]:
        """匹配每个文本框到对应的单元格"""
        matched = {}
        for i, cell_box in enumerate(cell_boxes):
            for j, text_box in enumerate(text_boxes):
                if self.compute_intersection_over_text(cell_box, text_box) > self.iou_threshold:
                    matched.setdefault(i, []).append(j)
        return matched

    def generate_table_structure_from_cells(self, cell_boxes: List[List[float]]) -> Tuple:
        """从单元格坐标生成表格结构：行列数、rowspan/colspan、cell_map"""
        x_coords = [x for cell in cell_boxes for x in (cell[0], cell[2])]
        y_coords = [y for cell in cell_boxes for y in (cell[1], cell[3])]

        def cluster_positions(positions, tol):
            if not positions:
                return []
            positions = sorted(set(positions))
            clusters = []
            current_cluster = [positions[0]]
            for pos in positions[1:]:
                if abs(pos - current_cluster[-1]) <= tol:
                    current_cluster.append(pos)
                else:
                    clusters.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [pos]
            clusters.append(sum(current_cluster) / len(current_cluster))
            return clusters

        x_positions = cluster_positions(x_coords, self.coordinate_tolerance)
        y_positions = cluster_positions(y_coords, self.coordinate_tolerance)

        num_rows = len(y_positions) - 1
        num_cols = len(x_positions) - 1

        cells_info = []
        cell_map = {}  # (r, c) -> cell_index

        for idx, cell in enumerate(cell_boxes):
            x1, y1, x2, y2 = cell
            x1_idx = min(range(len(x_positions)), key=lambda i: abs(x_positions[i] - x1))
            x2_idx = min(range(len(x_positions)), key=lambda i: abs(x_positions[i] - x2))
            y1_idx = min(range(len(y_positions)), key=lambda i: abs(y_positions[i] - y1))
            y2_idx = min(range(len(y_positions)), key=lambda i: abs(y_positions[i] - y2))

            col_start = min(x1_idx, x2_idx)
            col_end = max(x1_idx, x2_idx)
            row_start = min(y1_idx, y2_idx)
            row_end = max(y1_idx, y2_idx)

            rowspan = max(1, row_end - row_start)
            colspan = max(1, col_end - col_start)

            cells_info.append({
                "row_start": row_start,
                "col_start": col_start,
                "rowspan": rowspan,
                "colspan": colspan,
                "index": idx
            })

            for r in range(row_start, row_start + rowspan):
                for c in range(col_start, col_start + colspan):
                    cell_map[(r, c)] = idx

        return cells_info, num_rows, num_cols, cell_map

    def generate_html(
            self,
    ) -> str:
        """
        主接口：生成完整的HTML表格字符串
        Returns:
            完整的HTML文档字符串
        """
        if not self.cell_boxes:
            empty_table = "<table></table>"
            return self._wrap_html_document(empty_table)

        # 转换文本框为bbox格式
        converted_text_boxes = [
            self.convert_four_points_to_bbox(four_points)
            for four_points in self.ocr_boxes
        ]

        # 排序单元格
        sorted_cells, _ = self.sort_table_cells_boxes(self.cell_boxes)

        # 匹配文本到单元格
        text_cell_mapping = self.match_text_to_cells(sorted_cells, converted_text_boxes)

        # 生成表格结构
        cells_info, num_rows, num_cols, cell_map = self.generate_table_structure_from_cells(sorted_cells)

        # 构建HTML表格
        table_html = "<table>"
        for r in range(num_rows):
            table_html += "<tr>"
            c = 0
            while c < num_cols:
                key = (r, c)
                if key in cell_map:
                    cell_index = cell_map[key]
                    cell_info = cells_info[cell_index]

                    # 只在左上角位置生成<td>
                    if cell_info["row_start"] == r and cell_info["col_start"] == c:
                        rowspan = cell_info["rowspan"]
                        colspan = cell_info["colspan"]

                        # 获取文本内容和对齐方式
                        # 获取文本内容和对齐方式
                        cell_content = ""
                        text_align = "center"
                        if cell_index in text_cell_mapping:
                            # 先把文本框索引按阅读顺序排好
                            text_indices = text_cell_mapping[cell_index]
                            # 用左上角 y、x 作排序键（行优先）
                            text_indices.sort(
                                key=lambda idx: (
                                    converted_text_boxes[idx][1],  # y1
                                    converted_text_boxes[idx][0]  # x1
                                )
                            )

                            texts = []
                            for text_idx in text_indices:
                                if text_idx < len(self.ocr_results):
                                    texts.append(self.ocr_results[text_idx])
                                    if len(texts) == 1:  # 取第一个文本框算对齐
                                        text_box = converted_text_boxes[text_idx]
                                        text_align = self.calculate_text_alignment(
                                            sorted_cells[cell_index], text_box
                                        )
                            cell_content = " ".join(texts)

                        # 构造属性
                        attrs = []
                        if rowspan > 1:
                            attrs.append(f'rowspan="{rowspan}"')
                        if colspan > 1:
                            attrs.append(f'colspan="{colspan}"')
                        attrs.append(f'style="text-align: {text_align};"')

                        attr_str = " " + " ".join(attrs) if attrs else ""
                        table_html += f"<td{attr_str}>{cell_content}</td>"

                    c += cell_info["colspan"]
                else:
                    # table_html += "<td></td>"
                    c += 1
            table_html += "</tr>"
        table_html += "</table>"

        return self._wrap_html_document(table_html)

    @staticmethod
    def _wrap_html_document(content: str) -> str:
        """包装成完整的HTML文档"""
        return f"""<!DOCTYPE html>
                <html lang="zh">
                <head>
                    <meta charset="UTF-8">
                    <title>表格识别结果</title>
                    <style>
                        table {{
                            border-collapse: collapse;
                            width: 100%;
                        }}
                        th, td {{
                            border: 1px solid #999;
                            padding: 8px;
                            text-align: center;
                        }}
                    </style>
                </head>
                <body>
                {content}
                </body>
                </html>"""


# MIME 类型映射表：用于扩展名 fallback
TEXT_MIME_MAP = {
    'txt': 'text/plain',
    'csv': 'text/csv',
    'log': 'text/plain',
    'json': 'application/json',
    'xml': 'application/xml',
    'html': 'text/html',
    'htm': 'text/html',
    'md': 'text/markdown',
    'yaml': 'application/yaml',
    'yml': 'application/yaml',
}


class FileTypeDetector:
    """
    异步通用文件类型检测器，支持本地文件路径和在线 URL。

    优先使用 `filetype` 基于魔数（magic bytes）检测真实 MIME 类型；
    若失败，则尝试通过扩展名进行 fallback（仅限 TEXT_MIME_MAP 中的类型）。
    """

    # 下载 URL 时读取的字节数（filetype 推荐至少 262 字节，4KB 足够）
    URL_CHUNK_SIZE: int = 4096

    @classmethod
    async def detect(cls, source: Union[str, Path], session: Optional[aiohttp.ClientSession] = None) -> Optional[
        filetype.Type]:
        """
        异步检测文件类型。

        :param source: 本地文件路径（str 或 Path）或在线 URL（str）
        :param session: 可选的 aiohttp.ClientSession（用于连接复用，提升并发性能）
        :return: filetype.Type 对象（若可识别），否则 None
        :raises: ValueError, aiohttp.ClientError, OSError, FileNotFoundError 等
        """
        if isinstance(source, str) and validators.url(source):
            return await cls._detect_from_url(source, session)
        else:
            return await cls._detect_from_file(source)

    @classmethod
    async def _detect_from_file(cls, file_path: Union[str, Path]) -> Optional[filetype.Type]:
        """从本地文件检测类型"""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found or is not a regular file: {path}")
        return filetype.guess(str(path))

    @classmethod
    async def _detect_from_url(cls, url: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[
        filetype.Type]:
        """从 URL 异步下载部分内容并检测类型"""
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                chunk = await resp.content.read(cls.URL_CHUNK_SIZE)
            if not chunk:
                raise ValueError("Received empty content from URL")
            return filetype.guess(chunk)
        finally:
            if close_session:
                await session.close()

    @classmethod
    async def get_info(cls, source: Union[str, Path], session: Optional[aiohttp.ClientSession] = None) -> Optional[
        Dict[str, Any]]:
        """
        获取文件类型信息，包含 MIME、扩展名及常见类别标志。

        :return: 包含以下字段的字典（若可识别）：
            - mime: str
            - extension: str
        若完全无法识别，返回 None。
        """
        kind = await cls.detect(source, session)
        if kind is not None:
            return {
                "mime": kind.mime,
                "extension": kind.extension
            }
        # Fallback: 尝试从扩展名推断（仅限已知文本类型）
        ext = await cls._extract_extension(source)
        if ext and ext in TEXT_MIME_MAP:
            mime = TEXT_MIME_MAP[ext]
            return {
                "mime": mime,
                "extension": ext
            }
        return None  # 完全无法识别

    @staticmethod
    async def _extract_extension(source: Union[str, Path]) -> Optional[str]:
        """安全地从路径或 URL 提取小写扩展名（不含点）"""
        try:
            if isinstance(source, str) and validators.url(source):
                parsed = urlparse(source)
                path = parsed.path
            else:
                path = str(source)
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            return ext if ext else None
        except Exception:
            return None


class PDFExtractor:
    def __init__(
            self,
            ocr_func: Optional[Callable[[numpy.ndarray], Awaitable[OCRResult]]] = None,
            max_workers: Optional[int] = None,
            table_format: str = "text"
    ):
        """
        初始化 PDFExtractor。
        :param ocr_func: 异步 OCR 函数，接收 numpy.ndarray (HWC, RGB)，返回 Awaitable[OCRResult]。
                         若为 None，则图像区域返回空 OCRResult。
        :param max_workers: 并行线程数，默认为 CPU 核心数
        :param table_format: 表格输出格式，"text"（默认）表示转为字符串（tab分隔），"list" 表示保留原始列表并用 str() 转为字符串
        """
        if table_format not in ("text", "list"):
            raise ValueError("table_format must be 'text' or 'list'")
        self.ocr_func = ocr_func
        self.max_workers = max_workers or os.cpu_count()
        self.table_format = table_format
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        atexit.register(self._shutdown_executor)

    def _shutdown_executor(self):
        if self.executor:
            self.executor.shutdown(wait=True)

    def _bytes_to_ndarray(self, img_bytes: bytes) -> Optional[numpy.ndarray]:
        try:
            nparr = numpy.frombuffer(img_bytes, numpy.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
        except Exception:
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                return numpy.array(img)
            except Exception:
                return None

    def _process_single_page(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """同步处理单页（仅提取原始数据，不执行 OCR）"""
        doc = fitz.open(pdf_path)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = doc.load_page(page_num)
                plumber_page = pdf.pages[page_num]
                page_elements = []

                # --- 1. 提取表格 ---
                tables = plumber_page.find_tables()
                for table in tables:
                    bbox = table.bbox
                    table_data = table.extract()
                    page_elements.append({
                        "type": "table",
                        "bbox": bbox,
                        "data": table_data,
                        "position": bbox[1]
                    })
                # --- 2. 提取图像（仅保存图像数组，不 OCR）---
                image_info = page.get_images(full=True)
                for img in image_info:
                    xref = img[0]
                    try:
                        bbox = page.get_image_bbox(img)
                        if bbox.is_empty:
                            continue
                        x0, y0, x1, y1 = bbox
                    except Exception:
                        x0, y0, x1, y1 = 0, 0, 0, 0
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    img_array = self._bytes_to_ndarray(img_bytes)
                    page_elements.append({
                        "type": "image",
                        "bbox": (x0, y0, x1, y1),
                        "data": img_array,
                        "position": y0
                    })
                # --- 3. 提取文本块（排除表格区域）---
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" not in block:
                        continue
                    lines = block["lines"]
                    if not lines:
                        continue
                    text_lines = []
                    all_span_rects = []
                    for line in lines:
                        for span in line["spans"]:
                            text_lines.append(span["text"])
                            all_span_rects.append(fitz.Rect(span["bbox"]))
                    block_rect = fitz.Rect(all_span_rects[0])
                    for r in all_span_rects[1:]:
                        block_rect |= r
                    x0, y0, x1, y1 = block_rect
                    # 跳过与表格重叠的文本
                    skip = False
                    for tab in tables:
                        tab_rect = fitz.Rect(tab.bbox)
                        if abs(block_rect & tab_rect) > 5:
                            skip = True
                            break
                    if skip:
                        continue
                    text = " ".join(text_lines).strip()
                    if not text:
                        continue
                    page_elements.append({
                        "type": "text",
                        "bbox": (x0, y0, x1, y1),
                        "data": text,
                        "position": y0
                    })
                page_elements.sort(key=lambda e: e["position"])
                return {
                    "page": page_num + 1,
                    "elements": page_elements
                }
        finally:
            doc.close()

    async def extract_all_from_pdf(self, pdf_path: str) -> List[Dict[str, Any] | None]:
        """异步并行提取所有页面原始元素（不含 OCR）"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(self.executor, self._process_single_page, pdf_path, i)
            for i in range(total_pages)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        full_content = [None] * total_pages
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing page {i + 1}: {result}")
                full_content[i] = {"page": i + 1, "elements": []}
            else:
                full_content[i] = result
        return full_content

    async def get_page_text(self, page_elements: List[Dict]) -> List[OCRResult]:
        """
        将一页的元素转换为 OCRResult 列表。
        - text/table: 立即构造 OCRResult
        - image: 调用 self.ocr_func（返回 OCRResult）
        """

        async def _process_element(elem: Dict) -> OCRResult:
            elem_type = elem["type"]
            if elem_type == "text":
                return OCRResult(text=elem["data"])
            elif elem_type == "table":
                table_data = elem["data"]
                if self.table_format == "text":
                    table_str = "\n".join(
                        "\t".join(str(cell) if cell is not None else "" for cell in row)
                        for row in table_data
                    )
                else:
                    table_str = str(table_data)
                return OCRResult(text=table_str)
            elif elem_type == "image":
                img_array = elem["data"]
                if self.ocr_func is not None and img_array is not None:
                    try:
                        ocr_result: OCRResult = await self.ocr_func(img_array)
                        return ocr_result
                    except Exception:
                        return OCRResult(text="")
                else:
                    return OCRResult(text="")
            else:
                return OCRResult(text="")

        tasks = [_process_element(elem) for elem in page_elements]
        ocr_results = await asyncio.gather(*tasks, return_exceptions=True)
        final_results: List[OCRResult] = []
        for i, res in enumerate(ocr_results):
            if isinstance(res, Exception):
                print(f"Exception in element {i}: {res}")
                final_results.append(OCRResult(text=""))
            else:
                final_results.append(res)
        return final_results

    async def get_full_text(self, pdf_path: str) -> PdfResult:
        """提取整个 PDF，返回每页的 PdfPageResult 列表"""
        delete_pdf = False
        if validators.url(pdf_path):
            pdf_path = await file_downloader.download(pdf_path)
            delete_pdf = True
        content = await self.extract_all_from_pdf(pdf_path)
        full_results: PdfResult = PdfResult(results=[])
        for page in content:
            page_index = page["page"]
            ocr_data = await self.get_page_text(page["elements"])
            page_text = " ".join(ocr.text for ocr in ocr_data)
            full_results.text += page_text
            full_results.results.append(
                PdfPageResult(
                    page_index=page_index,
                    text=page_text,
                    ocr_data=ocr_data
                )
            )
        if delete_pdf:
            os.remove(pdf_path)
        return full_results
