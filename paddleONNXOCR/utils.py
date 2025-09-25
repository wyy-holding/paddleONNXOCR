import aiohttp
import numpy
import cv2
import base64
import imghdr
import math
import onnxruntime
from typing import Sequence, Any, Tuple, List, Dict
from deskew import determine_skew
from modelscope import snapshot_download

from paddleONNXOCR.models_enum import *
from paddleONNXOCR.predict.ocr_dataclass import OCRResult


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
        return [inp.name for inp in session.get_inputs()], [output.name for output in session.get_outputs()], session.get_inputs()[0].shape

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
            boxes: 文本框列表

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

        # 方法1: 分析相邻框的位置关系
        horizontal_score = 0
        vertical_score = 0

        for i in range(len(box_infos)):
            for j in range(i + 1, len(box_infos)):
                box1 = box_infos[i]
                box2 = box_infos[j]

                # 计算两个框的相对位置
                dx = abs(box1['cx'] - box2['cx'])
                dy = abs(box1['cy'] - box2['cy'])

                # 计算Y轴重叠
                y_overlap = min(box1['y_max'], box2['y_max']) - max(box1['y_min'], box2['y_min'])
                y_overlap_ratio = y_overlap / min(box1['height'], box2['height']) if min(box1['height'],
                                                                                         box2['height']) > 0 else 0

                # 计算X轴重叠
                x_overlap = min(box1['x_max'], box2['x_max']) - max(box1['x_min'], box2['x_min'])
                x_overlap_ratio = x_overlap / min(box1['width'], box2['width']) if min(box1['width'],
                                                                                       box2['width']) > 0 else 0

                # 如果Y轴有较大重叠且X轴距离较大，可能是横向布局
                if y_overlap_ratio > 0.5 and dx > dy:
                    horizontal_score += 1

                # 如果X轴有较大重叠且Y轴距离较大，可能是纵向布局
                if x_overlap_ratio > 0.5 and dy > dx:
                    vertical_score += 1

        # 方法2: 分析整体布局的宽高比
        all_x_min = min(box['x_min'] for box in box_infos)
        all_x_max = max(box['x_max'] for box in box_infos)
        all_y_min = min(box['y_min'] for box in box_infos)
        all_y_max = max(box['y_max'] for box in box_infos)

        total_width = all_x_max - all_x_min
        total_height = all_y_max - all_y_min

        # 方法3: 分析文本框的平均宽高比
        avg_aspect_ratio = numpy.mean([box['width'] / box['height'] if box['height'] > 0 else 1
                                       for box in box_infos])

        # 综合判断
        # 如果大部分文本框是横向的（宽>高），且整体布局也是横向的
        if avg_aspect_ratio > 1.5 and horizontal_score > vertical_score:
            return "horizontal"

        # 如果文本框较窄（高>宽），或者垂直排列特征明显
        if avg_aspect_ratio < 0.7 or vertical_score > horizontal_score * 1.5:
            return "vertical"

        # 默认返回横向
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
        """
        将文本框按行分组

        Args:
            box_infos: 包含框信息的字典列表

        Returns:
            按行分组的框信息列表
        """
        if not box_infos:
            return []

        # 按y_min排序
        box_infos.sort(key=lambda x: x['y_min'])

        rows = []
        current_row = [box_infos[0]]
        current_row_y_min = box_infos[0]['y_min']
        current_row_y_max = box_infos[0]['y_max']

        for box_info in box_infos[1:]:
            # 计算与当前行的重叠度
            overlap_threshold = 0.5  # 重叠阈值

            # 计算Y方向的重叠
            y_overlap = min(current_row_y_max, box_info['y_max']) - max(current_row_y_min, box_info['y_min'])
            box_height = box_info['height']

            # 如果重叠度足够高，认为是同一行
            if y_overlap > box_height * overlap_threshold:
                current_row.append(box_info)
                # 更新当前行的边界
                current_row_y_min = min(current_row_y_min, box_info['y_min'])
                current_row_y_max = max(current_row_y_max, box_info['y_max'])
            else:
                # 开始新的一行
                rows.append(current_row)
                current_row = [box_info]
                current_row_y_min = box_info['y_min']
                current_row_y_max = box_info['y_max']

        # 添加最后一行
        if current_row:
            rows.append(current_row)

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
    def is_base64_image(
            content: str
    ) -> bool | bytes:
        """
        判断一个字符串是否是 Base64 编码的图片内容
        """
        try:
            decoded_data = base64.b64decode(content, validate=True)
        except Exception:
            return False
        try:
            image_type = imghdr.what(None, h=decoded_data)
            if image_type is not None:
                return decoded_data
        except Exception:
            ...
        return False

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
