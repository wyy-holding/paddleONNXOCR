from enum import Enum


class TextLineModels(Enum):
    L_CNET_X0_25 = "PP-LCNet_x0_25_text_line_ori_infer.onnx"
    L_CNET_X1_0 = "PP-LCNet_x1_0_text_line_ori_infer.onnx"


class DetModels(Enum):
    MOBILE = "PP-OCRv5_mobile_det_infer.onnx"
    SERVER = "PP-OCRv5_server_det_infer.onnx"


class RecModels(Enum):
    MOBILE = "PP-OCRv5_mobile_rec_infer.onnx"
    SERVER = "PP-OCRv5_server_rec_infer.onnx"


class TableModels(Enum):
    L_CNet_x1_0 = "PP-LCNet_x1_0_table_cls.onnx"


class ImageModels(Enum):
    L_CNet_x1_0 = "PP-LCNet_x1_0_doc_ori.onnx"
    UVDOC = "UVDoc.onnx"
