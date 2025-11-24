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
    WIRED_TABLE = "RT-DETR-L_wired_table_cell_det.onnx"
    WIRELESS_TABLE = "RT-DETR-L_wireless_table_cell_det.onnx"


class ImageModels(Enum):
    L_CNet_x1_0 = "PP-LCNet_x1_0_doc_ori.onnx"
    UVDOC = "UVDoc.onnx"
    DOC_LAYOUT_S = "PP-DocLayout-S.onnx"
    DOC_LAYOUT_M = "PP-DocLayout-M.onnx"
    DOC_LAYOUT_L = "PP-DocLayout-L.onnx"
    DOC_LAYOUT_PLUS_L = "PP-DocLayout_plus-L.onnx"
    DOC_BLOCK_LAYOUT = "PP-DocBlockLayout.onnx"
    TABLE_LAYOUT_PICODET = "PicoDet_layout_1x_table.onnx"
