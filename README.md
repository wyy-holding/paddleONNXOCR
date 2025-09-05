# paddle-onnxocr

## 安装

```
pip install --no-cache-dir paddle-onnxocr
```

## 使用示例

单次推理

```python
from paddleONNXOCR import PredictSystem
from paddleONNXOCR.predict.ocr_dataclass import OCRResult


async def main():
    """
    推理在线图片
    :return:
    """
    async with PredictSystem() as predictor_system:
        ocr_result: OCRResult = await predictor_system.predict(
            "https://wx2.sinaimg.cn/mw690/005AKOR6ly1hvv14x3e1rj30j615hwfl.jpg"
        )
        print(ocr_result.text)
        print(ocr_result.json)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
```

批量推理

```python
import cv2
from PIL import Image
from typing import AsyncGenerator
from paddleONNXOCR import PredictSystem
from paddleONNXOCR.predict.ocr_dataclass import OCRResult


async def main():
    """
    推理在线图片
    :return:
    """
    async with PredictSystem() as predictor_system:
        ocr_result: AsyncGenerator[OCRResult, None] = predictor_system.predict_batch([
            "https://wx2.sinaimg.cn/mw690/005AKOR6ly1hvv14x3e1rj30j615hwfl.jpg",
            cv2.imread("test.png"),
            Image.open("test.png")
        ])
        async for result in ocr_result:
            if result is not None:
                print(result.text)
                print(result.json)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
```

默认情况下，会自动从modelscope下载以下模型:
```
PP-LCNet_x0_25_text_line_ori_infer.onnx-->文本行方向检测模型
PP-LCNet_x1_0_doc_ori.onnx->文档方向分类
PP-OCRv5_mobile_det_infer.onnx->文本检测mobile模型
PP-OCRv5_mobile_rec_infer.onnx->文本识别mobile模型
```

### 更改模型

```python
from paddleONNXOCR import PredictSystem
from paddleONNXOCR.models_enum import DetModels, RecModels

# 切换成server版本
PredictSystem(det_model_name=DetModels.SERVER, rec_model_name=RecModels.SERVER)
```
### 依赖项目
```
opencv-python-headless
shapely
pyclipper
onnxruntime
pillow
validators
aiohttp
psutil
deskew
modelscope
```