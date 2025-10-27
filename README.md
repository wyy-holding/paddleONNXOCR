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
from paddleONNXOCR import PredictSystem


async def main():
    """
    推理在线图片
    :return:
    """
    async with PredictSystem() as predictor_system:
        ocr_result =await predictor_system.predict_batch([
            "https://wx2.sinaimg.cn/mw690/005AKOR6ly1hvv14x3e1rj30j615hwfl.jpg",
            cv2.imread("test.png"),
            Image.open("test.png")
        ])
        print(ocr_result)



if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
```
单例调用
```python
from paddleONNXOCR import PredictSystem

async def main():
    predictor_system = PredictSystem()
    await predictor_system.__aenter__()
    return predictor_system
# 外部拿到实力调用推理，参考api/__init__.py中的lifespan
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

### 传递本地模型路径

```python
from paddleONNXOCR import PredictSystem

PredictSystem(det_model_path="testDir/xxx.onnx")
```

### 模型启用

```python
from paddleONNXOCR import PredictSystem

PredictSystem()
# use_angle_cls: 启用文本放方向检测，默认True
# use_deskew: 启用倾斜图像旋转矫正，默认False
# use_uvdoc: 启用图像矫正，默认False
# use_doc_cls: 启用图像方向分类，默认True
```

PS:具体参数请点到每一个方法内，有完整解释。

# api接口服务
提供了dokcer构建启动的方式.
执行bash命令，自动构建docker服务启动.
windows下请使用wsl子系统.
```bash
bash run.sh
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
deskew
modelscope
filetype
pdfplumber
aiofiles
```
