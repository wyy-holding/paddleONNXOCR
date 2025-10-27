import cv2
import numpy
import validators
from typing import Union
from PIL import Image
import aiohttp
from io import BytesIO

from paddleONNXOCR.utils import UtilsCommon


class ImageLoader:
    """图像下载类"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    }

    @staticmethod
    async def load_image(
            image_path: Union[str, numpy.ndarray, Image.Image]
    ) -> numpy.ndarray:
        match image_path:
            case str() if validators.url(image_path):
                async with aiohttp.ClientSession(headers=ImageLoader.headers) as session:
                    async with session.get(image_path) as response:
                        response.raise_for_status()
                        content_type = response.headers.get('Content-Type', '').lower()
                        if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                            raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")
                        content = await response.read()
                        try:
                            pil_image = Image.open(BytesIO(content))
                            return numpy.array(pil_image.convert("RGB"))
                        except Exception as e:
                            raise ValueError(f"Failed to load image from URL: {str(e)}")
            case str() if await UtilsCommon.is_base64_image(image_path):
                return await UtilsCommon.base64_to_numpy_rgb(image_path)
            case str():
                return cv2.imread(image_path)
            case numpy.ndarray():
                return image_path
            case Image.Image():
                return numpy.array(image_path.convert("RGB"))
            case _:
                raise TypeError(f"Unsupported image type: {type(image_path)}")
