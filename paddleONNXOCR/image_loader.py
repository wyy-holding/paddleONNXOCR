import cv2
import numpy
import validators
from typing import Union
from PIL import Image
import aiohttp
from io import BytesIO
import base64


class ImageLoader:
    """图像下载类"""
    @staticmethod
    async def load_image(
            image_path: Union[str, numpy.ndarray, Image.Image]
    ) -> numpy.ndarray:
        match image_path:
            case str() if validators.url(image_path):
                async with aiohttp.ClientSession() as session:
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
            case str() if image_path.startswith(('data:image/', '/9j/', 'iVBOR')):
                try:
                    if image_path.startswith('data:image/'):
                        image_path = image_path.split(',', 1)[1]
                    image_data = base64.b64decode(image_path)
                    pil_image = Image.open(BytesIO(image_data))
                    return numpy.array(pil_image.convert("RGB"))
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 image: {str(e)}")
            case str():
                return cv2.imread(image_path)
            case numpy.ndarray():
                return image_path
            case Image.Image():
                return numpy.array(image_path.convert("RGB"))
            case _:
                raise TypeError(f"Unsupported image type: {type(image_path)}")