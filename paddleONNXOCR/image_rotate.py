import math
import cv2
import numpy
from deskew import determine_skew
from paddleONNXOCR.predict.predict_doc_cls import DocumentOrientationDetector


class ImageRotate:
    """
    文本图像方向检测后旋转与deskew检测
    """

    @staticmethod
    async def doc_cls_rotate_image(
            img_array: numpy.ndarray,
            doc_cls_model: DocumentOrientationDetector
    ) -> numpy.ndarray:
        doc_cls_result = await doc_cls_model.predict(img_array)
        rotation_angle = doc_cls_result["class_name"]
        match rotation_angle:
            case "90":
                img_array = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
            case "180":
                img_array = cv2.rotate(img_array, cv2.ROTATE_180)
            case "270":
                img_array = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
        return img_array

    @staticmethod
    async def deskew_rotate_image(
            img_array: numpy.ndarray
    ) -> numpy.ndarray:
        grayscale = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)
        if angle is None:
            return img_array
        old_width, old_height = img_array.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(numpy.sin(angle_radian) * old_height) + abs(numpy.cos(angle_radian) * old_width)
        height = abs(numpy.sin(angle_radian) * old_width) + abs(numpy.cos(angle_radian) * old_height)
        image_center = tuple(numpy.array(img_array.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(img_array, rot_mat, (int(round(height)), int(round(width))), borderValue=(0, 0, 0))
