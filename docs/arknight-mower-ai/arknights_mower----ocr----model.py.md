# `arknights-mower\arknights_mower\ocr\model.py`

```
# 导入copy模块，用于复制对象
import copy
# 导入traceback模块，用于追踪异常
import traceback

# 导入cv2模块，用于图像处理
import cv2
# 导入numpy模块，用于数值计算
import numpy as np
# 导入PIL模块中的Image类
from PIL import Image

# 导入自定义的日志模块中的logger对象
from ..utils.log import logger
# 导入自定义的config模块中的crnn_model_path和dbnet_model_path
from .config import crnn_model_path, dbnet_model_path
# 导入自定义的crnn模块中的CRNNHandle类
from .crnn import CRNNHandle
# 导入自定义的dbnet模块中的DBNET类
from .dbnet import DBNET
# 导入自定义的utils模块中的fix函数
from .utils import fix

# 定义函数sorted_boxes，用于对文本框进行排序
def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    # 获取文本框的数量
    num_boxes = dt_boxes.shape[0]
    # 对文本框按照从上到下，从左到右的顺序进行排序
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    # 对排序后的文本框进行微调
    for i in range(num_boxes - 1):
        if abs(_boxes[i+1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

# 定义函数get_rotate_crop_image，用于获取旋转裁剪后的图像
def get_rotate_crop_image(img, points):
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    img_crop_width = int(np.linalg.norm(points[0] - points[1]))
    img_crop_height = int(np.linalg.norm(points[0] - points[3]))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height], [0, img_crop_height]])

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img_crop,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

# 定义类OcrHandle，用于处理OCR识别相关操作
class OcrHandle(object):
    # 初始化方法，创建文本检测和文本识别的模型对象
    def __init__(self):
        self.text_handle = DBNET(dbnet_model_path)  # 创建文本检测模型对象
        self.crnn_handle = CRNNHandle(crnn_model_path)  # 创建文本识别模型对象

    # 使用文本识别模型对图像进行识别，并返回识别结果及相关信息
    def crnnRecWithBox(self, im, boxes_list, score_list, is_rgb=False):
        results = []  # 存储识别结果的列表
        boxes_list = sorted_boxes(np.array(boxes_list))  # 对文本框进行排序

        count = 1  # 初始化计数器
        for (box, score) in zip(boxes_list, score_list):  # 遍历文本框和对应的分数

            tmp_box = copy.deepcopy(box)  # 复制文本框信息
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))  # 获取旋转裁剪后的图像

            try:
                if is_rgb:  # 如果是 RGB 图像
                    partImg = Image.fromarray(partImg_array).convert('RGB')  # 转换为 RGB 模式
                    simPred = self.crnn_handle.predict_rbg(partImg)  # 使用文本识别模型进行识别
                else:  # 如果不是 RGB 图像
                    partImg = Image.fromarray(partImg_array).convert('L')  # 转换为灰度模式
                    simPred = self.crnn_handle.predict(partImg)  # 使用文本识别模型进行识别
            except Exception as e:  # 捕获异常
                logger.debug(traceback.format_exc())  # 记录异常信息
                continue  # 继续下一次循环

            if simPred.strip() != '':  # 如果识别结果不为空
                results.append([count, simPred, tmp_box.tolist(), score])  # 将识别结果及相关信息添加到结果列表
                count += 1  # 计数器加一

        return results  # 返回识别结果列表

    # 对输入图像进行文本检测和文本识别，并返回最终的识别结果
    def predict(self, img, is_rgb=False):
        short_size = min(img.shape[:-1])  # 获取图像的最小边长
        short_size = short_size // 32 * 32  # 将最小边长调整为32的倍数
        boxes_list, score_list = self.text_handle.process(img, short_size)  # 使用文本检测模型进行文本检测
        result = self.crnnRecWithBox(img, boxes_list, score_list, is_rgb)  # 使用文本识别模型进行文本识别
        for i in range(len(result)):  # 遍历识别结果列表
            result[i][1] = fix(result[i][1])  # 对识别结果进行修正
        logger.debug(result)  # 记录最终的识别结果
        return result  # 返回最终的识别结果
```