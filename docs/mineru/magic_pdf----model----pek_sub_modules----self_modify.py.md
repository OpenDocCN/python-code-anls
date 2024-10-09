# `.\MinerU\magic_pdf\model\pek_sub_modules\self_modify.py`

```
# 导入所需的库
import time  # 时间相关功能
import copy  # 深拷贝相关功能
import base64  # 编码和解码Base64数据
import cv2  # OpenCV库，用于图像处理
import numpy as np  # 数组和矩阵运算
from io import BytesIO  # 字节流处理
from PIL import Image  # 图像处理库

# 导入PaddleOCR及其相关工具
from paddleocr import PaddleOCR  # OCR工具
from paddleocr.ppocr.utils.logging import get_logger  # 日志记录工具
from paddleocr.ppocr.utils.utility import check_and_read, alpha_to_color, binarize_img  # 工具函数
from paddleocr.tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop  # 识别相关工具

# 导入自定义库
from magic_pdf.libs.boxbase import __is_overlaps_y_exceeds_threshold  # 重叠检查工具
from magic_pdf.pre_proc.ocr_dict_merge import merge_spans_to_line  # 合并字典工具

# 获取日志记录器实例
logger = get_logger()


def img_decode(content: bytes):
    # 将字节内容转换为NumPy数组并解码为图像
    np_arr = np.frombuffer(content, dtype=np.uint8)  # 从字节内容创建NumPy数组
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)  # 解码图像


def check_img(img):
    # 检查并处理输入图像
    if isinstance(img, bytes):  # 如果输入是字节
        img = img_decode(img)  # 解码为图像
    if isinstance(img, str):  # 如果输入是文件名
        image_file = img  # 保存文件名
        img, flag_gif, flag_pdf = check_and_read(image_file)  # 检查并读取图像
        if not flag_gif and not flag_pdf:  # 如果不是GIF或PDF
            with open(image_file, 'rb') as f:  # 以二进制方式打开文件
                img_str = f.read()  # 读取文件内容
                img = img_decode(img_str)  # 解码为图像
            if img is None:  # 如果解码失败
                try:  # 尝试其他方法
                    buf = BytesIO()  # 创建字节流
                    image = BytesIO(img_str)  # 封装文件内容为字节流
                    im = Image.open(image)  # 使用PIL打开图像
                    rgb = im.convert('RGB')  # 转换为RGB格式
                    rgb.save(buf, 'jpeg')  # 保存为JPEG格式到字节流
                    buf.seek(0)  # 重置字节流位置
                    image_bytes = buf.read()  # 读取字节流内容
                    data_base64 = str(base64.b64encode(image_bytes), encoding="utf-8")  # 编码为Base64
                    image_decode = base64.b64decode(data_base64)  # 解码Base64
                    img_array = np.frombuffer(image_decode, np.uint8)  # 转换为NumPy数组
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 解码为图像
                except:  # 捕获异常
                    logger.error("error in loading image:{}".format(image_file))  # 记录错误日志
                    return None  # 返回None表示失败
        if img is None:  # 如果图像仍然为空
            logger.error("error in loading image:{}".format(image_file))  # 记录错误日志
            return None  # 返回None表示失败
    if isinstance(img, np.ndarray) and len(img.shape) == 2:  # 如果是灰度图像
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转换为BGR格式

    return img  # 返回处理后的图像


def sorted_boxes(dt_boxes):
    """
    将文本框按照从上到下、从左到右的顺序排序
    args:
        dt_boxes(array): 检测到的文本框，形状为[4, 2]
    return:
        排序后的文本框(array)，形状为[4, 2]
    """
    num_boxes = dt_boxes.shape[0]  # 获取文本框数量
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))  # 按y坐标和x坐标排序
    _boxes = list(sorted_boxes)  # 复制排序结果

    for i in range(num_boxes - 1):  # 遍历每个文本框
        for j in range(i, -1, -1):  # 反向遍历已排序的文本框
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \  # 如果y坐标接近
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):  # 如果x坐标较小
                tmp = _boxes[j]  # 交换文本框
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break  # 如果不满足条件，停止内层循环
    return _boxes  # 返回排序后的文本框


def bbox_to_points(bbox):
    """ 将bbox格式转换为四个顶点的数组 """
    x0, y0, x1, y1 = bbox  # 解包边界框坐标
    # 返回一个包含四个顶点坐标的 NumPy 数组，表示一个矩形
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype('float32') 
# 定义一个将四个顶点数组转换为边界框格式的函数
def points_to_bbox(points):
    """ 将四个顶点的数组转换为bbox格式 """
    # 从顶点数组中提取第一个点的 x 和 y 坐标
    x0, y0 = points[0]
    # 从第二个点中提取 x 坐标，y 坐标用下划线忽略
    x1, _ = points[1]
    # 从第三个点中提取 y 坐标，x 坐标用下划线忽略
    _, y1 = points[2]
    # 返回边界框的四个坐标
    return [x0, y0, x1, y1]


# 定义一个合并重叠区间的函数
def merge_intervals(intervals):
    # 根据区间的起始值对区间进行排序
    intervals.sort(key=lambda x: x[0])

    # 初始化合并后的区间列表
    merged = []
    # 遍历每个区间
    for interval in intervals:
        # 如果合并列表为空或当前区间与上一个区间不重叠，则直接添加当前区间
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # 否则，存在重叠，将当前区间与上一个区间合并
            merged[-1][1] = max(merged[-1][1], interval[1])

    # 返回合并后的区间列表
    return merged


# 定义一个根据掩码移除区间的函数
def remove_intervals(original, masks):
    # 合并所有掩码区间
    merged_masks = merge_intervals(masks)

    # 初始化结果列表
    result = []
    # 提取原始区间的起始和结束值
    original_start, original_end = original

    # 遍历每个合并后的掩码区间
    for mask in merged_masks:
        mask_start, mask_end = mask

        # 如果掩码的起始值在原始范围之后，则忽略该掩码
        if mask_start > original_end:
            continue

        # 如果掩码的结束值在原始范围之前，则忽略该掩码
        if mask_end < original_start:
            continue

        # 从原始范围中移除掩码部分
        if original_start < mask_start:
            result.append([original_start, mask_start - 1])

        # 更新原始区间的起始值为掩码结束值之后的部分
        original_start = max(mask_end + 1, original_start)

    # 如果原始范围中还有剩余部分，则添加到结果列表中
    if original_start <= original_end:
        result.append([original_start, original_end])

    # 返回最终结果
    return result


# 定义一个更新检测框的函数
def update_det_boxes(dt_boxes, mfd_res):
    # 初始化新的检测框列表
    new_dt_boxes = []
    # 遍历每个检测框
    for text_box in dt_boxes:
        # 将检测框的顶点转换为边界框格式
        text_bbox = points_to_bbox(text_box)
        # 初始化掩码列表
        masks_list = []
        # 遍历每个合并后的结果
        for mf_box in mfd_res:
            mf_bbox = mf_box['bbox']
            # 检查当前边界框与合并后的边界框是否重叠，若重叠则添加到掩码列表
            if __is_overlaps_y_exceeds_threshold(text_bbox, mf_bbox):
                masks_list.append([mf_bbox[0], mf_bbox[2]])
        # 定义文本框的 x 坐标范围
        text_x_range = [text_bbox[0], text_bbox[2]]
        # 根据掩码移除文本框的范围
        text_remove_mask_range = remove_intervals(text_x_range, masks_list)
        # 初始化临时检测框列表
        temp_dt_box = []
        # 遍历移除掩码后的范围
        for text_remove_mask in text_remove_mask_range:
            # 将移除后的范围转换为顶点并添加到临时检测框列表
            temp_dt_box.append(bbox_to_points([text_remove_mask[0], text_bbox[1], text_remove_mask[1], text_bbox[3]]))
        # 如果临时检测框列表不为空，则将其扩展到新的检测框列表中
        if len(temp_dt_box) > 0:
            new_dt_boxes.extend(temp_dt_box)
    # 返回更新后的检测框列表
    return new_dt_boxes


# 定义一个合并重叠区间的函数
def merge_overlapping_spans(spans):
    """
    Merges overlapping spans on the same line.

    :param spans: A list of span coordinates [(x1, y1, x2, y2), ...]
    :return: A list of merged spans
    """
    # 如果输入的区间列表为空，返回一个空列表
    if not spans:
        return []

    # 根据起始 x 坐标对区间进行排序
    spans.sort(key=lambda x: x[0])

    # 初始化合并后的区间列表
    merged = []
    # 遍历所有的 span
    for span in spans:
        # 解包 span 的坐标，分别赋值给 x1, y1, x2, y2
        x1, y1, x2, y2 = span
        # 如果合并后的列表为空或最后一个 span 的右侧不与当前 span 重叠，直接添加当前 span
        if not merged or merged[-1][2] < x1:
            merged.append(span)
        else:
            # 如果存在水平重叠，合并当前 span 与上一个 span
            last_span = merged.pop()
            # 更新合并后的 span 的左上角坐标为较小的 (x1, y1)，右下角坐标为较大的 (x2, y2)
            x1 = min(last_span[0], x1)
            y1 = min(last_span[1], y1)
            x2 = max(last_span[2], x2)
            y2 = max(last_span[3], y2)
            # 将合并后的 span 重新添加回列表
            merged.append((x1, y1, x2, y2))

    # 返回合并后的 span 列表
    return merged
# 合并检测到的边界框
def merge_det_boxes(dt_boxes):
    """
    合并检测框。

    该函数接受一个检测到的边界框列表，每个边界框由四个角点表示。
    目的是将这些边界框合并为更大的文本区域。

    参数：
    dt_boxes (list): 包含多个文本检测框的列表，每个框由四个角点定义。

    返回：
    list: 一个包含合并文本区域的列表，每个区域由四个角点表示。
    """
    # 将检测框转换为包含边界框和类型的字典格式
    dt_boxes_dict_list = []
    for text_box in dt_boxes:
        # 将角点转换为边界框
        text_bbox = points_to_bbox(text_box)
        # 创建字典以存储边界框和类型
        text_box_dict = {
            'bbox': text_bbox,
            'type': 'text',
        }
        # 将字典添加到列表中
        dt_boxes_dict_list.append(text_box_dict)

    # 将相邻的文本区域合并为行
    lines = merge_spans_to_line(dt_boxes_dict_list)

    # 初始化一个新列表以存储合并后的文本区域
    new_dt_boxes = []
    for line in lines:
        line_bbox_list = []
        for span in line:
            # 将每个跨度的边界框添加到列表中
            line_bbox_list.append(span['bbox'])

        # 合并同一行内重叠的文本区域
        merged_spans = merge_overlapping_spans(line_bbox_list)

        # 将合并后的文本区域转换回点格式并添加到新检测框列表
        for span in merged_spans:
            new_dt_boxes.append(bbox_to_points(span))

    # 返回合并后的文本区域列表
    return new_dt_boxes


# 修改后的PaddleOCR类
class ModifiedPaddleOCR(PaddleOCR):
    # 实现可调用对象，处理图像并返回检测和识别结果
        def __call__(self, img, cls=True, mfd_res=None):
            # 初始化时间字典以记录各个阶段耗时
            time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
    
            # 检查输入图像是否有效
            if img is None:
                # 记录调试信息，表示未提供有效图像
                logger.debug("no valid image provided")
                # 返回空结果和时间字典
                return None, None, time_dict
    
            # 记录开始时间
            start = time.time()
            # 复制原始图像以进行处理
            ori_im = img.copy()
            # 使用文本检测器处理图像，获得检测框和耗时
            dt_boxes, elapse = self.text_detector(img)
            # 更新时间字典中的检测时间
            time_dict['det'] = elapse
    
            # 检查检测结果是否为空
            if dt_boxes is None:
                # 记录调试信息，表示未找到检测框
                logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
                # 记录结束时间
                end = time.time()
                # 更新时间字典中的总耗时
                time_dict['all'] = end - start
                # 返回空结果和时间字典
                return None, None, time_dict
            else:
                # 记录调试信息，输出检测框数量和耗时
                logger.debug("dt_boxes num : {}, elapsed : {}".format(
                    len(dt_boxes), elapse))
            # 初始化裁剪图像列表
            img_crop_list = []
    
            # 对检测框进行排序
            dt_boxes = sorted_boxes(dt_boxes)
    
            # 合并重叠的检测框
            dt_boxes = merge_det_boxes(dt_boxes)
    
            # 如果提供了额外的检测结果
            if mfd_res:
                # 记录合并检测框的开始时间
                bef = time.time()
                # 更新检测框
                dt_boxes = update_det_boxes(dt_boxes, mfd_res)
                # 记录合并检测框的结束时间
                aft = time.time()
                # 记录调试信息，输出新检测框数量和耗时
                logger.debug("split text box by formula, new dt_boxes num : {}, elapsed : {}".format(
                    len(dt_boxes), aft - bef))
    
            # 遍历每个检测框
            for bno in range(len(dt_boxes)):
                # 深拷贝当前检测框
                tmp_box = copy.deepcopy(dt_boxes[bno])
                # 根据检测框类型裁剪图像
                if self.args.det_box_type == "quad":
                    img_crop = get_rotate_crop_image(ori_im, tmp_box)
                else:
                    img_crop = get_minarea_rect_crop(ori_im, tmp_box)
                # 将裁剪后的图像添加到列表中
                img_crop_list.append(img_crop)
            # 如果使用角度分类器且类标志为真
            if self.use_angle_cls and cls:
                # 使用文本分类器处理裁剪图像，获得分类结果和耗时
                img_crop_list, angle_list, elapse = self.text_classifier(
                    img_crop_list)
                # 更新时间字典中的分类时间
                time_dict['cls'] = elapse
                # 记录调试信息，输出分类结果数量和耗时
                logger.debug("cls num  : {}, elapsed : {}".format(
                    len(img_crop_list), elapse))
    
            # 使用文本识别器处理裁剪图像，获得识别结果和耗时
            rec_res, elapse = self.text_recognizer(img_crop_list)
            # 更新时间字典中的识别时间
            time_dict['rec'] = elapse
            # 记录调试信息，输出识别结果数量和耗时
            logger.debug("rec_res num  : {}, elapsed : {}".format(
                len(rec_res), elapse))
            # 如果需要保存裁剪结果
            if self.args.save_crop_res:
                # 绘制并保存裁剪结果
                self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                       rec_res)
            # 初始化过滤后的框和结果列表
            filter_boxes, filter_rec_res = [], []
            # 遍历检测框和识别结果
            for box, rec_result in zip(dt_boxes, rec_res):
                # 解包识别结果，获取文本和分数
                text, score = rec_result
                # 根据分数筛选有效检测框和识别结果
                if score >= self.drop_score:
                    filter_boxes.append(box)
                    filter_rec_res.append(rec_result)
            # 记录结束时间
            end = time.time()
            # 更新时间字典中的总耗时
            time_dict['all'] = end - start
            # 返回过滤后的框、识别结果和时间字典
            return filter_boxes, filter_rec_res, time_dict
```