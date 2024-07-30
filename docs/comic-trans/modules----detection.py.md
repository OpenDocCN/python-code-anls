# `.\comic-translate\modules\detection.py`

```py
from ultralytics import YOLO
import largestinteriorrectangle as lir
from .utils.textblock import TextBlock
import numpy as np 
import cv2

class TextBlockDetector:
    def __init__(self, bubble_model_path: str, text_model_path: str, device: str):
        # 初始化文本块检测器，加载泡泡检测和文本分割模型
        self.bubble_detection = YOLO(bubble_model_path)
        self.text_segmentation = YOLO(text_model_path)
        self.device = device

    def detect(self, img):
        # 检测函数，输入图像img，返回文本块列表
        
        # 获取图像的高度和宽度
        h, w, _ = img.shape
        # 确定用于模型推理的图像尺寸
        size = (h, w) if h >= w * 5 else 1024
        
        # 使用泡泡检测模型进行推理
        bble_detec_result = self.bubble_detection(img, device=self.device, imgsz=size, conf=0.1, verbose=False)[0]
        # 使用文本分割模型进行推理
        txt_seg_result = self.text_segmentation(img, device=self.device, imgsz=size, conf=0.1, verbose=False)[0]

        # 合并两个模型的结果
        combined = combine_results(bble_detec_result, txt_seg_result)

        # 创建文本块对象列表
        blk_list = [TextBlock(txt_bbox, txt_seg_points, bble_bbox, txt_class)
                    for txt_bbox, bble_bbox, txt_seg_points, txt_class in combined]
        
        return blk_list

def calculate_iou(rect1, rect2) -> float:
    """
    Calculate the Intersection over Union (IoU) of two rectangles.
    
    Parameters:
    rect1, rect2: The coordinates of the rectangles in the format
    [x1, y1, x2, y2], where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate.
    
    Returns:
    iou: the Intersection over Union (IoU) metric as a float.
    """
    # 计算两个矩形的交集区域
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    
    union_area = rect1_area + rect2_area - intersection_area
    
    iou = intersection_area / union_area if union_area != 0 else 0
    
    return iou

def do_rectangles_overlap(rect1, rect2, iou_threshold: float = 0.2) -> bool:
    """
    Determines whether two rectangles refer to the same object based on an IoU threshold.
    
    Parameters:
    rect1, rect2: as described in the calculate_iou function.
    iou_threshold: float value representing the threshold above which the rectangles are
    considered to be referring to the same object.
    
    Returns:
    overlap: a boolean indicating whether the two rectangles refer to the same object.
    """
    # 计算两个矩形的IoU
    iou = calculate_iou(rect1, rect2)
    # 判断是否重叠
    overlap = iou >= iou_threshold
    return overlap

def does_rectangle_fit(bigger_rect, smaller_rect):
    x1, y1, x2, y2 = bigger_rect
    px1, py1, px2, py2 = smaller_rect
    
    # 确保坐标顺序正确
    # 第一个矩形
    left1, top1, right1, bottom1 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    # 第二个矩形
    left2, top2, right2, bottom2 = min(px1, px2), min(py1, py2), max(px1, px2), max(py1, py2)
    
    # 检查第二个矩形是否能够完全包含在第一个矩形内部
    fits_horizontally = left1 <= left2 and right1 >= right2
    # 检查第一个矩形是否垂直适配于第二个矩形
    fits_vertically = top1 <= top2 and bottom1 >= bottom2
    
    # 返回两个矩形是否水平和垂直适配的布尔结果
    return fits_horizontally and fits_vertically
# 从YOLO对象中获取气泡检测结果的边界框坐标，并转换为整数类型的numpy数组
bubble_bounding_boxes = np.array(bubble_detec_results.boxes.xyxy.cpu(), dtype="int")

# 从YOLO对象中获取文本分割结果的边界框坐标，并转换为整数类型的numpy数组
text_bounding_boxes = np.array(text_seg_results.boxes.xyxy.cpu(), dtype="int")

# 初始化空列表，用于存储文本分割结果的掩模轮廓点
segment_points = []

# 如果文本分割结果中存在掩模信息，则将每个掩模的轮廓点转换为整数类型，并存入segment_points列表
if text_seg_results.masks is not None:
    segment_points = list(map(lambda a: a.astype("int"), text_seg_results.masks.xy))

# 初始化空列表，用于存储最终匹配结果
raw_results = []

# 初始化一个与text_bounding_boxes长度相同的布尔列表，表示每个文本边界框是否已经匹配到气泡边界框
text_matched = [False] * len(text_bounding_boxes)

# 如果存在文本分割掩模信息，则开始匹配文本边界框和气泡边界框
if segment_points:
    for txt_idx, txt_box in enumerate(text_bounding_boxes):
        for bble_box in bubble_bounding_boxes:
            # 如果文本边界框可以完全包含在气泡边界框内，则将匹配结果存入raw_results，并标记文本已匹配
            if does_rectangle_fit(bble_box, txt_box):
                raw_results.append((txt_box, bble_box, segment_points[txt_idx], 'text_bubble'))
                text_matched[txt_idx] = True
                break
            # 如果文本边界框与气泡边界框存在重叠区域，则将匹配结果存入raw_results，并标记文本已匹配
            elif do_rectangles_overlap(bble_box, txt_box):
                raw_results.append((txt_box, bble_box, segment_points[txt_idx], 'text_free'))
                text_matched[txt_idx] = True
                break

# 如果文本边界框未匹配到任何气泡边界框，则根据需要将文本边界框信息存入raw_results
# if not text_matched[txt_idx]:
#     raw_results.append((txt_box, None, segment_points[txt_idx], 'text_free'))

# 返回最终的匹配结果列表
return raw_results



# 调整输入图像的对比度和亮度
def adjust_contrast_brightness(img: np.ndarray, contrast: float = 1.0, brightness: int = 0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    # 根据对比度参数计算需要调整的亮度值
    brightness += int(round(255 * (1 - contrast) / 2))
    # 使用cv2.addWeighted函数对图像进行对比度和亮度调整
    return cv2.addWeighted(img, contrast, img, 0, brightness)



# 确保输入图像为灰度图像，如果不是则进行颜色空间转换
def ensure_gray(img: np.ndarray):
    # 如果图像通道数大于2，则将图像转换为灰度图像并返回副本
    if len(img.shape) > 2:
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # 否则直接返回输入图像的副本
    return img.copy()



# 创建气泡掩模图像
def make_bubble_mask(frame: np.ndarray):
    # 复制输入帧图像
    image = frame.copy()

    # 对图像应用高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 使用Canny边缘检测算法检测图像边缘
    edges = cv2.Canny(blurred, 50, 150)

    # 在图像中找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 创建与原始图像大小相同的黑色图像
    stage_1 = cv2.drawContours(np.zeros_like(image), contours, -1, (255, 255, 255), thickness=2)
    stage_1 = cv2.bitwise_not(stage_1)
    stage_1 = cv2.cvtColor(stage_1, cv2.COLOR_BGR2GRAY)

    # 应用阈值处理，获取二值化图像
    _, binary_image = cv2.threshold(stage_1, 200, 255, cv2.THRESH_BINARY)

    # 在二值化图像中找到连通组件
    num_labels, labels = cv2.connectedComponents(binary_image)
    largest_island_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    mask = np.zeros_like(image)
    mask[labels == largest_island_label] = 255

    # 对掩模图像应用形态学操作，去除黑色斑点
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # 返回处理后的气泡掩模图像
    # 此处应该继续添加其他处理的代码...
    # 使用 cv2 库的函数创建一个椭圆形的结构元素，用于形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # 对输入的掩模图像进行开运算，通过椭圆形结构元素去除小的噪声区域
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 调用自定义函数 adjust_contrast_brightness 对掩模图像进行对比度和亮度的调整，并返回处理后的结果
    return adjust_contrast_brightness(mask, 100)
# 对输入的帧掩码图像进行灰度化处理，确保是灰度图像
gray = ensure_gray(frame_mask)

# 对灰度图像进行阈值化处理，将灰度值大于200的像素设为255（白色），其余设为0（黑色）
ret, thresh = cv2.threshold(gray, 200, 255, 0)

# 寻找图像中的轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 找到面积最大的轮廓
largest_contour = max(contours, key=cv2.contourArea)

# 从最大轮廓中提取多边形的顶点坐标
polygon = np.array([largest_contour[:, 0, :]])

# 使用某个函数 lir 来计算多边形的边界框
rect = lir.lir(polygon)

# 获取边界框的左上角和右下角坐标
x1, y1 = lir.pt1(rect)
x2, y2 = lir.pt2(rect)

# 返回边界框的左上角 (x1, y1) 和右下角 (x2, y2) 坐标
return x1, y1, x2, y2
```