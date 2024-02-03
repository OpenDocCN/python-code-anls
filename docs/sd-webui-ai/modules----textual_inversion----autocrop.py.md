# `stable-diffusion-webui\modules\textual_inversion\autocrop.py`

```
# 导入所需的库
import cv2
import requests
import os
import numpy as np
from PIL import ImageDraw
from modules import paths_internal
from pkg_resources import parse_version

# 定义颜色常量
GREEN = "#0F0"
BLUE = "#00F"
RED = "#F00"

# 定义函数，智能裁剪图像以突出主题内容
def crop_image(im, settings):
    """ Intelligently crop an image to the subject matter """

    # 根据图像的宽高比进行缩放
    scale_by = 1
    if is_landscape(im.width, im.height):
        scale_by = settings.crop_height / im.height
    elif is_portrait(im.width, im.height):
        scale_by = settings.crop_width / im.width
    elif is_square(im.width, im.height):
        if is_square(settings.crop_width, settings.crop_height):
            scale_by = settings.crop_width / im.width
        elif is_landscape(settings.crop_width, settings.crop_height):
            scale_by = settings.crop_width / im.width
        elif is_portrait(settings.crop_width, settings.crop_height):
            scale_by = settings.crop_height / im.height

    # 根据缩放比例调整图像大小
    im = im.resize((int(im.width * scale_by), int(im.height * scale_by)))
    im_debug = im.copy()

    # 获取焦点
    focus = focal_point(im_debug, settings)

    # 根据焦点计算裁剪坐标
    y_half = int(settings.crop_height / 2)
    x_half = int(settings.crop_width / 2)

    x1 = focus.x - x_half
    if x1 < 0:
        x1 = 0
    elif x1 + settings.crop_width > im.width:
        x1 = im.width - settings.crop_width

    y1 = focus.y - y_half
    if y1 < 0:
        y1 = 0
    elif y1 + settings.crop_height > im.height:
        y1 = im.height - settings.crop_height

    x2 = x1 + settings.crop_width
    y2 = y1 + settings.crop_height

    crop = [x1, y1, x2, y2]

    results = []

    # 将裁剪后的图像添加到结果列表中
    results.append(im.crop(tuple(crop)))
    # 如果设置为需要标注图像
    if settings.annotate_image:
        # 创建一个可以在图像上绘制的对象
        d = ImageDraw.Draw(im_debug)
        # 将裁剪框的坐标转换为列表
        rect = list(crop)
        # 调整矩形的右下角坐标，使其不包含右下角像素
        rect[2] -= 1
        rect[3] -= 1
        # 在调试图像上绘制矩形框
        d.rectangle(rect, outline=GREEN)
        # 将标注后的调试图像添加到结果列表中
        results.append(im_debug)
        # 如果设置为需要在桌面查看图像
        if settings.destop_view_image:
            # 在桌面显示调试图像
            im_debug.show()

    # 返回结果列表
    return results
# 计算图像的焦点位置
def focal_point(im, settings):
    # 如果设置中角点权重大于0，则计算图像的角点
    corner_points = image_corner_points(im, settings) if settings.corner_points_weight > 0 else []
    # 如果设置中熵点权重大于0，则计算图像的熵点
    entropy_points = image_entropy_points(im, settings) if settings.entropy_points_weight > 0 else []
    # 如果设置中人脸点权重大于0，则计算图像的人脸点
    face_points = image_face_points(im, settings) if settings.face_points_weight > 0 else []

    # 初始化感兴趣点列表
    pois = []

    # 初始化权重总和
    weight_pref_total = 0
    # 如果存在角点，则加上角点权重
    if corner_points:
        weight_pref_total += settings.corner_points_weight
    # 如果存在熵点，则加上熵点权重
    if entropy_points:
        weight_pref_total += settings.entropy_points_weight
    # 如果存在人脸点，则加上人脸点权重
    if face_points:
        weight_pref_total += settings.face_points_weight

    # 初始化角点质心
    corner_centroid = None
    # 如果存在角点，则计算角点质心
    if corner_points:
        corner_centroid = centroid(corner_points)
        # 设置角点质心的权重
        corner_centroid.weight = settings.corner_points_weight / weight_pref_total
        # 将角点质心添加到感兴趣点列表中
        pois.append(corner_centroid)

    # 初始化熵点质心
    entropy_centroid = None
    # 如果存在熵点，则计算熵点质心
    if entropy_points:
        entropy_centroid = centroid(entropy_points)
        # 设置熵点质心的权重
        entropy_centroid.weight = settings.entropy_points_weight / weight_pref_total
        # 将熵点质心添加到感兴趣点列表中
        pois.append(entropy_centroid)

    # 初始化人脸点质心
    face_centroid = None
    # 如果存在人脸点，则计算人脸点质心
    if face_points:
        face_centroid = centroid(face_points)
        # 设置人脸点质心的权重
        face_centroid.weight = settings.face_points_weight / weight_pref_total
        # 将人脸点质心添加到感兴趣点列表中
        pois.append(face_centroid)

    # 计算感兴趣点的平均值
    average_point = poi_average(pois, settings)
    # 如果设置了标注图像的选项
    if settings.annotate_image:
        # 创建一个可以在图像上绘制的对象
        d = ImageDraw.Draw(im)
        # 计算最大尺寸
        max_size = min(im.width, im.height) * 0.07
        # 如果存在角落质心
        if corner_centroid is not None:
            # 设置颜色为蓝色
            color = BLUE
            # 计算角落质心的边界框
            box = corner_centroid.bounding(max_size * corner_centroid.weight)
            # 在图像上绘制角落质心的权重值
            d.text((box[0], box[1] - 15), f"Edge: {corner_centroid.weight:.02f}", fill=color)
            # 在图像上绘制椭圆
            d.ellipse(box, outline=color)
            # 如果角落点的数量大于1
            if len(corner_points) > 1:
                # 遍历角落点列表
                for f in corner_points:
                    # 在图像上绘制矩形
                    d.rectangle(f.bounding(4), outline=color)
        # 如果存在熵质心
        if entropy_centroid is not None:
            # 设置颜色为黄色
            color = "#ff0"
            # 计算熵质心的边界框
            box = entropy_centroid.bounding(max_size * entropy_centroid.weight)
            # 在图像上绘制熵质心的权重值
            d.text((box[0], box[1] - 15), f"Entropy: {entropy_centroid.weight:.02f}", fill=color)
            # 在图像上绘制椭圆
            d.ellipse(box, outline=color)
            # 如果熵点的数量大于1
            if len(entropy_points) > 1:
                # 遍历熵点列表
                for f in entropy_points:
                    # 在图像上绘制矩形
                    d.rectangle(f.bounding(4), outline=color)
        # 如果存在人脸质心
        if face_centroid is not None:
            # 设置颜色为红色
            color = RED
            # 计算人脸质心的边界框
            box = face_centroid.bounding(max_size * face_centroid.weight)
            # 在图像上绘制人脸质心的权重值
            d.text((box[0], box[1] - 15), f"Face: {face_centroid.weight:.02f}", fill=color)
            # 在图像上绘制椭圆
            d.ellipse(box, outline=color)
            # 如果人脸点的数量大于1
            if len(face_points) > 1:
                # 遍历人脸点列表
                for f in face_points:
                    # 在图像上绘制矩形
                    d.rectangle(f.bounding(4), outline=color)

        # 在图像上绘制平均点的椭圆
        d.ellipse(average_point.bounding(max_size), outline=GREEN)

    # 返回平均点
    return average_point
# 根据给定的图像和设置检测人脸关键点
def image_face_points(im, settings):
    # 如果设置中包含 DNN 模型路径
    if settings.dnn_model_path is not None:
        # 创建人脸检测器对象，使用指定的 DNN 模型路径和参数
        detector = cv2.FaceDetectorYN.create(
            settings.dnn_model_path,
            "",
            (im.width, im.height),
            0.9,  # 分数阈值
            0.3,  # 非极大值抑制阈值
            5000  # 在非极大值抑制前保留的顶部 k 个
        )
        # 使用人脸检测器检测图像中的人脸
        faces = detector.detect(np.array(im))
        # 初始化结果列表
        results = []
        # 如果检测到人脸
        if faces[1] is not None:
            # 遍历每个检测到的人脸
            for face in faces[1]:
                x = face[0]
                y = face[1]
                w = face[2]
                h = face[3]
                # 将人脸关键点信息添加到结果列表中
                results.append(
                    PointOfInterest(
                        int(x + (w * 0.5)),  # 人脸焦点左右为中心
                        int(y + (h * 0.33)),  # 人脸焦点上下接近头顶
                        size=w,
                        weight=1 / len(faces[1])
                    )
                )
        # 返回检测到的人脸关键点结果列表
        return results
    else:
        # 将图像转换为 NumPy 数组
        np_im = np.array(im)
        # 将彩色图像转换为灰度图像
        gray = cv2.cvtColor(np_im, cv2.COLOR_BGR2GRAY)

        # 定义人脸检测器和对应的最小尺寸比例
        tries = [
            [f'{cv2.data.haarcascades}haarcascade_eye.xml', 0.01],
            [f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml', 0.05],
            [f'{cv2.data.haarcascades}haarcascade_profileface.xml', 0.05],
            [f'{cv2.data.haarcascades}haarcascade_frontalface_alt.xml', 0.05],
            [f'{cv2.data.haarcascades}haarcascade_frontalface_alt2.xml', 0.05],
            [f'{cv2.data.haarcascades}haarcascade_frontalface_alt_tree.xml', 0.05],
            [f'{cv2.data.haarcascades}haarcascade_eye_tree_eyeglasses.xml', 0.05],
            [f'{cv2.data.haarcascades}haarcascade_upperbody.xml', 0.05]
        ]
        # 遍历不同的人脸检测器
        for t in tries:
            # 加载人脸检测器
            classifier = cv2.CascadeClassifier(t[0])
            # 计算最小尺寸
            minsize = int(min(im.width, im.height) * t[1])  # 至少是最小边的 N% 
            try:
                # 使用人脸检测器检测人脸
                faces = classifier.detectMultiScale(gray, scaleFactor=1.1,
                                                    minNeighbors=7, minSize=(minsize, minsize),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
            except Exception:
                continue

            # 如果检测到人脸
            if faces:
                # 将检测到的人脸转换为矩形坐标
                rects = [[f[0], f[1], f[0] + f[2], f[1] + f[3]] for f in faces]
                # 返回人脸中心点的兴趣点列表
                return [PointOfInterest((r[0] + r[2]) // 2, (r[1] + r[3]) // 2, size=abs(r[0] - r[2]),
                                        weight=1 / len(rects)) for r in rects]
    # 如果没有检测到人脸，则返回空列表
    return []
# 计算图像的角点，返回感兴趣点的列表
def image_corner_points(im, settings):
    # 将图像转换为灰度图像
    grayscale = im.convert("L")

    # 在图像底部防止焦点聚集在水印附近的简单尝试
    gd = ImageDraw.Draw(grayscale)
    gd.rectangle([0, im.height * .9, im.width, im.height], fill="#999")

    # 将灰度图像转换为 NumPy 数组
    np_im = np.array(grayscale)

    # 使用 OpenCV 的 goodFeaturesToTrack 函数检测图像中的角点
    points = cv2.goodFeaturesToTrack(
        np_im,
        maxCorners=100,
        qualityLevel=0.04,
        minDistance=min(grayscale.width, grayscale.height) * 0.06,
        useHarrisDetector=False,
    )

    # 如果未检测到角点，则返回空列表
    if points is None:
        return []

    # 创建感兴趣点的列表
    focal_points = []
    for point in points:
        x, y = point.ravel()
        focal_points.append(PointOfInterest(x, y, size=4, weight=1 / len(points)))

    return focal_points


# 计算图像的熵点，返回感兴趣点的列表
def image_entropy_points(im, settings):
    # 判断图像是横向还是纵向
    landscape = im.height < im.width
    portrait = im.height > im.width
    if landscape:
        move_idx = [0, 2]
        move_max = im.size[0]
    elif portrait:
        move_idx = [1, 3]
        move_max = im.size[1]
    else:
        return []

    e_max = 0
    crop_current = [0, 0, settings.crop_width, settings.crop_height]
    crop_best = crop_current
    while crop_current[move_idx[1]] < move_max:
        crop = im.crop(tuple(crop_current))
        e = image_entropy(crop)

        if (e > e_max):
            e_max = e
            crop_best = list(crop_current)

        crop_current[move_idx[0]] += 4
        crop_current[move_idx[1]] += 4

    x_mid = int(crop_best[0] + settings.crop_width / 2)
    y_mid = int(crop_best[1] + settings.crop_height / 2)

    return [PointOfInterest(x_mid, y_mid, size=25, weight=1.0)]


# 计算图像的熵
def image_entropy(im):
    # 计算灰度图像的熵
    band = np.asarray(im.convert("1"), dtype=np.uint8)
    hist, _ = np.histogram(band, bins=range(0, 256))
    hist = hist[hist > 0]
    return -np.log2(hist / hist.sum()).sum()


# 计算感兴趣点的中心
def centroid(pois):
    x = [poi.x for poi in pois]
    y = [poi.y for poi in pois]
    # 返回一个 PointOfInterest 对象，该对象的 x 坐标为 pois 列表中所有 x 坐标的平均值，y 坐标为所有 y 坐标的平均值
    return PointOfInterest(sum(x) / len(pois), sum(y) / len(pois))
# 计算一组兴趣点的加权平均值，并返回平均点
def poi_average(pois, settings):
    # 初始化权重、x坐标和y坐标
    weight = 0.0
    x = 0.0
    y = 0.0
    # 遍历每个兴趣点
    for poi in pois:
        # 累加权重
        weight += poi.weight
        # 根据权重累加加权的x坐标
        x += poi.x * poi.weight
        # 根据权重累加加权的y坐标
        y += poi.y * poi.weight
    # 计算加权平均的x坐标
    avg_x = round(weight and x / weight)
    # 计算加权平均的y坐标
    avg_y = round(weight and y / weight)

    # 返回平均点对象
    return PointOfInterest(avg_x, avg_y)


# 判断是否为横向布局
def is_landscape(w, h):
    return w > h


# 判断是否为纵向布局
def is_portrait(w, h):
    return h > w


# 判断是否为正方形
def is_square(w, h):
    return w == h


# 设置OpenCV模型目录
model_dir_opencv = os.path.join(paths_internal.models_path, 'opencv')
# 根据OpenCV版本选择模型文件路径和下载链接
if parse_version(cv2.__version__) >= parse_version('4.8'):
    model_file_path = os.path.join(model_dir_opencv, 'face_detection_yunet_2023mar.onnx')
    model_url = 'https://github.com/opencv/opencv_zoo/blob/b6e370b10f641879a87890d44e42173077154a05/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?raw=true'
else:
    model_file_path = os.path.join(model_dir_opencv, 'face_detection_yunet.onnx')
    model_url = 'https://github.com/opencv/opencv_zoo/blob/91fb0290f50896f38a0ab1e558b74b16bc009428/models/face_detection_yunet/face_detection_yunet_2022mar.onnx?raw=true'


# 下载并缓存模型文件
def download_and_cache_models():
    if not os.path.exists(model_file_path):
        os.makedirs(model_dir_opencv, exist_ok=True)
        print(f"downloading face detection model from '{model_url}' to '{model_file_path}'")
        response = requests.get(model_url)
        with open(model_file_path, "wb") as f:
            f.write(response.content)
    return model_file_path


# 定义兴趣点类
class PointOfInterest:
    def __init__(self, x, y, weight=1.0, size=10):
        self.x = x
        self.y = y
        self.weight = weight
        self.size = size

    # 计算兴趣点的边界框
    def bounding(self, size):
        return [
            self.x - size // 2,
            self.y - size // 2,
            self.x + size // 2,
            self.y + size // 2
        ]


# 定义设置类
class Settings:
    # 初始化函数，设置默认参数值
    def __init__(self, crop_width=512, crop_height=512, corner_points_weight=0.5, entropy_points_weight=0.5, face_points_weight=0.5, annotate_image=False, dnn_model_path=None):
        # 设置裁剪宽度
        self.crop_width = crop_width
        # 设置裁剪高度
        self.crop_height = crop_height
        # 设置角点权重
        self.corner_points_weight = corner_points_weight
        # 设置熵点权重
        self.entropy_points_weight = entropy_points_weight
        # 设置人脸点权重
        self.face_points_weight = face_points_weight
        # 设置是否标注图像
        self.annotate_image = annotate_image
        # 设置是否显示桌面视图图像
        self.destop_view_image = False
        # 设置深度神经网络模型路径
        self.dnn_model_path = dnn_model_path
```