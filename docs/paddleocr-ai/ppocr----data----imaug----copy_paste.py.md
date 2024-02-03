# `.\PaddleOCR\ppocr\data\imaug\copy_paste.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
import copy
import cv2
import random
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from ppocr.data.imaug.iaa_augment import IaaAugment
from ppocr.data.imaug.random_crop_data import is_poly_outside_rect
from tools.infer.utility import get_rotate_crop_image

# 定义 CopyPaste 类
class CopyPaste(object):
    # 初始化方法，设置默认参数
    def __init__(self, objects_paste_ratio=0.2, limit_paste=True, **kwargs):
        self.ext_data_num = 1
        self.objects_paste_ratio = objects_paste_ratio
        self.limit_paste = limit_paste
        augmenter_args = [{'type': 'Resize', 'args': {'size': [0.5, 3]}}]
        # 初始化图像增强器
        self.aug = IaaAugment(augmenter_args)
    # 定义一个类的调用方法，接受一个数据字典作为参数
    def __call__(self, data):
        # 获取多边形的点数
        point_num = data['polys'].shape[1]
        # 获取原始图像
        src_img = data['image']
        # 将原始多边形转换为列表形式
        src_polys = data['polys'].tolist()
        # 获取原始文本
        src_texts = data['texts']
        # 将原始忽略标签转换为列表形式
        src_ignores = data['ignore_tags'].tolist()
        # 获取额外数据中的第一个数据
        ext_data = data['ext_data'][0]
        # 获取额外图像
        ext_image = ext_data['image']
        # 获取额外多边形
        ext_polys = ext_data['polys']
        # 获取额外文本
        ext_texts = ext_data['texts']
        # 获取额外忽略标签
        ext_ignores = ext_data['ignore_tags']

        # 获取不被忽略的索引
        indexs = [i for i in range(len(ext_ignores)) if not ext_ignores[i]]
        # 计算选择的数量，取最大值为1，最小值为对象粘贴比例乘以额外多边形的长度和30的最小值
        select_num = max(
            1, min(int(self.objects_paste_ratio * len(ext_polys)), 30))

        # 随机打乱索引
        random.shuffle(indexs)
        # 选择前select_num个索引
        select_idxs = indexs[:select_num]
        # 根据选择的索引获取对应的多边形和忽略标签
        select_polys = ext_polys[select_idxs]
        select_ignores = ext_ignores[select_idxs]

        # 将原始图像从BGR转换为RGB
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        # 将额外图像从BGR转换为RGB
        ext_image = cv2.cvtColor(ext_image, cv2.COLOR_BGR2RGB)
        # 将原始图像转换为PIL图像，并转换为RGBA模式
        src_img = Image.fromarray(src_img).convert('RGBA')
        # 遍历选择的索引、多边形和忽略标签
        for idx, poly, tag in zip(select_idxs, select_polys, select_ignores):
            # 获取旋转裁剪后的图像
            box_img = get_rotate_crop_image(ext_image, poly)

            # 将裁剪后的图像粘贴到原始图像中
            src_img, box = self.paste_img(src_img, box_img, src_polys)
            if box is not None:
                # 将box转换为列表形式
                box = box.tolist() 
                # 将box的长度扩展到与point_num相同
                for _ in range(len(box), point_num):
                    box.append(box[-1])
                # 将box添加到原始多边形中
                src_polys.append(box)
                # 将额外文本添加到原始文本中
                src_texts.append(ext_texts[idx])
                # 将额外忽略标签添加到原始忽略标签中
                src_ignores.append(tag)
        # 将原始图像从RGB转换为BGR
        src_img = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
        # 获取原始图像的高度和宽度
        h, w = src_img.shape[:2]
        # 将原始多边形转换为数组形式
        src_polys = np.array(src_polys)
        # 对多边形的x坐标进行裁剪，限制在0到图像宽度之间
        src_polys[:, :, 0] = np.clip(src_polys[:, :, 0], 0, w)
        # 对多边形的y坐标进行裁剪，限制在0到图像高度之间
        src_polys[:, :, 1] = np.clip(src_polys[:, :, 1], 0, h)
        # 更新数据字典中的图像、多边形、文本和忽略标签
        data['image'] = src_img
        data['polys'] = src_polys
        data['texts'] = src_texts
        data['ignore_tags'] = np.array(src_ignores)
        # 返回更新后的数据字典
        return data
    # 将 numpy 数组表示的图像转换为 PIL 图像对象，并转换为 RGBA 模式
    box_img_pil = Image.fromarray(box_img).convert('RGBA')
    # 获取源图像和盒子图像的宽度和高度
    src_w, src_h = src_img.size
    box_w, box_h = box_img_pil.size

    # 随机生成旋转角度
    angle = np.random.randint(0, 360)
    # 创建盒子的坐标数组
    box = np.array([[[0, 0], [box_w, 0], [box_w, box_h], [0, box_h]]])
    # 旋转盒子坐标
    box = rotate_bbox(box_img, box, angle)[0]
    # 旋转盒子图像
    box_img_pil = box_img_pil.rotate(angle, expand=1)
    # 更新盒子图像的宽度和高度
    box_w, box_h = box_img_pil.width, box_img_pil.height
    # 如果源图像的宽度减去盒子图像的宽度小于0，或者源图像的高度减去盒子图像的高度小于0，则返回源图像和空盒子
    if src_w - box_w < 0 or src_h - box_h < 0:
        return src_img, None

    # 选择粘贴的坐标
    paste_x, paste_y = self.select_coord(src_polys, box, src_w - box_w,
                                         src_h - box_h)
    # 如果粘贴坐标为 None，则返回源图像和空盒子
    if paste_x is None:
        return src_img, None
    # 更新盒子的坐标
    box[:, 0] += paste_x
    box[:, 1] += paste_y
    # 分离盒子图像的 R、G、B 和 Alpha 通道
    r, g, b, A = box_img_pil.split()
    # 在源图像上粘贴盒子图像
    src_img.paste(box_img_pil, (paste_x, paste_y), mask=A)

    # 返回源图像和盒子
    return src_img, box
    # 选择粘贴坐标，根据限制条件选择合适的粘贴位置
    def select_coord(self, src_polys, box, endx, endy):
        # 如果限制了粘贴次数
        if self.limit_paste:
            # 计算包围框的最小和最大坐标
            xmin, ymin, xmax, ymax = box[:, 0].min(), box[:, 1].min(), box[:, 0].max(), box[:, 1].max()
            # 进行50次尝试
            for _ in range(50):
                # 随机生成粘贴位置的 x 和 y 坐标
                paste_x = random.randint(0, endx)
                paste_y = random.randint(0, endy)
                # 计算新的包围框坐标
                xmin1 = xmin + paste_x
                xmax1 = xmax + paste_x
                ymin1 = ymin + paste_y
                ymax1 = ymax + paste_y

                # 统计在新包围框内的多边形数量
                num_poly_in_rect = 0
                # 遍历源多边形列表
                for poly in src_polys:
                    # 如果多边形不在新包围框外
                    if not is_poly_outside_rect(poly, xmin1, ymin1, xmax1 - xmin1, ymax1 - ymin1):
                        num_poly_in_rect += 1
                        break
                # 如果没有多边形在新包围框内
                if num_poly_in_rect == 0:
                    # 返回找到的合适粘贴位置
                    return paste_x, paste_y
            # 如果50次尝试都没有找到合适位置，则返回空值
            return None, None
        else:
            # 如果没有限制粘贴次数，直接随机生成粘贴位置的 x 和 y 坐标
            paste_x = random.randint(0, endx)
            paste_y = random.randint(0, endy)
            # 返回粘贴位置坐标
            return paste_x, paste_y
# 计算两个多边形的并集的面积
def get_union(pD, pG):
    return Polygon(pD).union(Polygon(pG)).area


# 计算两个多边形的交集与并集的比值
def get_intersection_over_union(pD, pG):
    return get_intersection(pD, pG) / get_union(pD, pG)


# 计算两个多边形的交集的面积
def get_intersection(pD, pG):
    return Polygon(pD).intersection(Polygon(pG)).area


# 旋转图像和文本多边形框
def rotate_bbox(img, text_polys, angle, scale=1):
    """
    from https://github.com/WenmuZhou/DBNet.pytorch/blob/master/data_loader/modules/augment.py
    Args:
        img: np.ndarray
        text_polys: np.ndarray N*4*2
        angle: int
        scale: int

    Returns:

    """
    # 获取图像的宽度和高度
    w = img.shape[1]
    h = img.shape[0]

    # 将角度转换为弧度
    rangle = np.deg2rad(angle)
    # 计算旋转后的图像宽度和高度
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
    # 获取旋转矩阵
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    # ---------------------- rotate box ----------------------
    # 旋转文本多边形框
    rot_text_polys = list()
    for bbox in text_polys:
        point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
        point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
        point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
        point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
        rot_text_polys.append([point1, point2, point3, point4])
    return np.array(rot_text_polys, dtype=np.float32)
```