# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\data_loader\modules\make_shrink_map.py`

```
import numpy as np
import cv2

# 对多边形进行缩放的函数，返回缩放后的多边形
def shrink_polygon_py(polygon, shrink_ratio):
    # 计算多边形的中心点坐标
    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()
    # 根据缩放比例对多边形进行缩放
    polygon[:, 0] = cx + (polygon[:, 0] - cx) * shrink_ratio
    polygon[:, 1] = cy + (polygon[:, 1] - cy) * shrink_ratio
    return polygon

# 使用 pyclipper 库对多边形进行缩放的函数
def shrink_polygon_pyclipper(polygon, shrink_ratio):
    from shapely.geometry import Polygon
    import pyclipper
    # 创建多边形对象
    polygon_shape = Polygon(polygon)
    # 计算缩放距离
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    # 创建 PyclipperOffset 对象
    padding = pyclipper.PyclipperOffset()
    # 添加路径
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    # 执行缩放操作
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        shrinked = np.array(shrinked)
    else:
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
    return shrinked

# 创建 MakeShrinkMap 类
class MakeShrinkMap():
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''

    # 初始化函数
    def __init__(self,
                 min_text_size=8,
                 shrink_ratio=0.4,
                 shrink_type='pyclipper'):
        # 缩放函数字典
        shrink_func_dict = {
            'py': shrink_polygon_py,
            'pyclipper': shrink_polygon_pyclipper
        }
        # 根据缩放类型选择对应的缩放函数
        self.shrink_func = shrink_func_dict[shrink_type]
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        # 从输入数据中获取图像、文本框和忽略标签信息
        image = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        # 获取图像的高度和宽度
        h, w = image.shape[:2]
        # 对文本框和忽略标签进行验证，确保在图像范围内
        text_polys, ignore_tags = self.validate_polygons(text_polys,
                                                         ignore_tags, h, w)
        # 创建一个全零矩阵作为 ground truth
        gt = np.zeros((h, w), dtype=np.float32)
        # 创建一个全一矩阵作为 mask
        mask = np.ones((h, w), dtype=np.float32)
        # 遍历每个文本框
        for i in range(len(text_polys)):
            polygon = text_polys[i]
            # 计算文本框的高度和宽度
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            # 如果文本框被标记为忽略或者高度、宽度小于最小文本尺寸
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                # 在 mask 上填充多边形区域为 0
                cv2.fillPoly(mask,
                             polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                # 对文本框进行缩小操作
                shrinked = self.shrink_func(polygon, self.shrink_ratio)
                # 如果缩小后的文本框为空
                if shrinked.size == 0:
                    # 在 mask 上填充多边形区域为 0
                    cv2.fillPoly(mask,
                                 polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                # 在 ground truth 上填充缩小后的文本框区域为 1
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)

        # 将缩小后的 ground truth 和 mask 存入输入数据中
        data['shrink_map'] = gt
        data['shrink_mask'] = mask
        return data
    # 验证多边形的有效性，根据指定的条件对多边形进行处理
    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        # 如果多边形列表为空，则直接返回
        if len(polygons) == 0:
            return polygons, ignore_tags
        # 断言多边形列表和忽略标签列表的长度相等
        assert len(polygons) == len(ignore_tags)
        # 遍历每个多边形
        for polygon in polygons:
            # 将多边形的 x 坐标限制在 [0, w-1] 范围内
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            # 将多边形的 y 坐标限制在 [0, h-1] 范围内
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        # 遍历每个多边形
        for i in range(len(polygons)):
            # 计算多边形的面积
            area = self.polygon_area(polygons[i])
            # 如果多边形的面积绝对值小于1，则将对应的忽略标签设为True
            if abs(area) < 1:
                ignore_tags[i] = True
            # 如果多边形的面积大于0，则翻转多边形的顺序
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        # 返回处理后的多边形列表和忽略标签列表
        return polygons, ignore_tags

    # 计算多边形的面积
    def polygon_area(self, polygon):
        # 使用 OpenCV 计算多边形的轮廓面积
        return cv2.contourArea(polygon)
        # 以下是手动计算多边形面积的代码，已被注释掉
        # edge = 0
        # for i in range(polygon.shape[0]):
        #     next_index = (i + 1) % polygon.shape[0]
        #     edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] - polygon[i, 1])
        #
        # return edge / 2.
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 从 shapely.geometry 模块中导入 Polygon 类
    from shapely.geometry import Polygon
    # 导入 pyclipper 模块
    import pyclipper

    # 创建一个包含四个点的多边形数组
    polygon = np.array([[0, 0], [100, 10], [100, 100], [10, 90]])
    # 对多边形进行缩小操作，缩小比例为 0.4
    a = shrink_polygon_py(polygon, 0.4)
    # 打印结果
    print(a)
    # 对缩小后的多边形再进行放大操作，放大比例为 1 / 0.4
    print(shrink_polygon_py(a, 1 / 0.4))
    # 使用 pyclipper 模块对多边形进行缩小操作，缩小比例为 0.4
    b = shrink_polygon_pyclipper(polygon, 0.4)
    # 打印结果
    print(b)
    # 创建一个 shapely Polygon 对象
    poly = Polygon(b)
    # 计算多边形的面积乘以 1.5 除以周长，得到距离
    distance = poly.area * 1.5 / poly.length
    # 创建一个 PyclipperOffset 对象
    offset = pyclipper.PyclipperOffset()
    # 添加路径到 PyclipperOffset 对象中，采用圆形连接方式
    offset.AddPath(b, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    # 执行偏移操作，得到扩展后的多边形
    expanded = np.array(offset.Execute(distance))
    # 计算扩展后多边形的最小外接矩形
    bounding_box = cv2.minAreaRect(expanded)
    # 获取最小外接矩形的四个顶点坐标
    points = cv2.boxPoints(bounding_box)
    # 打印顶点坐标
    print(points)
```