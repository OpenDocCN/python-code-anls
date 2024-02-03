# `.\PaddleOCR\PPOCRLabel\libs\create_ml_io.py`

```py
# 版权声明，允许在特定条件下使用和分发软件
# 导入所需的模块和库
import json
from pathlib import Path
from libs.constants import DEFAULT_ENCODING
import os

# 定义常量
JSON_EXT = '.json'
ENCODE_METHOD = DEFAULT_ENCODING

# 创建一个类 CreateMLWriter
class CreateMLWriter:
    # 初始化方法，接受多个参数
    def __init__(self, foldername, filename, imgsize, shapes, outputfile, databasesrc='Unknown', localimgpath=None):
        # 初始化实例变量
        self.foldername = foldername
        self.filename = filename
        self.databasesrc = databasesrc
        self.imgsize = imgsize
        self.boxlist = []
        self.localimgpath = localimgpath
        self.verified = False
        self.shapes = shapes
        self.outputfile = outputfile
    # 将标注数据写入输出文件
    def write(self):
        # 检查输出文件是否存在
        if os.path.isfile(self.outputfile):
            # 如果存在，读取文件内容并解析为字典
            with open(self.outputfile, "r") as file:
                input_data = file.read()
                outputdict = json.loads(input_data)
        else:
            # 如果不存在，创建空列表
            outputdict = []

        # 创建包含图片文件名和空标注列表的字典
        outputimagedict = {
            "image": self.filename,
            "annotations": []
        }

        # 遍历每个形状
        for shape in self.shapes:
            points = shape["points"]

            # 计算矩形的坐标
            x1 = points[0][0]
            y1 = points[0][1]
            x2 = points[1][0]
            y2 = points[2][1]

            height, width, x, y = self.calculate_coordinates(x1, x2, y1, y2)

            # 创建包含标签和坐标信息的字典
            shapedict = {
                "label": shape["label"],
                "coordinates": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
            }
            outputimagedict["annotations"].append(shapedict)

        # 检查图片是否已经在输出中
        exists = False
        for i in range(0, len(outputdict)):
            if outputdict[i]["image"] == outputimagedict["image"]:
                exists = True
                outputdict[i] = outputimagedict
                break

        # 如果图片不在输出中，将其添加到输出字典中
        if not exists:
            outputdict.append(outputimagedict)

        # 将输出字典转换为 JSON 格式并写入输出文件
        Path(self.outputfile).write_text(json.dumps(outputdict), ENCODE_METHOD)

    # 计算矩形的高度、宽度、中心点坐标
    def calculate_coordinates(self, x1, x2, y1, y2):
        if x1 < x2:
            xmin = x1
            xmax = x2
        else:
            xmin = x2
            xmax = x1
        if y1 < y2:
            ymin = y1
            ymax = y2
        else:
            ymin = y2
            ymax = y1
        width = xmax - xmin
        if width < 0:
            width = width * -1
        height = ymax - ymin
        # 计算矩形中心点坐标
        x = xmin + width / 2
        y = ymin + height / 2
        return height, width, x, y
# 创建一个用于读取 CreateML 数据的类
class CreateMLReader:
    # 初始化方法，接收 JSON 文件路径和图像文件路径
    def __init__(self, jsonpath, filepath):
        # 将 JSON 文件路径和图像文件路径保存到对象属性中
        self.jsonpath = jsonpath
        self.shapes = []  # 初始化图像形状列表
        self.verified = False  # 初始化验证状态为 False
        self.filename = filepath.split("/")[-1:][0]  # 从文件路径中提取文件名
        try:
            self.parse_json()  # 解析 JSON 文件
        except ValueError:
            print("JSON decoding failed")  # 捕获 JSON 解码失败的异常并打印错误信息

    # 解析 JSON 文件的方法
    def parse_json(self):
        with open(self.jsonpath, "r") as file:  # 打开 JSON 文件
            inputdata = file.read()  # 读取文件内容

        outputdict = json.loads(inputdata)  # 将 JSON 数据解析为字典
        self.verified = True  # 设置验证状态为 True

        if len(self.shapes) > 0:  # 如果图像形状列表不为空
            self.shapes = []  # 清空图像形状列表
        for image in outputdict:  # 遍历 JSON 数据中的图像信息
            if image["image"] == self.filename:  # 如果图像文件名匹配当前文件名
                for shape in image["annotations"]:  # 遍历图像的注释信息
                    self.add_shape(shape["label"], shape["coordinates"])  # 添加图像形状

    # 添加图像形状的方法
    def add_shape(self, label, bndbox):
        xmin = bndbox["x"] - (bndbox["width"] / 2)  # 计算矩形左上角 x 坐标
        ymin = bndbox["y"] - (bndbox["height"] / 2)  # 计算矩形左上角 y 坐标

        xmax = bndbox["x"] + (bndbox["width"] / 2)  # 计算矩形右下角 x 坐标
        ymax = bndbox["y"] + (bndbox["height"] / 2)  # 计算矩形右下角 y 坐标

        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]  # 构建矩形四个顶点坐标
        self.shapes.append((label, points, None, None, True))  # 将标签、顶点坐标和其他信息添加到图像形状列表中

    # 获取图像形状列表的方法
    def get_shapes(self):
        return self.shapes  # 返回图像形状列表
```