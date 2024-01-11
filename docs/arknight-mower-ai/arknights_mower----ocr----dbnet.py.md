# `arknights-mower\arknights_mower\ocr\dbnet.py`

```
import cv2  # 导入 OpenCV 库
import numpy as np  # 导入 NumPy 库
import onnxruntime as rt  # 导入 ONNX 运行时库
from .decode import SegDetectorRepresenter  # 从当前目录下的 decode 模块中导入 SegDetectorRepresenter 类

mean = (0.485, 0.456, 0.406)  # 设置均值
std = (0.229, 0.224, 0.225)  # 设置标准差


class DBNET():
    def __init__(self, model_path):
        sess_options = rt.SessionOptions()  # 创建 ONNX 运行时会话选项对象
        sess_options.log_severity_level = 3  # 设置日志级别为 3
        self.sess = rt.InferenceSession(model_path, sess_options)  # 创建 ONNX 推理会话对象
        self.decode_handel = SegDetectorRepresenter()  # 创建 SegDetectorRepresenter 实例

    def process(self, img, short_size):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 格式转换为 RGB 格式
        h, w = img.shape[:2]  # 获取图像的高度和宽度
        if h < w:
            scale_h = short_size / h  # 计算高度缩放比例
            tar_w = w * scale_h  # 计算目标宽度
            tar_w = tar_w - tar_w % 32  # 调整目标宽度为 32 的倍数
            tar_w = max(32, tar_w)  # 取目标宽度和 32 中的最大值
            scale_w = tar_w / w  # 计算宽度缩放比例
        else:
            scale_w = short_size / w  # 计算宽度缩放比例
            tar_h = h * scale_w  # 计算目标高度
            tar_h = tar_h - tar_h % 32  # 调整目标高度为 32 的倍数
            tar_h = max(32, tar_h)  # 取目标高度和 32 中的最大值
            scale_h = tar_h / h  # 计算高度缩放比例

        img = cv2.resize(img, None, fx=scale_w, fy=scale_h)  # 缩放图像
        img = img.astype(np.float32)  # 将图像数据类型转换为 float32

        img /= 255.0  # 归一化图像数据
        img -= mean  # 减去均值
        img /= std  # 除以标准差
        img = img.transpose(2, 0, 1)  # 转置图像数据
        transformed_image = np.expand_dims(img, axis=0)  # 在第 0 轴上增加维度
        out = self.sess.run(  # 使用 ONNX 推理会话运行模型
            ['out1'], {'input0': transformed_image.astype(np.float32)})  # 输入转换后的图像数据并获取输出
        box_list, score_list = self.decode_handel(out[0][0], h, w)  # 解析输出得到文本框列表和得分列表
        if len(box_list) > 0:  # 如果文本框列表不为空
            idx = box_list.reshape(
                box_list.shape[0], -1).sum(axis=1) > 0  # 计算非空文本框的索引
            box_list, score_list = box_list[idx], score_list[idx]  # 根据索引筛选文本框和得分
        else:  # 如果文本框列表为空
            box_list, score_list = [], []  # 设置文本框列表和得分列表为空
        return box_list, score_list  # 返回文本框列表和得分列表
```