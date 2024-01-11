# `arknights-mower\arknights_mower\ocr\crnn.py`

```
# 导入所需的库
import numpy as np
import onnxruntime as rt
from PIL import Image

# 从 keys 模块中导入 alphabetChinese 列表和从 utils 模块中导入 resizeNormalize 和 strLabelConverter 函数
from .keys import alphabetChinese as alphabet
from .utils import resizeNormalize, strLabelConverter

# 使用 alphabet 列表创建转换器对象
converter = strLabelConverter(''.join(alphabet))

# 定义 CRNNHandle 类
class CRNNHandle:
    # 初始化方法，接受模型路径参数
    def __init__(self, model_path):
        # 创建会话选项对象
        sess_options = rt.SessionOptions()
        # 设置日志级别
        sess_options.log_severity_level = 3
        # 创建推理会话对象
        self.sess = rt.InferenceSession(model_path, sess_options)

    # 预测方法，接受图像参数
    def predict(self, image):
        # 计算图像高度与32的比例
        scale = image.size[1] * 1.0 / 32
        # 根据比例调整图像宽度
        w = image.size[0] / scale
        w = int(w)
        # 创建 resizeNormalize 转换器对象，对图像进行处理
        transformer = resizeNormalize((w, 32))
        image = transformer(image)
        # 调整图像维度顺序
        image = image.transpose(2, 0, 1)
        # 在第0维度上增加一个维度
        transformed_image = np.expand_dims(image, axis=0)
        # 复制第0维度的数据，构成3通道图像
        transformed_image = np.array([[transformed_image[0, 0]] * 3])
        # 运行推理会话，获取预测结果
        preds = self.sess.run(
            ['out'], {'input': transformed_image.astype(np.float32)})
        preds = preds[0]
        length = preds.shape[0]
        preds = preds.reshape(length, -1)
        preds = np.argmax(preds, axis=1)
        preds = preds.reshape(-1)
        # 使用转换器对象解码预测结果
        sim_pred = converter.decode(preds, length, raw=False)
        return sim_pred

    # 预测 RGB 方法，接受图像参数
    def predict_rbg(self, image):
        # 计算图像高度与32的比例
        scale = image.size[1] * 1.0 / 32
        # 根据比例调整图像宽度
        w = image.size[0] / scale
        w = int(w)
        # 调整图像大小
        image = image.resize((w, 32), Image.BILINEAR)
        # 将图像转换为浮点类型的数组
        image = np.array(image, dtype=np.float32)
        # 数据归一化
        image -= 127.5
        image /= 127.5
        # 调整图像维度顺序
        image = image.transpose(2, 0, 1)
        # 在第0维度上增加一个维度
        transformed_image = np.expand_dims(image, axis=0)
        # 运行推理会话，获取预测结果
        preds = self.sess.run(
            ['out'], {'input': transformed_image.astype(np.float32)})
        preds = preds[0]
        length = preds.shape[0]
        preds = preds.reshape(length, -1)
        preds = np.argmax(preds, axis=1)
        preds = preds.reshape(-1)
        # 使用转换器对象解码预测结果
        sim_pred = converter.decode(preds, length, raw=False)
        return sim_pred
```