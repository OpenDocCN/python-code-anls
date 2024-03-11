# `.\VisualGLM-6B\model\infer_util.py`

```
# 导入所需的库
import os
from PIL import Image
from io import BytesIO
import base64
import re
import argparse
import torch
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
import hashlib
from .visualglm import VisualGLMModel

# 获取推理设置，包括 GPU 设备和量化参数
def get_infer_setting(gpu_device=0, quant=None):
    # 设置环境变量 CUDA_VISIBLE_DEVICES 为指定的 GPU 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    # 定义参数对象
    args = argparse.Namespace(
        fp16=True,
        skip_init=True,
        device='cuda' if quant is None else 'cpu',
    )
    # 从预训练模型加载 VisualGLMModel 模型
    model, args = VisualGLMModel.from_pretrained('visualglm-6b', args)
    # 添加自动回归的 mixin
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    # 确保量化参数在指定范围内
    assert quant in [None, 4, 8]
    # 如果有量化参数，则对模型进行量化
    if quant is not None:
        quantize(model.transformer, quant)
    # 设置模型为评估模式
    model.eval()
    # 将模型移动到 GPU 上
    model = model.cuda()
    # 从预训练模型加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # 返回模型和 tokenizer
    return model, tokenizer

# 判断文本中是否包含中文字符
def is_chinese(text):
    # 定义中文字符的正则表达式
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    return zh_pattern.search(text)

# 生成输入数据，包括文本、图片、历史记录和生成参数
def generate_input(input_text, input_image_prompt, history=[], input_para=None, image_is_encoded=True):
    # 如果图片未经编码，则直接使用输入的图片数据
    if not image_is_encoded:
        image = input_image_prompt
    else:
        # 解码 base64 编码的图片数据
        decoded_image = base64.b64decode(input_image_prompt)
        image = Image.open(BytesIO(decoded_image))

    # 构建输入数据字典
    input_data = {'input_query': input_text, 'input_image': image, 'history': history, 'gen_kwargs': input_para}
    return input_data

# 处理图片数据，保存图片并返回图片路径
def process_image(image_encoded):
    # 解码 base64 编码的图片数据
    decoded_image = base64.b64decode(image_encoded)
    image = Image.open(BytesIO(decoded_image))
    # 计算图片数据的哈希值
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    # 构建图片保存路径
    image_path = f'./examples/{image_hash}.png'
    # 如果图片文件不存在，则保存图片
    if not os.path.isfile(image_path):
        image.save(image_path)
    # 返回图片的绝对路径
    return os.path.abspath(image_path)  
```