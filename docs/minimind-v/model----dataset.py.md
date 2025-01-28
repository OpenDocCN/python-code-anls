# `.\minimind-v\model\dataset.py`

```
# 导入所需的库
import json  # 用于处理 JSON 数据格式
import random  # 用于生成随机数或选择随机项
import re  # 用于正则表达式操作

# 导入数据处理和计算库
import pandas as pd  # 用于数据处理，特别是数据帧（DataFrame）
import numpy as np  # 用于处理数值计算和数组操作

# 导入图像处理和模型相关的库
from PIL import Image  # 用于图像处理
from torch.utils.data import Dataset, DataLoader  # 用于数据加载与处理，Dataset 定义数据集，DataLoader 用于批量加载数据
import torch  # PyTorch 框架，主要用于深度学习
from model.vision_utils import get_img_process  # 导入自定义的图像处理函数
import os  # 用于处理操作系统功能，如文件路径

# 设置环境变量，关闭 tokenizer 的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 定义一个自定义的 Dataset 类，用于预训练数据集
class PretrainDataset(Dataset):
    def __init__(self, json_path, tokenizer, vision_model=None, max_length=1024,
                 prompt_max_len=512,
                 answer_max_len=256,
                 image_special_token='<' * 25 + '>' * 25):
        
        super().__init__()  # 调用父类 Dataset 的初始化方法
        
        # 打开 JSON 文件并加载内容为 Python 字典
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 初始化最大长度、提示语最大长度、回答最大长度
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        
        # 初始化 tokenizer 和视觉模型（如果有的话）
        self.tokenizer = tokenizer
        self.vision_model, self.preprocess = vision_model  # 将视觉模型和预处理函数存储起来
        self.padding = 0  # 初始化 padding 参数（可能用于填充数据）
        
        # 获取 '<s>assistant' 对应的 token id，用于标识某种特殊开始符号
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']
        
        # 数据集图片所在的文件夹路径
        self.dataset_path = './dataset/pretrain_images/'
        
        # 定义特殊的图像 token
        self.image_special_token = image_special_token

    # 返回数据集的长度
    def __len__(self):
        return len(self.data)

    # 查找子列表在主列表中的最后位置索引
    def find_sublist_index(self, main_list, sub_list) -> int:
        last_index = -1  # 初始化返回的索引，默认 -1 表示未找到
        for i in range(len(main_list) - len(sub_list) + 1):  # 遍历主列表的每个位置
            if main_list[i:i + len(sub_list)] == sub_list:  # 如果发现子列表匹配
                last_index = i  # 更新索引
        return last_index  # 返回最后一次匹配的索引

    # 安全地执行 eval，避免异常
    def safe_eval(self, s):
        try:
            res = eval(s)  # 尝试执行传入的字符串作为 Python 代码
        except Exception as e:
            return []  # 如果发生异常，返回空列表
        return res  # 返回执行结果
    # 定义获取指定索引位置样本的方法
    def __getitem__(self, index: int):
        # 获取数据中指定索引位置的样本
        sample = self.data[index]
        # 获取样本中的图像文件名
        image_name = sample['image']
        # 获取样本中的对话列表
        conversation = sample['conversations']
        # 初始化存储消息的列表
        messages = []
        # 遍历 conversation 列表，每次步长为 2
        for i in range(0, len(conversation), 2):
            # 检查是否有配对的问题和回答
            if i + 1 < len(conversation):
                # 获取问题并替换图像占位符
                q = conversation[i]['value'].replace('<image>', self.image_special_token)
                # 获取回答
                a = conversation[i + 1]['value']
    
                # 如果问题和回答都有值，则添加到消息列表
                if q and a:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})
    
        # 使用 tokenizer 应用聊天模板，生成新的提示信息
        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 获取新提示信息的输入ID，并截断至最大长度
        input_id = self.tokenizer(new_prompt).data['input_ids'][:self.max_length]
    
        # 计算问题的长度，寻找问句的结束标志的位置
        question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        # 计算填充长度
        padding_len = self.max_length - len(input_id)
        # 补充输入ID至最大长度
        input_id = input_id + [self.padding] * padding_len
        # 计算掩码的长度
        mask_len = len(input_id) - question_length - padding_len
        # 生成掩码，用于计算损失
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len
    
        # 将输入ID转换为 NumPy 数组
        input_id = np.array(input_id)
    
        # 创建 X，Y 张量，X 为输入数据，Y 为目标数据
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        # 转换掩码为 NumPy 数组
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
    
        # 将 X、Y 和 loss_mask 转换为 PyTorch 张量
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)
    
        # 打开图像文件
        image = Image.open(f'{self.dataset_path}{image_name}')
        # 获取图像处理器并对图像进行预处理
        image_encoders = get_img_process(image, self.preprocess)
    
        # 返回处理后的数据，包括输入、目标、损失掩码和图像编码器
        return X_tensor, Y_tensor, loss_mask_tensor, image_encoders
# 定义一个名为 SFTDataset 的类，继承自 Dataset 类
class SFTDataset(Dataset):
    # 初始化函数，接受参数 json_path, tokenizer, vision_model, max_length, prompt_max_len, answer_max_len, image_special_token
    def __init__(self, json_path, tokenizer, vision_model=None, max_length=1024,
                 prompt_max_len=512,
                 answer_max_len=256,
                 image_special_token='<' * 25 + '>' * 25):
        # 调用父类的初始化函数
        super().__init__()
        # 打开并读取指定路径的 JSON 文件，将数据加载到 self.data 中
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 设置最大长度、提示最大长度、答案最大长度、分词器、视觉模型、预处理函数、填充值、起始符号 ID、数据集路径和图像特殊标记
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        self.tokenizer = tokenizer
        self.vision_model, self.preprocess = vision_model
        self.padding = 0
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']
        self.dataset_path = './dataset/sft_images/'
        self.image_special_token = image_special_token

    # 返回数据集的长度
    def __len__(self):
        return len(self.data)

    # 查找子列表在主列表中的索引位置
    def find_sublist_index(self, main_list, sub_list) -> int:
        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i + len(sub_list)] == sub_list:
                last_index = i
        return last_index

    # 安全地评估字符串表达式
    def safe_eval(self, s):
        try:
            res = eval(s)
        except Exception as e:
            return []
        return res
    # 获取指定索引的数据样本并处理，返回用于训练的张量和图像编码
    def __getitem__(self, index: int):
        # 获取数据样本
        sample = self.data[index]
        # 根据样本中的图片名称构建图像文件名
        image_name = 'COCO_train2014_' + sample['image']
        # 获取该样本中的对话内容
        conversation = sample['conversations']
        # 初始化消息列表，用于存放对话中的问题和回答
        messages = []
        # 遍历对话列表，每两个元素为一对问题和回答
        # for i in range(0, len(conversation), 2):
        for i in range(0, 1):  # 这里只处理第一个对话
            # 如果当前对话有问题和回答配对
            if i + 1 < len(conversation):
                # 获取问题，并将其中的<image>替换为特定的图像占位符
                q = conversation[i]['value'].replace('<image>', self.image_special_token)
                # 获取回答
                a = conversation[i + 1]['value']
    
                # 如果问题和回答都存在，添加到消息列表
                if q and a:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})
    
        # 使用 tokenizer 对消息列表应用聊天模板，生成新提示
        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 不进行标记化
            add_generation_prompt=True  # 添加生成提示
        )
        # 将新提示输入到 tokenizer 中进行编码，获取输入的 ID，并限制最大长度
        input_id = self.tokenizer(new_prompt).data['input_ids'][:self.max_length]
    
        # 计算问题部分的长度
        question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        # 计算填充的长度
        padding_len = self.max_length - len(input_id)
        # 对输入 ID 进行填充，填充部分为 padding 标记
        input_id = input_id + [self.padding] * padding_len
        # 计算掩码长度（不包括问题部分和填充部分）
        mask_len = len(input_id) - question_length - padding_len
        # 生成损失掩码：问题部分为 0，掩码部分为 1，填充部分为 0
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len
    
        # 将 input_id 转换为 numpy 数组
        input_id = np.array(input_id)
    
        # 构建 X 和 Y，分别是输入和目标数据，Y 是 X 向后偏移一个位置
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        # 将损失掩码转换为 numpy 数组
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
    
        # 将 X, Y 和损失掩码转换为 PyTorch 张量
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)
    
        # 打开图像文件
        image = Image.open(f'{self.dataset_path}{image_name}')
        # 对图像进行预处理，得到图像编码
        image_encoders = get_img_process(image, self.preprocess)
    
        # 返回训练所需的张量和图像编码
        return X_tensor, Y_tensor, loss_mask_tensor, image_encoders
# 定义一个名为 SFTDataset_multi 的类，继承自 Dataset 类
class SFTDataset_multi(Dataset):
    # 初始化函数，接受参数 json_path, tokenizer, vision_model, max_length, prompt_max_len, answer_max_len, image_special_token
    def __init__(self, json_path, tokenizer, vision_model=None, max_length=1024,
                 prompt_max_len=512,
                 answer_max_len=256,
                 image_special_token='<' * 25 + '>' * 25):

        # 调用父类的初始化函数
        super().__init__()
        # 以只读方式打开 json 文件，并加载数据到 self.data 中
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 设置类的属性值
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        self.tokenizer = tokenizer
        self.vision_model, self.preprocess = vision_model
        self.padding = 0
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']
        self.dataset_path = './dataset/sft_multi_images/'
        self.image_special_token = image_special_token

    # 返回数据集的长度
    def __len__(self):
        return len(self.data)

    # 定义一个名为 find_sublist_index 的方法，用于查找子列表在主列表中的位置
    def find_sublist_index(self, main_list, sub_list) -> int:
        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i + len(sub_list)] == sub_list:
                last_index = i
        return last_index

    # 定义一个名为 safe_eval 的方法，用于安全地执行字符串表达式
    def safe_eval(self, s):
        try:
            res = eval(s)
        except Exception as e:
            return []
        return res
```