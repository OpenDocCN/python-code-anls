# `.\HippoRAG\src\lm_wrapper\huggingface_util.py`

```py
# 从 typing 模块引入 Union 和 List 类型注解
from typing import Union, List

# 导入 numpy 库作为 np
import numpy as np
# 导入 PyTorch 库作为 torch
import torch
# 导入 tqdm 库用于进度条显示
from tqdm import tqdm
# 从 transformers 模块导入 AutoModel 和 AutoTokenizer
from transformers import AutoModel, AutoTokenizer

# 从 src.lm_wrapper 模块导入 EmbeddingModelWrapper 类
from src.lm_wrapper import EmbeddingModelWrapper
# 从 src.processing 模块导入 mean_pooling_embedding_with_normalization 和 mean_pooling_embedding 函数
from src.processing import mean_pooling_embedding_with_normalization, mean_pooling_embedding


# 定义 HuggingFaceWrapper 类，继承自 EmbeddingModelWrapper
class HuggingFaceWrapper(EmbeddingModelWrapper):
    # 初始化方法
    def __init__(self, model_name: str, device='cuda'):
        # 存储模型名称
        self.model_name = model_name
        # 处理模型名称，将 '/' 和 '.' 替换为 '_'
        self.model_name_processed = model_name.replace('/', '_').replace('.', '_')
        # 从预训练模型中加载模型并移动到指定设备
        self.model = AutoModel.from_pretrained(model_name).to(device)
        # 从预训练模型中加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 存储设备信息
        self.device = device

    # 编码文本的方法
    def encode_text(self, text: Union[str, List], instruction=None, norm=True, return_cpu=False, return_numpy=False):
        # 选择合适的编码函数
        encoding_func = mean_pooling_embedding_with_normalization if norm else mean_pooling_embedding
        # 在不计算梯度的情况下进行计算
        with torch.no_grad():
            # 如果文本是字符串，将其转换为列表
            if isinstance(text, str):
                text = [text]
            # 初始化结果列表
            res = []
            # 如果文本长度大于 1，遍历所有文本
            if len(text) > 1:
                for t in tqdm(text, total=len(text), desc=f"HF model {self.model_name} encoding"):
                    # 使用选择的编码函数处理每个文本
                    res.append(encoding_func(t, self.tokenizer, self.model, self.device))
            else:
                # 如果只有一个文本，直接处理
                res = [encoding_func(text[0], self.tokenizer, self.model, self.device)]
            # 将结果列表转换为 PyTorch 张量
            res = torch.stack(res)
            # 压缩张量的维度
            res = torch.squeeze(res, dim=1)

        # 如果指定返回 CPU 张量，将张量移动到 CPU
        if return_cpu:
            res = res.cpu()
        # 如果指定返回 NumPy 数组，将张量转换为 NumPy 数组
        if return_numpy:
            res = res.numpy()
        # 返回处理后的结果
        return res

    # 计算查询与文档的得分矩阵的方法
    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray):
        """

        @param query_vec: 查询向量
        @param doc_vecs: 文档向量矩阵
        @return: 查询-文档得分矩阵
        """
        # 计算文档向量与查询向量的点积
        return np.dot(doc_vecs, query_vec.T)
```