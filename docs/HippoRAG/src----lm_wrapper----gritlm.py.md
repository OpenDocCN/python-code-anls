# `.\HippoRAG\src\lm_wrapper\gritlm.py`

```py
# See https://github.com/ContextualAI/gritlm
from typing import Union, List  # 从 typing 模块导入 Union 和 List 类型，用于类型提示

import numpy  # 导入 numpy 库用于数值计算
import numpy as np  # 将 numpy 库别名为 np
import torch  # 导入 PyTorch 库用于深度学习
from gritlm import GritLM  # 从 gritlm 模块导入 GritLM 类，用于语言模型

from src.lm_wrapper import EmbeddingModelWrapper  # 从 src.lm_wrapper 模块导入 EmbeddingModelWrapper 基类


def gritlm_instruction(instruction):
    # 如果提供了 instruction，则格式化为 GritLM 需要的输入格式；否则返回关闭标签
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


class GritWrapper(EmbeddingModelWrapper):
    # GritWrapper 类继承自 EmbeddingModelWrapper，用于封装 GritLM 模型的功能
    def __init__(self, model_name: str = 'GritLM/GritLM-7B', **kwargs):
        """
        加载 GritLM 模型。如果只需要嵌入功能，传递 `mode="embedding"` 以节省内存（没有语言模型头部）。
        加载 8x7B 版本可能需要多 GPU。
        @param model_name: 模型名称
        @param kwargs: 其他参数
        """
        self.model = GritLM(model_name, torch_dtype='auto', **kwargs)  # 初始化 GritLM 模型实例

    def encode_list(self, texts: list, instruction: str, batch_size=96):
        # 使用 GritLM 模型对文本列表进行编码
        return self.model.encode(texts, instruction=gritlm_instruction(instruction), batch_size=batch_size)

    def encode_text(self, text: Union[str, List], instruction: str = '', norm=True, return_numpy=False, return_cpu=False):
        # 将文本进行编码，支持字符串或列表，应用指令并进行规范化
        if isinstance(text, str):
            text = [text]  # 如果 text 是字符串，则转换为列表
        if isinstance(text, list):
            res = self.encode_list(text, instruction)  # 对列表文本进行编码
        else:
            raise ValueError(f"Expected str or list, got {type(text)}")  # 如果 text 既不是字符串也不是列表，则引发错误
        if isinstance(res, torch.Tensor):
            if return_cpu:
                res = res.cpu()  # 如果 return_cpu 为 True，将 tensor 转移到 CPU 上
            if return_numpy:
                res = res.numpy()  # 如果 return_numpy 为 True，将 tensor 转换为 numpy 数组
        if norm:
            if isinstance(res, torch.Tensor):
                res = res.T.divide(torch.linalg.norm(res, dim=1)).T  # 对 tensor 进行规范化
            if isinstance(res, np.ndarray):
                res = (res.T / np.linalg.norm(res, axis=1)).T  # 对 numpy 数组进行规范化
        return res  # 返回编码和规范化后的结果

    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray):
        """
        @param query_vec: 查询向量
        @param doc_vecs: 文档矩阵
        @return: 查询-文档评分矩阵
        """
        return np.dot(doc_vecs, query_vec.T)  # 计算文档矩阵与查询向量的点积，得到查询-文档评分矩阵

    def generate(self, messages: list, max_new_tokens=256, do_sample=False):
        """
        @param messages: 消息列表，例如 [{"role": "user", "content": "Please write me a poem."}]
        @return: 生成的文本
        """
        encoded = self.model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")  # 将消息编码为模型输入
        encoded = encoded.to(self.model.device)  # 将编码的输入移到模型设备上
        gen = self.model.generate(encoded, max_new_tokens=max_new_tokens, do_sample=do_sample)  # 生成文本
        decoded = self.model.tokenizer.batch_decode(gen)  # 解码生成的文本
        return decoded  # 返回解码后的文本
```