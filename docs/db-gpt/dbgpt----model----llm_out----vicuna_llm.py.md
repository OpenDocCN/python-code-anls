# `.\DB-GPT-src\dbgpt\model\llm_out\vicuna_llm.py`

```py
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import json  # 导入用于处理 JSON 数据的模块
from typing import Any, List, Mapping, Optional  # 引入类型提示模块
from urllib.parse import urljoin  # 导入用于处理 URL 的模块

import requests  # 导入发送 HTTP 请求的模块
from langchain.embeddings.base import Embeddings  # 导入自定义的嵌入模型基类
from langchain.llms.base import LLM  # 导入自定义的语言模型基类

from dbgpt._private.config import Config  # 导入私有配置类
from dbgpt._private.pydantic import BaseModel  # 导入基本模型类

CFG = Config()  # 实例化配置类对象


class VicunaLLM(LLM):
    vicuna_generate_path = "generate_stream"  # 定义生成文本流的路径

    def _call(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> str:
        params = {
            "prompt": prompt,  # 设置生成文本的提示语句
            "temperature": temperature,  # 设置生成文本的温度参数
            "max_new_tokens": max_new_tokens,  # 设置生成文本的最大新词汇数量
            "stop": stop,  # 可选参数，停止词列表
        }
        response = requests.post(
            url=urljoin(CFG.MODEL_SERVER, self.vicuna_generate_path),  # 构建完整的生成文本 API URL
            data=json.dumps(params),  # 将参数转换为 JSON 格式并发送 POST 请求
        )

        skip_echo_len = len(params["prompt"]) + 1 - params["prompt"].count("</s>") * 3  # 计算跳过回显的长度
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):  # 迭代处理响应数据块
            if chunk:  # 如果数据块非空
                data = json.loads(chunk.decode())  # 解析 JSON 数据块
                if data["error_code"] == 0:  # 如果返回的数据没有错误
                    output = data["text"][skip_echo_len:].strip()  # 获取生成的文本输出并去除首尾空白字符
                    yield output  # 返回生成的文本

    @property
    def _llm_type(self) -> str:
        return "custome"  # 返回当前语言模型的类型为自定义

    def _identifying_params(self) -> Mapping[str, Any]:
        return {}  # 返回空的识别参数映射


class VicunaEmbeddingLLM(BaseModel, Embeddings):
    vicuna_embedding_path = "embedding"  # 定义嵌入查询的路径

    def _call(self, prompt: str) -> str:
        p = prompt.strip()  # 去除输入提示的首尾空白字符
        print("Sending prompt ", p)  # 打印发送的提示语句

        response = requests.post(
            url=urljoin(CFG.MODEL_SERVER, self.vicuna_embedding_path),  # 构建完整的嵌入查询 API URL
            json={"prompt": p},  # 将提示语句封装为 JSON 发送 POST 请求
        )
        response.raise_for_status()  # 检查响应状态，如果不是成功状态则抛出异常
        return response.json()["response"]  # 返回嵌入查询的响应数据

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Vicuna's server embedding endpoint for embedding search docs.

        Args:
            texts: The list of text to embed

        Returns:
            List of embeddings. one for each text.
        """
        results = []
        for text in texts:  # 遍历文本列表
            response = self.embed_query(text)  # 调用嵌入查询方法
            results.append(response)  # 将每个文本的嵌入结果添加到结果列表中
        return results  # 返回所有文本的嵌入结果列表

    def embed_query(self, text: str) -> List[float]:
        """Call out to Vicuna's server embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text
        """
        embedding = self._call(text)  # 调用内部的嵌入方法获取文本的嵌入表示
        return embedding  # 返回文本的嵌入表示
```