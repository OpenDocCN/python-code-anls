# `.\models\roformer\tokenization_utils.py`

```
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization utils for RoFormer."""

from typing import List

from tokenizers import NormalizedString, PreTokenizedString, normalizers


class JiebaPreTokenizer:
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        # 初始化BERT风格的文本规范化器，用于清理文本，处理中文字符，不去除重音符号，不转换为小写
        self.normalizers = normalizers.BertNormalizer(
            clean_text=False,
            handle_chinese_chars=True,
            strip_accents=False,
            lowercase=False,
        )
        try:
            import rjieba
        except ImportError:
            # 如果导入rjieba失败，引发ImportError并提供安装链接
            raise ImportError(
                "You need to install rjieba to use RoFormerTokenizer. "
                "See https://pypi.org/project/rjieba/ for installation."
            )
        # 导入成功后，将rjieba赋值给self.jieba
        self.jieba = rjieba

    def jieba_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = []

        # 使用rjieba对normalized_string进行分词，hmm参数设为False以提高速度
        for token, start, end in self.jieba.tokenize(str(normalized_string), hmm=False):
            # 如果分词结果在词汇表中，则将对应的NormalizedString加入splits列表
            if token in self.vocab:
                splits.append(normalized_string[start:end])
            else:
                # 否则，对token进行文本规范化处理，并按照处理后的结果拆分为多个token加入splits列表
                token_list = self.normalizers.normalize_str(token).split()
                for token in token_list:
                    if token:
                        end = start + len(token)
                        splits.append(normalized_string[start:end])
                        start = end

        # 返回分词后的NormalizedString列表
        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        # 使用jieba_split方法对PreTokenizedString对象进行分词处理
        pretok.split(self.jieba_split)
```