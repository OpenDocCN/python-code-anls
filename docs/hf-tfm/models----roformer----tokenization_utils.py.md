# `.\transformers\models\roformer\tokenization_utils.py`

```
# 设置编码格式为 utf-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有，保留所有权利
# 根据 Apache 许可证，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，没有任何明示或暗示的保证或条件
# 有关特定语言的权限和限制，请参阅许可证

"""RoFormer 的标记化工具"""

from typing import List
# 导入必要的库和模块
from tokenizers import NormalizedString, PreTokenizedString, normalizers

class JiebaPreTokenizer:
    def __init__(self, vocab) -> None:
        # 初始化 JiebaPreTokenizer 类
        self.vocab = vocab
        # 声明 normalizers.BertNormalizer 对象
        self.normalizers = normalizers.BertNormalizer(
            clean_text=False,
            handle_chinese_chars=True,
            strip_accents=False,
            lowercase=False,
        )
        # 尝试导入 rjieba 模块，不存在则抛出异常
        try:
            import rjieba
        except ImportError:
            raise ImportError(
                "To use RoFormerTokenizer, you need to install rjieba. "
                "Please visit https://pypi.org/project/rjieba/ for installation instructions."
            )
        self.jieba = rjieba

    def jieba_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = []

        # 利用 rjieba 模块对字符串进行分词
        for token, start, end in self.jieba.tokenize(str(normalized_string), hmm=False):
            # 如果分词结果在词汇表中存在，则添加到 splits 中
            if token in self.vocab:
                splits.append(normalized_string[start:end])
            else:
                # 对分词结果进行规范化处理并切分
                token_list = self.normalizers.normalize_str(token).split()
                for token in token_list:
                    if token:
                        end = start + len(token)
                        splits.append(normalized_string[start:end])
                        start = end

        # 利用 rjieba 模块对字符串进行分词，效率更高但无法通过测试
        # for token in self.jieba.cut(str(normalized_string), False):
        #     if token in self.vocab:
        #         splits.append(NormalizedString(token))
        #     else:
        #         token_list = self.normalizers.normalize_str(token).split()
        #         for token in token_list:
        #             if token:
        #                 splits.append(NormalizedString(token))

        return splits

    # 对预标记化字符串进行分��处理
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.jieba_split)
```