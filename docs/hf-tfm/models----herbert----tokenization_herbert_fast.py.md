# `.\models\herbert\tokenization_herbert_fast.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，包括作者和许可证信息
# 从 typing 模块导入 List、Optional、Tuple 类型
# 从 tokenization_utils_fast 模块中导入 PreTrainedTokenizerFast 类
# 从 utils 模块中导入 logging 函数
# 从当前目录下的 tokenization_herbert 模块中导入 HerbertTokenizer 类
# 获取 logger 对象
# 定义词汇文件的名称
# 定义预训练模型的词汇文件映射
# 定义预训练模型的位置嵌入大小
# 定义预训练模型的初始化配置
# 定义 HerbertTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
# HerbertTokenizerFast 类的构造函数，接受词汇文件、合并文件等参数
# 设置 HerbertTokenizerFast 类的词汇文件名称
# 设置 HerbertTokenizerFast 类的预训练词汇文件映射
# 设置 HerbertTokenizerFast 类的预训练初始化配置
# 设置 HerbertTokenizerFast 类的最大模型输入大小
# 设置 HerbertTokenizerFast 类的慢速分词器类为 HerbertTokenizer
# HerbertTokenizerFast 类的初始化方法，接受词汇文件、合并文件等参数
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。像 BERT 一样，HerBERT 序列的格式如下：

        - 单个序列: `<s> X </s>`
        - 一对序列: `<s> A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                将添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """

        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep

        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。当使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ):
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. HerBERT, like
        BERT sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Define the separator token ID and the classification token ID
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # Check if there is a second list of token IDs provided
        if token_ids_1 is None:
            # If not, return a mask with zeros for the first sequence
            return len(cls + token_ids_0 + sep) * [0]
        # If there is a second list of token IDs, return a mask with zeros for the first sequence and ones for the second sequence
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Save the vocabulary files to the specified directory with the given filename prefix
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # Return the list of saved files as a tuple
        return tuple(files)
```