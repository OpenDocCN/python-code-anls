# `.\translation\src\translator\translator.py`

```
# 导入 MBart50TokenizerFast 和 MBartForConditionalGeneration 类
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# 从 base.config 模块导入 Config 类
from base.config import Config

# 定义一个 Translator 类，继承自 Config 类
class Translator(Config):
    """mBART translator"""

    # 构造函数
    def __init__(self) -> None:
        super().__init__()
        # 初始化 mBART 模型，加载预训练模型
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.config["translator"]["model"]
        )
        # 初始化 mBART 分词器，加载预训练分词器
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            self.config["translator"]["model"], use_fast=False
        )

    # 定义翻译方法
    def translate(self, document: str, source_lang: str, target_lang: str) -> str:
        """
        Translate a given document based on the source and target language
        Args:
            document (str): document to translate
            source_lang (str): token for source language
            target_lang (str): token for target language
        Returns:
            str: translation
        """

        # 设置分词器的源语言
        self.tokenizer.src_lang = source_lang
        # 对输入文档进行编码，返回 PyTorch 张量
        encoded = self.tokenizer(document, return_tensors="pt")
        # 使用模型生成翻译结果的 tokens
        generated_tokens = self.model.generate(
            **encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang]
        )

        # 解码生成的 tokens，并返回翻译结果字符串
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
```