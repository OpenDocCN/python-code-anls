# `.\Langchain-Chatchat\text_splitter\ali_text_splitter.py`

```py
from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List


class AliTextSplitter(CharacterTextSplitter):
    # 继承自CharacterTextSplitter类的AliTextSplitter类，用于文本分割
    def __init__(self, pdf: bool = False, **kwargs):
        # 初始化方法，pdf参数默认为False
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        # 分割文本的方法，接受一个字符串参数text，返回一个字符串列表
        # use_document_segmentation参数指定是否用语义切分文档，此处采取的文档语义分割模型为达摩院开源的nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
        # 如果使用模型进行文档语义切分，那么需要安装modelscope[nlp]：pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        # 考虑到使用了三个模型，可能对于低配置gpu不太友好，因此这里将模型load进cpu计算，有需要的话可以替换device为自己的显卡id
        if self.pdf:
            # 如果是pdf格式的文本，进行一些文本处理
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        try:
            from modelscope.pipelines import pipeline
        except ImportError:
            # 如果导入modelscope失败，抛出ImportError异常
            raise ImportError(
                "Could not import modelscope python package. "
                "Please install modelscope with `pip install modelscope`. "
            )

        # 使用modelscope库的pipeline方法创建一个文档分割的pipeline对象
        p = pipeline(
            task="document-segmentation",
            model='damo/nlp_bert_document-segmentation_chinese-base',
            device="cpu")
        # 对文本进行文档分割
        result = p(documents=text)
        # 将分割后的文本按换行符分割成列表
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list
```