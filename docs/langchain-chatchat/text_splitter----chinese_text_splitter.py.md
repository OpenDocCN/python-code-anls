# `.\Langchain-Chatchat\text_splitter\chinese_text_splitter.py`

```py
# 从 langchain.text_splitter 模块中导入 CharacterTextSplitter 类
from langchain.text_splitter import CharacterTextSplitter
# 导入 re 模块，用于正则表达式操作
import re
# 从 typing 模块中导入 List 类型
from typing import List

# 定义一个名为 ChineseTextSplitter 的类，继承自 CharacterTextSplitter 类
class ChineseTextSplitter(CharacterTextSplitter):
    # 初始化方法，接受参数 pdf（默认为 False）和 sentence_size（默认为 250），以及其他关键字参数
    def __init__(self, pdf: bool = False, sentence_size: int = 250, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置实例属性 pdf 为传入的 pdf 参数值
        self.pdf = pdf
        # 设置实例属性 sentence_size 为传入的 sentence_size 参数值
        self.sentence_size = sentence_size

    # 定义一个名为 split_text1 的方法，接受参数 text（字符串类型），返回一个字符串列表
    def split_text1(self, text: str) -> List[str]:
        # 如果 self.pdf 为 True
        if self.pdf:
            # 使用正则表达式替换连续三个以上的换行符为一个换行符
            text = re.sub(r"\n{3,}", "\n", text)
            # 使用正则表达式将空白字符替换为一个空格
            text = re.sub('\s', ' ', text)
            # 将连续两个换行符替换为空字符串
            text = text.replace("\n\n", "")
        
        # 定义一个正则表达式模式，用于分割句子
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
        # 初始化一个空列表 sent_list，用于存储分割后的句子
        sent_list = []
        # 遍历通过正则表达式模式分割后的文本
        for ele in sent_sep_pattern.split(text):
            # 如果 ele 匹配正则表达式模式并 sent_list 不为空
            if sent_sep_pattern.match(ele) and sent_list:
                # 将 ele 添加到 sent_list 最后一个元素中
                sent_list[-1] += ele
            # 如果 ele 不为空
            elif ele:
                # 将 ele 添加到 sent_list 中
                sent_list.append(ele)
        
        # 返回分割后的句子列表
        return sent_list
    # 定义一个方法，用于将文本分割成句子列表
    def split_text(self, text: str) -> List[str]:   ##此处需要进一步优化逻辑
        # 如果是 PDF 格式的文本
        if self.pdf:
            # 替换多余的换行符为单个换行符
            text = re.sub(r"\n{3,}", r"\n", text)
            # 替换空白字符为单个空格
            text = re.sub('\s', " ", text)
            # 替换连续两个换行符为一个空字符
            text = re.sub("\n\n", "", text)

        # 使用正则表达式进行文本分割
        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        
        # 将文本按照换行符分割成列表
        ls = [i for i in text.split("\n") if i]
        # 遍历每个句子
        for ele in ls:
            # 如果句子长度超过设定的句子长度
            if len(ele) > self.sentence_size:
                # 使用正则表达式进行进一步的分句处理
                ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                # 遍历每个子句
                for ele_ele1 in ele1_ls:
                    # 如果子句长度超过设定的句子长度
                    if len(ele_ele1) > self.sentence_size:
                        # 使用正则表达式进行进一步的分句处理
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")
                        # 遍历每个子子句
                        for ele_ele2 in ele2_ls:
                            # 如果子子句长度超过设定的句子长度
                            if len(ele_ele2) > self.sentence_size:
                                # 使用正则表达式进行进一步的分句处理
                                ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[ele2_id + 1:]
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
        # 返回处理后的句子列表
        return ls
```