# `.\PaddleOCR\StyleText\engine\corpus_generators.py`

```
# 版权声明和许可信息
# 版权所有 (c) 2020 PaddlePaddle 作者。保留所有权利。
# 根据 Apache 许可证 2.0 版本（“许可证”）授权;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件
# 根据许可证“按原样”分发，
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证以了解特定语言的权限和
# 限制。
import random

# 导入自定义的日志记录模块
from utils.logging import get_logger

# 定义 FileCorpus 类
class FileCorpus(object):
    # 初始化方法
    def __init__(self, config):
        # 获取日志记录器
        self.logger = get_logger()
        # 记录日志信息
        self.logger.info("using FileCorpus")

        # 定义字符列表
        self.char_list = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        # 从配置中获取语料文件路径和语言信息
        corpus_file = config["CorpusGenerator"]["corpus_file"]
        self.language = config["CorpusGenerator"]["language"]
        # 读取语料文件内容
        with open(corpus_file, 'r') as f:
            corpus_raw = f.read()
        # 将语料内容按换行符分割成列表
        self.corpus_list = corpus_raw.split("\n")[:-1]
        # 断言语料列表长度大于0
        assert len(self.corpus_list) > 0
        # 随机打乱语料列表顺序
        random.shuffle(self.corpus_list)
        # 初始化索引
        self.index = 0

    # 生成方法
    def generate(self, corpus_length=0):
        # 如果索引超出语料列表长度，重置索引并重新打乱语料列表
        if self.index >= len(self.corpus_list):
            self.index = 0
            random.shuffle(self.corpus_list)
        # 获取当前语料
        corpus = self.corpus_list[self.index]
        # 如果指定了语料长度，截取语料
        if corpus_length != 0:
            corpus = corpus[0:corpus_length]
        # 如果指定的语料长度大于实际语料长度，记录警告信息
        if corpus_length > len(corpus):
            self.logger.warning("generated corpus is shorter than expected.")
        # 更新索引
        self.index += 1
        # 返回语言和语料
        return self.language, corpus

# 定义 EnNumCorpus 类
class EnNumCorpus(object):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 获取日志记录器对象
        self.logger = get_logger()
        # 记录日志信息
        self.logger.info("using NumberCorpus")
        # 初始化数字列表
        self.num_list = "0123456789"
        # 初始化英文字母列表
        self.en_char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        # 从配置中获取图像高度
        self.height = config["Global"]["image_height"]
        # 从配置中获取图像宽度
        self.max_width = config["Global"]["image_width"]

    # 生成随机文本
    def generate(self, corpus_length=0):
        # 初始化文本内容为空字符串
        corpus = ""
        # 如果未指定文本长度，则随机生成一个长度
        if corpus_length == 0:
            corpus_length = random.randint(5, 15)
        # 循环生成文本内容
        for i in range(corpus_length):
            # 以0.2的概率选择英文字母或数字
            if random.random() < 0.2:
                corpus += "{}".format(random.choice(self.en_char_list))
            else:
                corpus += "{}".format(random.choice(self.num_list))
        # 返回生成的文本内容和类型
        return "en", corpus
```