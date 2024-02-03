# `jieba\test\test_tokenize.py`

```
# 设置文件编码为utf-8
# 导入print_function和unicode_literals功能
from __future__ import print_function, unicode_literals
# 导入sys模块
import sys
# 将上级目录添加到sys.path中
sys.path.append("../")
# 导入jieba模块
import jieba

# 设置全局变量g_mode为"default"
g_mode = "default"

# 定义函数cuttest，用于对输入的句子进行分词处理
def cuttest(test_sent):
    # 使用全局变量g_mode作为分词模式，对输入的句子进行分词处理
    result = jieba.tokenize(test_sent, mode=g_mode)
    # 遍历分词结果，打印每个词的信息
    for tk in result:
        print("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2])

# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
```