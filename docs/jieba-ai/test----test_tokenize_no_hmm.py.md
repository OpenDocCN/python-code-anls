# `jieba\test\test_tokenize_no_hmm.py`

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
    # 声明全局变量g_mode
    global g_mode
    # 使用jieba模块的tokenize函数对输入的句子进行分词，设置分词模式为g_mode，关闭HMM
    result = jieba.tokenize(test_sent, mode=g_mode, HMM=False)
    # 遍历分词结果
    for tk in result:
        # 打印每个分词的信息，包括词语、起始位置和结束位置
        print("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))

# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
```