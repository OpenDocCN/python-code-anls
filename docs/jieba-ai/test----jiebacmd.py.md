# `jieba\test\jiebacmd.py`

```py
'''
usage example (find top 100 words in abc.txt):

cat abc.txt | python jiebacmd.py | sort | uniq -c | sort -nr -k1 | head -100


'''

# 导入 Python 2/3 兼容模块
from __future__ import unicode_literals
# 导入 sys 模块
import sys
# 将上级目录添加到 sys.path 中
sys.path.append("../")

# 导入 jieba 分词模块
import jieba

# 默认编码为 utf-8
default_encoding='utf-8'

# 如果命令行参数数量大于 1，则将第一个参数作为默认编码
if len(sys.argv)>1:
    default_encoding = sys.argv[1]

# 读取标准输入流的每一行数据
while True:
    line = sys.stdin.readline()
    # 如果读取到空行，则跳出循环
    if line=="":
        break
    # 去除行两端的空白字符
    line = line.strip()
    # 使用 jieba 分词对每一行进行分词
    for word in jieba.cut(line):
        # 打印分词结果
        print(word)
```