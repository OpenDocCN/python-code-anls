# `jieba\test\extract_tags_with_weight.py`

```py
# 导入 sys 模块
import sys
# 将上级目录添加到 sys.path 中
sys.path.append('../')

# 导入 jieba 模块
import jieba
# 导入 jieba.analyse 模块
import jieba.analyse
# 导入 OptionParser 类
from optparse import OptionParser

# 定义 USAGE 字符串
USAGE = "usage:    python extract_tags_with_weight.py [file name] -k [top k] -w [with weight=1 or 0]"

# 创建 OptionParser 对象
parser = OptionParser(USAGE)
# 添加 -k 选项
parser.add_option("-k", dest="topK")
# 添加 -w 选项
parser.add_option("-w", dest="withWeight")
# 解析命令行参数
opt, args = parser.parse_args()

# 如果参数个数小于 1，打印 USAGE 字符串并退出
if len(args) < 1:
    print(USAGE)
    sys.exit(1)

# 获取文件名
file_name = args[0]

# 如果未指定 topK 参数，默认为 10，否则转换为整数
if opt.topK is None:
    topK = 10
else:
    topK = int(opt.topK)

# 如果未指定 withWeight 参数，默认为 False，否则根据参数值设置 withWeight
if opt.withWeight is None:
    withWeight = False
else:
    if int(opt.withWeight) is 1:
        withWeight = True
    else:
        withWeight = False

# 读取文件内容
content = open(file_name, 'rb').read()

# 使用 jieba.analyse.extract_tags 提取关键词
tags = jieba.analyse.extract_tags(content, topK=topK, withWeight=withWeight)

# 如果 withWeight 为 True，打印关键词和权重
if withWeight is True:
    for tag in tags:
        print("tag: %s\t\t weight: %f" % (tag[0],tag[1]))
# 否则，打印关键词列表
else:
    print(",".join(tags))
```