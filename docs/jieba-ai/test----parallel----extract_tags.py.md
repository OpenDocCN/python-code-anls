# `jieba\test\parallel\extract_tags.py`

```
# 导入 sys 模块
import sys
# 将相对路径添加到系统路径中
sys.path.append('../../')

# 导入 jieba 分词模块
import jieba
# 启用并行分词，设置并行数为 4
jieba.enable_parallel(4)
# 导入 jieba.analyse 模块
import jieba.analyse
# 导入 OptionParser 类
from optparse import OptionParser

# 定义用法提示信息
USAGE ="usage:    python extract_tags.py [file name] -k [top k]"

# 创建 OptionParser 对象，传入用法提示信息
parser = OptionParser(USAGE)
# 添加一个选项 -k，指定参数的名称为 topK
parser.add_option("-k",dest="topK")
# 解析命令行参数，将结果保存到 opt 和 args 中
opt, args = parser.parse_args()

# 如果参数个数小于 1，则打印用法提示信息并退出程序
if len(args) <1:
    print(USAGE)
    sys.exit(1)

# 获取文件名参数
file_name = args[0]

# 如果未指定 topK 参数，则默认为 10，否则将其转换为整数
if opt.topK==None:
    topK=10
else:
    topK = int(opt.topK)

# 读取文件内容
content = open(file_name,'rb').read()

# 使用 jieba.analyse 模块提取关键词，指定 topK 参数
tags = jieba.analyse.extract_tags(content,topK=topK)

# 将提取的关键词以逗号分隔输出
print(",".join(tags))
```