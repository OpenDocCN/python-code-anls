# `jieba\test\extract_tags_stop_words.py`

```
# 导入 sys 模块
import sys
# 将上级目录添加到 sys.path 中
sys.path.append('../')

# 导入 jieba 分词模块
import jieba
import jieba.analyse
# 导入 OptionParser 类
from optparse import OptionParser

# 定义 USAGE 字符串
USAGE = "usage:    python extract_tags_stop_words.py [file name] -k [top k]"

# 创建 OptionParser 对象，传入 USAGE 字符串
parser = OptionParser(USAGE)
# 添加 -k 选项
parser.add_option("-k", dest="topK")
# 解析命令行参数
opt, args = parser.parse_args()

# 如果参数个数小于 1，打印 USAGE 字符串，退出程序
if len(args) < 1:
    print(USAGE)
    sys.exit(1)

# 获取文件名
file_name = args[0]

# 如果未指定 topK 值，默认为 10，否则转换为整数
if opt.topK is None:
    topK = 10
else:
    topK = int(opt.topK)

# 读取文件内容
content = open(file_name, 'rb').read()

# 设置停用词表路径
jieba.analyse.set_stop_words("../extra_dict/stop_words.txt")
# 设置 IDF 文件路径
jieba.analyse.set_idf_path("../extra_dict/idf.txt.big");

# 提取关键词
tags = jieba.analyse.extract_tags(content, topK=topK)

# 打印关键词列表
print(",".join(tags))
```