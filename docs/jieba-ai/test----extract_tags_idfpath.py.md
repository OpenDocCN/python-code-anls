# `jieba\test\extract_tags_idfpath.py`

```
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
USAGE = "usage:    python extract_tags_idfpath.py [file name] -k [top k]"

# 创建 OptionParser 对象，传入 USAGE 字符串
parser = OptionParser(USAGE)
# 添加 -k 选项
parser.add_option("-k", dest="topK")
# 解析命令行参数
opt, args = parser.parse_args()

# 如果参数个数小于 1
if len(args) < 1:
    # 打印 USAGE 字符串
    print(USAGE)
    # 退出程序
    sys.exit(1)

# 获取文件名
file_name = args[0]

# 如果 topK 选项为 None
if opt.topK is None:
    # 设置 topK 为 10
    topK = 10
else:
    # 将 topK 转换为整数
    topK = int(opt.topK)

# 读取文件内容
content = open(file_name, 'rb').read()

# 设置 IDF 文件路径
jieba.analyse.set_idf_path("../extra_dict/idf.txt.big");

# 提取关键词标签
tags = jieba.analyse.extract_tags(content, topK=topK)

# 打印关键词标签，用逗号分隔
print(",".join(tags))
```