# `jieba\test\test_pos_file.py`

```
# 导入 Python 未来支持的 print 函数
from __future__ import print_function
# 导入 sys 模块
import sys
# 导入 time 模块
import time
# 将上级目录添加到 sys.path 中
sys.path.append("../")
# 导入 jieba 模块
import jieba
# 初始化 jieba
jieba.initialize()
# 导入 jieba.posseg 模块并重命名为 pseg
import jieba.posseg as pseg

# 从命令行参数中获取 URL
url = sys.argv[1]
# 读取 URL 对应文件的内容
content = open(url,"rb").read()
# 记录开始时间
t1 = time.time()
# 对内容进行分词，返回词性标注结果
words = list(pseg.cut(content))

# 记录结束时间
t2 = time.time()
# 计算时间消耗
tm_cost = t2-t1

# 打开文件 "1.log" 以写入模式
log_f = open("1.log","w")
# 将词性标注结果转换为字符串，用 ' / ' 连接，写入文件
log_f.write(' / '.join(map(str, words)))

# 打印速度信息，即每秒处理的字节数
print('speed' , len(content)/tm_cost, " bytes/second")
```