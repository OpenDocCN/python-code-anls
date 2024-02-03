# `jieba\test\parallel\test_file.py`

```
# 导入 sys 模块，用于处理命令行参数和系统路径
import sys
# 导入 time 模块，用于时间相关操作
import time
# 将上级目录添加到 Python 搜索路径中
sys.path.append("../../")
# 导入 jieba 模块，用于中文分词
import jieba

# 启用 jieba 并行分词功能
jieba.enable_parallel()

# 从命令行参数中获取 URL
url = sys.argv[1]
# 读取 URL 对应文件的内容
content = open(url,"rb").read()
# 记录开始时间
t1 = time.time()
# 使用 jieba 分词并用空格连接结果
words = "/ ".join(jieba.cut(content))

# 记录结束时间
t2 = time.time()
# 计算分词耗时
tm_cost = t2-t1

# 打开文件 "1.log" 以二进制写入模式
log_f = open("1.log","wb")
# 将分词结果写入文件并编码为 UTF-8
log_f.write(words.encode('utf-8'))

# 打印分词速度
print('speed %s bytes/second' % (len(content)/tm_cost))
```