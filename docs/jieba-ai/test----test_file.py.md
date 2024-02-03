# `jieba\test\test_file.py`

```
# 导入时间模块
import time
# 导入系统模块
import sys
# 将上级目录添加到系统路径中
sys.path.append("../")
# 导入结巴分词模块
import jieba
# 初始化结巴分词

jieba.initialize()

# 从命令行参数中获取 URL
url = sys.argv[1]
# 读取 URL 对应文件的内容
content = open(url,"rb").read()
# 记录开始时间
t1 = time.time()
# 使用结巴分词对内容进行分词处理
words = "/ ".join(jieba.cut(content))

# 记录结束时间
t2 = time.time()
# 计算处理时间
tm_cost = t2-t1

# 打开日志文件，以二进制写入模式
log_f = open("1.log","wb")
# 将分词结果写入日志文件，使用 UTF-8 编码
log_f.write(words.encode('utf-8'))
# 关闭日志文件
log_f.close()

# 打印处理时间
print('cost ' + str(tm_cost))
# 打印处理速度
print('speed %s bytes/second' % (len(content)/tm_cost))
```