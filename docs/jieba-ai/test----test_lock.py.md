# `jieba\test\test_lock.py`

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import threading

# 初始化分词器函数，接受分词器对象和组号作为参数
def inittokenizer(tokenizer, group):
    # 打印线程开始信息，包括组号和当前线程标识
    print('===> Thread %s:%s started' % (group, threading.current_thread().ident))
    # 初始化分词器
    tokenizer.initialize()
    # 打印线程结束信息，包括组号和当前线程标识
    print('<=== Thread %s:%s finished' % (group, threading.current_thread().ident)

# 创建5个默认分词器对象列表
tokrs1 = [jieba.Tokenizer() for n in range(5)]
# 创建5个指定额外词典的分词器对象列表
tokrs2 = [jieba.Tokenizer('../extra_dict/dict.txt.small') for n in range(5)]

# 创建针对tokrs1列表中每个分词器对象的线程列表
thr1 = [threading.Thread(target=inittokenizer, args=(tokr, 1)) for tokr in tokrs1]
# 创建针对tokrs2列表中每个分词器对象的线程列表
thr2 = [threading.Thread(target=inittokenizer, args=(tokr, 2)) for tokr in tokrs2]
# 启动thr1列表中的所有线程
for thr in thr1:
    thr.start()
# 启动thr2列表中的所有线程
for thr in thr2:
    thr.start()
# 等待thr1列表中的所有线程结束
for thr in thr1:
    thr.join()
# 等待thr2列表中的所有线程结束
for thr in thr2:
    thr.join()

# 释放tokrs1和tokrs2列表
del tokrs1, tokrs2

# 打印分隔线
print('='*40)

# 创建默认分词器对象
tokr1 = jieba.Tokenizer()
# 创建指定额外词典的分词器对象
tokr2 = jieba.Tokenizer('../extra_dict/dict.txt.small')

# 创建针对tokr1对象的线程列表
thr1 = [threading.Thread(target=inittokenizer, args=(tokr1, 1)) for n in range(5)]
# 创建针对tokr2对象的线程列表
thr2 = [threading.Thread(target=inittokenizer, args=(tokr2, 2)) for n in range(5)]
# 启动thr1列表中的所有线程
for thr in thr1:
    thr.start()
# 启动thr2列表中的所有线程
for thr in thr2:
    thr.start()
# 等待thr1列表中的所有线程结束
for thr in thr1:
    thr.join()
# 等待thr2列表中的所有线程结束
for thr in thr2:
    thr.join()
```