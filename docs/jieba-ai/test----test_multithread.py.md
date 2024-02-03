# `jieba\test\test_multithread.py`

```
# 设置文件编码为utf-8
# 导入sys模块
# 导入threading模块
sys.path.append("../")  # 将上级目录添加到模块搜索路径中

# 导入jieba模块
import jieba

# 定义Worker类，继承自threading.Thread类
class Worker(threading.Thread):
    # 重写run方法
    def run(self):
        # 使用全模式对文本进行分词
        seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
        print("Full Mode:" + "/ ".join(seg_list))  # 输出分词结果，全模式

        # 使用默认模式对文本进行分词
        seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
        print("Default Mode:" + "/ ".join(seg_list))  # 输出分词结果，默认模式

        # 对文本进行分词
        seg_list = jieba.cut("他来到了网易杭研大厦")
        print(", ".join(seg_list))  # 输出分词结果

        # 使用搜索引擎模式对文本进行分词
        seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
        print(", ".join(seg_list))  # 输出分词结果，搜索引擎模式

# 创建Worker对象列表
workers = []
# 循环创建10个Worker对象并启动线程
for i in range(10):
    worker = Worker()
    workers.append(worker)
    worker.start()

# 等待所有线程执行完毕
for worker in workers:
    worker.join()
```