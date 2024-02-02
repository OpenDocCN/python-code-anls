# `ZeroNet\src\util\Noparallel.py`

```py
# 导入 gevent 模块
import gevent
# 导入 time 模块
import time
# 从 gevent.event 模块中导入 AsyncResult 类
from gevent.event import AsyncResult
# 从当前目录中导入 ThreadPool 模块
from . import ThreadPool

# 定义 Noparallel 类，用于限制同一时间内只允许函数运行一次
class Noparallel:  

    # 初始化方法
    def __init__(self, blocking=True, ignore_args=False, ignore_class=False, queue=False):
        # 创建线程字典
        self.threads = {}
        # 设置是否阻塞的标志
        self.blocking = blocking  # Blocking: Acts like normal function else thread returned
        # 设置是否队列的标志
        self.queue = queue  # Execute again when blocking is done
        # 设置是否已经加入队列的标志
        self.queued = False
        # 设置是否忽略函数参数的标志
        self.ignore_args = ignore_args  # Block does not depend on function call arguments
        # 设置是否忽略类实例的标志
        self.ignore_class = ignore_class  # Block does not depeds on class instance

    # 清理已完成的线程
    def cleanup(self, key, thread):
        # 如果线程字典中存在指定的键，则删除该键值对
        if key in self.threads:
            del(self.threads[key])

# 如果当前文件被直接运行
if __name__ == "__main__":

    # 定义 Test 类
    class Test():

        # 使用 Noparallel 装饰器，限制同一时间内只允许函数运行一次
        @Noparallel()
        # 定义 count 方法
        def count(self, num=5):
            # 循环 num 次
            for i in range(num):
                # 打印 self 和 i
                print(self, i)
                # 休眠 1 秒
                time.sleep(1)
            # 返回字符串
            return "%s return:%s" % (self, i)

    # 定义 TestNoblock 类
    class TestNoblock():

        # 使用 Noparallel 装饰器，设置为非阻塞模式
        @Noparallel(blocking=False)
        # 定义 count 方法
        def count(self, num=5):
            # 循环 num 次
            for i in range(num):
                # 打印 self 和 i
                print(self, i)
                # 休眠 1 秒
                time.sleep(1)
            # 返回字符串
            return "%s return:%s" % (self, i)
    # 定义一个测试阻塞的函数
    def testBlocking():
        # 创建一个Test类的实例
        test = Test()
        # 创建另一个Test类的实例
        test2 = Test()
        # 打印提示信息
        print("Counting...")
        print("Creating class1/thread1")
        # 创建一个协程来执行test.count()方法
        thread1 = gevent.spawn(test.count)
        print("Creating class1/thread2 (ignored)")
        # 创建另一个协程来执行test.count()方法
        thread2 = gevent.spawn(test.count)
        print("Creating class2/thread3")
        # 创建一个协程来执行test2.count()方法
        thread3 = gevent.spawn(test2.count)
    
        print("Joining class1/thread1")
        # 等待协程thread1执行完毕
        thread1.join()
        print("Joining class1/thread2")
        # 等待协程thread2执行完毕
        thread2.join()
        print("Joining class2/thread3")
        # 等待协程thread3执行完毕
        thread3.join()
    
        print("Creating class1/thread4 (its finished, allowed again)")
        # 创建一个协程来执行test.count()方法
        thread4 = gevent.spawn(test.count)
        print("Joining thread4")
        # 等待协程thread4执行完毕
        thread4.join()
    
        # 打印各个协程的返回值
        print(thread1.value, thread2.value, thread3.value, thread4.value)
        print("Done.")
    
    # 定义一个测试非阻塞的函数
    def testNoblocking():
        # 创建一个TestNoblock类的实例
        test = TestNoblock()
        # 创建另一个TestNoblock类的实例
        test2 = TestNoblock()
        print("Creating class1/thread1")
        # 调用test.count()方法，返回一个协程对象
        thread1 = test.count()
        print("Creating class1/thread2 (ignored)")
        # 调用test.count()方法，返回一个协程对象
        thread2 = test.count()
        print("Creating class2/thread3")
        # 调用test2.count()方法，返回一个协程对象
        thread3 = test2.count()
        print("Joining class1/thread1")
        # 等待协程thread1执行完毕
        thread1.join()
        print("Joining class1/thread2")
        # 等待协程thread2执行完毕
        thread2.join()
        print("Joining class2/thread3")
        # 等待协程thread3执行完毕
        thread3.join()
    
        print("Creating class1/thread4 (its finished, allowed again)")
        # 调用test.count()方法，返回一个协程对象
        thread4 = test.count()
        print("Joining thread4")
        # 等待协程thread4执行完毕
        thread4.join()
    
        # 打印各个协程的返回值
        print(thread1.value, thread2.value, thread3.value, thread4.value)
        print("Done.")
    # 定义一个测试基准性能的函数
    def testBenchmark():
        import time
    
        # 定义一个函数，用于打印当前线程数
        def printThreadNum():
            import gc
            from greenlet import greenlet
            # 获取所有的 greenlet 对象，打印数量
            objs = [obj for obj in gc.get_objects() if isinstance(obj, greenlet)]
            print("Greenlets: %s" % len(objs))
    
        # 打印当前线程数
        printThreadNum()
        # 创建一个 TestNoblock 实例
        test = TestNoblock()
        # 记录当前时间
        s = time.time()
        # 创建三个协程，执行 test.count 方法
        for i in range(3):
            gevent.spawn(test.count, i + 1)
        # 打印创建协程所花费的时间
        print("Created in %.3fs" % (time.time() - s))
        # 打印当前线程数
        printThreadNum()
        # 等待5秒
        time.sleep(5)
    
    # 定义一个测试异常处理的函数
    def testException():
        import time
        # 使用 Noparallel 装饰器定义一个 count 方法
        @Noparallel(blocking=True, queue=True)
        def count(self, num=5):
            s = time.time()
            # 抛出异常
            # raise Exception("err")
            for i in range(num):
                print(self, i)
                time.sleep(1)
            return "%s return:%s" % (s, i)
        # 定义一个调用者函数
        def caller():
            try:
                # 调用 count 方法
                print("Ret:", count(5))
            except Exception as err:
                # 捕获异常并打印
                print("Raised:", repr(err))
    
        # 创建四个协程，分别执行 caller 方法
        gevent.joinall([
            gevent.spawn(caller),
            gevent.spawn(caller),
            gevent.spawn(caller),
            gevent.spawn(caller)
        ])
    
    # 导入 gevent 的 monkey 模块，并对所有模块进行猴子补丁
    from gevent import monkey
    monkey.patch_all()
    
    # 调用 testException 函数
    testException()
    
    """
    testBenchmark()
    print("Testing blocking mode...")
    testBlocking()
    print("Testing noblocking mode...")
    testNoblocking()
    """
```