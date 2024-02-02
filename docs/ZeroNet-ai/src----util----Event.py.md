# `ZeroNet\src\util\Event.py`

```py
# 定义一个名为 Event 的类，继承自 list 类
class Event(list):

    # 定义 __call__ 方法，使实例对象可以像函数一样被调用
    def __call__(self, *args, **kwargs):
        # 遍历实例对象中的函数
        for f in self[:]:
            # 如果函数具有属性 "once"，并且在实例对象中
            if "once" in dir(f) and f in self:
                # 从实例对象中移除该函数
                self.remove(f)
            # 调用函数
            f(*args, **kwargs)

    # 定义 __repr__ 方法，返回实例对象的字符串表示形式
    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)

    # 定义 once 方法，用于向实例对象中添加函数，并标记为只执行一次
    def once(self, func, name=None):
        # 设置函数的属性 once 为 True
        func.once = True
        func.name = None
        # 如果指定了 name 参数
        if name:  
            # 获取实例对象中已经存在的函数的名称列表
            names = [f.name for f in self if "once" in dir(f)]
            # 如果指定的名称不在列表中
            if name not in names:
                # 设置函数的名称属性为指定的名称
                func.name = name
                # 将函数添加到实例对象中
                self.append(func)
        else:
            # 将函数添加到实例对象中
            self.append(func)
        return self


# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 定义一个名为 testBenchmark 的函数
    def testBenchmark():
        # 定义一个名为 say 的函数
        def say(pre, text):
            print("%s Say: %s" % (pre, text))

        # 导入 time 模块
        import time
        # 记录当前时间
        s = time.time()
        # 创建一个 Event 实例对象
        on_changed = Event()
        # 循环执行 1000 次
        for i in range(1000):
            # 向实例对象中添加一个只执行一次的 lambda 函数
            on_changed.once(lambda pre: say(pre, "once"), "once")
        # 打印执行 1000 次的时间
        print("Created 1000 once in %.3fs" % (time.time() - s))
        # 调用实例对象
        on_changed("#1")

    # 定义一个名为 testUsage 的函数
    def testUsage():
        # 定义一个名为 say 的函数
        def say(pre, text):
            print("%s Say: %s" % (pre, text))

        # 创建一个 Event 实例对象
        on_changed = Event()
        # 向实例对象中添加只执行一次的 lambda 函数
        on_changed.once(lambda pre: say(pre, "once"))
        on_changed.once(lambda pre: say(pre, "once"))
        on_changed.once(lambda pre: say(pre, "namedonce"), "namedonce")
        on_changed.once(lambda pre: say(pre, "namedonce"), "namedonce")
        # 向实例对象中添加 lambda 函数
        on_changed.append(lambda pre: say(pre, "always"))
        # 调用实例对象
        on_changed("#1")
        on_changed("#2")
        on_changed("#3")

    # 调用 testBenchmark 函数
    testBenchmark()
```