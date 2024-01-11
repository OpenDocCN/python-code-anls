# `ZeroNet\src\Test\TestEvent.py`

```
# 导入名为 util 的模块
import util

# 定义一个名为 ExampleClass 的类，继承自 object 类
class ExampleClass(object):
    # 初始化方法
    def __init__(self):
        # 创建一个空列表 called
        self.called = []
        # 创建一个名为 onChanged 的事件对象
        self.onChanged = util.Event()

    # 定义一个名为 increment 的方法，接受参数 title
    def increment(self, title):
        # 将 title 添加到 called 列表中
        self.called.append(title)

# 定义一个名为 TestEvent 的类
class TestEvent:
    # 定义一个名为 testEvent 的方法
    def testEvent(self):
        # 创建一个 ExampleClass 的实例对象 test_obj
        test_obj = ExampleClass()
        # 向 onChanged 事件对象添加一个匿名函数，调用 increment 方法并传入参数 "Called #1"
        test_obj.onChanged.append(lambda: test_obj.increment("Called #1"))
        # 向 onChanged 事件对象添加一个匿名函数，调用 increment 方法并传入参数 "Called #2"
        test_obj.onChanged.append(lambda: test_obj.increment("Called #2"))
        # 向 onChanged 事件对象添加一个匿名函数，调用 increment 方法并传入参数 "Once"
        test_obj.onChanged.once(lambda: test_obj.increment("Once"))

        # 断言 test_obj.called 为空列表
        assert test_obj.called == []
        # 调用 onChanged 事件对象
        test_obj.onChanged()
        # 断言 test_obj.called 包含 ["Called #1", "Called #2", "Once"]
        assert test_obj.called == ["Called #1", "Called #2", "Once"]
        # 再次调用 onChanged 事件对象
        test_obj.onChanged()
        # 再次调用 onChanged 事件对象
        test_obj.onChanged()
        # 断言 test_obj.called 包含 ["Called #1", "Called #2", "Once", "Called #1", "Called #2", "Called #1", "Called #2"]
        assert test_obj.called == ["Called #1", "Called #2", "Once", "Called #1", "Called #2", "Called #1", "Called #2"]

    # 定义一个名为 testOnce 的方法
    def testOnce(self):
        # 创建一个 ExampleClass 的实例对象 test_obj
        test_obj = ExampleClass()
        # 向 onChanged 事件对象添加一个匿名函数，调用 increment 方法并传入参数 "Once test #1"
        test_obj.onChanged.once(lambda: test_obj.increment("Once test #1"))

        # 断言 test_obj.called 为空列表
        assert test_obj.called == []
        # 调用 onChanged 事件对象
        test_obj.onChanged()
        # 断言 test_obj.called 包含 ["Once test #1"]
        assert test_obj.called == ["Once test #1"]
        # 再次调用 onChanged 事件对象
        test_obj.onChanged()
        # 再次调用 onChanged 事件对象
        test_obj.onChanged()
        # 断言 test_obj.called 包含 ["Once test #1"]
        assert test_obj.called == ["Once test #1"]

    # 定义一个名为 testOnceMultiple 的方法
    def testOnceMultiple(self):
        # 创建一个 ExampleClass 的实例对象 test_obj
        test_obj = ExampleClass()
        # 向 onChanged 事件对象添加一个匿名函数，调用 increment 方法并传入参数 "Once test #1"
        test_obj.onChanged.once(lambda: test_obj.increment("Once test #1"))
        # 向 onChanged 事件对象添加一个匿名函数，调用 increment 方法并传入参数 "Once test #2"
        test_obj.onChanged.once(lambda: test_obj.increment("Once test #2"))
        # 向 onChanged 事件对象添加一个匿名函数，调用 increment 方法并传入参数 "Once test #3"
        test_obj.onChanged.once(lambda: test_obj.increment("Once test #3"))

        # 断言 test_obj.called 为空列表
        assert test_obj.called == []
        # 调用 onChanged 事件对象
        test_obj.onChanged()
        # 断言 test_obj.called 包含 ["Once test #1", "Once test #2", "Once test #3"]
        assert test_obj.called == ["Once test #1", "Once test #2", "Once test #3"]
        # 再次调用 onChanged 事件对象
        test_obj.onChanged()
        # 再次调用 onChanged 事件对象
        test_obj.onChanged()
        # 断言 test_obj.called 包含 ["Once test #1", "Once test #2", "Once test #3"]
        assert test_obj.called == ["Once test #1", "Once test #2", "Once test #3"]
    # 定义一个测试方法，用于测试一次性事件处理
    def testOnceNamed(self):
        # 创建一个示例对象
        test_obj = ExampleClass()
        # 为相同类型的事件添加一次性处理器，确保同一类型只存储一个处理器
        test_obj.onChanged.once(lambda: test_obj.increment("Once test #1/1"), "type 1")
        test_obj.onChanged.once(lambda: test_obj.increment("Once test #1/2"), "type 1")
        test_obj.onChanged.once(lambda: test_obj.increment("Once test #2"), "type 2")

        # 断言调用列表为空
        assert test_obj.called == []
        # 触发事件
        test_obj.onChanged()
        # 断言调用列表包含指定的处理器结果
        assert test_obj.called == ["Once test #1/1", "Once test #2"]
        # 再次触发事件
        test_obj.onChanged()
        test_obj.onChanged()
        # 断言调用列表不会再添加重复的处理器结果
        assert test_obj.called == ["Once test #1/1", "Once test #2"]
```