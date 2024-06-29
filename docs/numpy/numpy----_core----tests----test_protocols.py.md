# `.\numpy\numpy\_core\tests\test_protocols.py`

```
# 导入 pytest 库，用于编写和运行测试用例
import pytest
# 导入 warnings 库，用于处理警告信息
import warnings
# 导入 numpy 库，并使用 np 别名引用
import numpy as np

# 使用 pytest.mark.filterwarnings("error") 装饰器标记此测试用例，将警告信息转换为错误
@pytest.mark.filterwarnings("error")
# 定义一个测试函数 test_getattr_warning，用于测试 __getattr__ 方法的警告行为
def test_getattr_warning():
    # 解决问题 gh-14735：确保仅清除 getattr 错误，并允许警告通过
    class Wrapper:
        # 初始化方法，接受一个数组作为参数
        def __init__(self, array):
            self.array = array
        
        # 定义 __len__ 方法，返回数组的长度
        def __len__(self):
            return len(self.array)
        
        # 定义 __getitem__ 方法，用于获取数组的元素
        def __getitem__(self, item):
            return type(self)(self.array[item])
        
        # 定义 __getattr__ 方法，处理动态属性访问
        def __getattr__(self, name):
            # 如果属性名以 "__array_" 开头，发出警告
            if name.startswith("__array_"):
                warnings.warn("object got converted", UserWarning, stacklevel=1)
            # 否则委托给数组对象的相应属性
            return getattr(self.array, name)
        
        # 定义 __repr__ 方法，返回对象的字符串表示形式
        def __repr__(self):
            return "<Wrapper({self.array})>".format(self=self)
    
    # 创建一个 Wrapper 实例，包装一个包含 0 到 9 的 numpy 数组
    array = Wrapper(np.arange(10))
    # 使用 pytest.raises 检测是否抛出特定类型和消息的警告
    with pytest.raises(UserWarning, match="object got converted"):
        np.asarray(array)

# 定义一个测试函数 test_array_called，测试自定义类的 __array__ 方法
def test_array_called():
    # 定义一个名为 Wrapper 的类
    class Wrapper:
        # 类属性，包含一个长度为 100 的字符串 '0'
        val = '0' * 100
        
        # 定义 __array__ 方法，返回一个包含 self.val 的 numpy 数组
        def __array__(self, dtype=None, copy=None):
            return np.array([self.val], dtype=dtype, copy=copy)
    
    # 创建一个 Wrapper 实例
    wrapped = Wrapper()
    # 将 Wrapper 实例转换为 numpy 数组，指定 dtype 为字符串
    arr = np.array(wrapped, dtype=str)
    # 断言 numpy 数组的 dtype 是否符合预期
    assert arr.dtype == 'U100'
    # 断言 numpy 数组的第一个元素是否与 Wrapper 类的 val 属性相等
    assert arr[0] == Wrapper.val
```