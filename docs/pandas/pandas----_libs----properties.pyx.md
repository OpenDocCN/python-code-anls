# `D:\src\scipysrc\pandas\pandas\_libs\properties.pyx`

```
# 导入必要的Cython库中的特定函数和类型
from cpython.dict cimport (
    PyDict_Contains,  # 导入字典操作中的成员存在性检查函数
    PyDict_GetItem,   # 导入字典操作中的获取成员函数
    PyDict_SetItem,   # 导入字典操作中的设置成员函数
)
from cython cimport Py_ssize_t  # 导入Cython中的整数类型


cdef class CachedProperty:
    # 定义一个Cython类 CachedProperty

    cdef readonly:
        object fget, name, __doc__  # 定义只读成员变量：获取函数、名称、文档字符串

    def __init__(self, fget):
        # 初始化方法，接受一个获取函数作为参数并存储其信息
        self.fget = fget  # 存储获取函数
        self.name = fget.__name__  # 存储获取函数的名称
        self.__doc__ = getattr(fget, "__doc__", None)  # 获取获取函数的文档字符串，如果没有则为None

    def __get__(self, obj, typ):
        # 获取方法，用于获取属性值

        if obj is None:
            # 如果在类上访问，而不是实例上
            return self

        # 获取缓存或者如果需要的话设置一个默认的缓存
        cache = getattr(obj, "_cache", None)  # 尝试获取对象的缓存属性
        if cache is None:
            try:
                cache = obj._cache = {}  # 如果没有缓存则创建一个空字典作为缓存
            except (AttributeError):
                return self  # 如果发生属性错误则返回自身

        if PyDict_Contains(cache, self.name):
            # 检查缓存中是否存在指定的属性名
            # 不需要增加对象的引用计数
            val = <object>PyDict_GetItem(cache, self.name)  # 从缓存中获取属性值
        else:
            val = self.fget(obj)  # 否则调用获取函数获取属性值
            PyDict_SetItem(cache, self.name, val)  # 将属性名和对应的属性值存入缓存

        return val  # 返回获取到的属性值

    def __set__(self, obj, value):
        # 设置方法，阻止设置属性的操作
        raise AttributeError("Can't set attribute")  # 抛出属性错误，禁止设置属性


cache_readonly = CachedProperty  # 设置一个别名，用于表示只读的缓存属性


cdef class AxisProperty:
    # 定义一个Cython类 AxisProperty

    cdef readonly:
        Py_ssize_t axis  # 定义只读成员变量：整数类型的轴
        object __doc__  # 定义只读成员变量：文档字符串

    def __init__(self, axis=0, doc=""):
        # 初始化方法，接受轴和文档字符串作为参数并存储其信息
        self.axis = axis  # 存储轴
        self.__doc__ = doc  # 存储文档字符串

    def __get__(self, obj, type):
        # 获取方法，用于获取属性值

        cdef:
            list axes  # 声明一个本地变量：轴列表

        if obj is None:
            # 如果在类上访问，而不是实例上
            return self

        else:
            axes = obj._mgr.axes  # 获取对象的管理器中的轴列表

        return axes[self.axis]  # 返回指定轴的值

    def __set__(self, obj, value):
        # 设置方法，用于设置属性值
        obj._set_axis(self.axis, value)  # 调用对象的设置轴方法设置属性值
```