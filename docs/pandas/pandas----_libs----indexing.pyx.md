# `D:\src\scipysrc\pandas\pandas\_libs\indexing.pyx`

```
# 定义一个 Cython 扩展类型 NDFrameIndexerBase，作为 _NDFrameIndexer 的基类，用于快速实例化和属性访问。
cdef class NDFrameIndexerBase:
    """
    A base class for _NDFrameIndexer for fast instantiation and attribute access.
    """
    # 定义 Cython 特有的变量声明部分
    cdef:
        # 声明一个 Py_ssize_t 类型的实例变量 _ndim，用于存储维度信息
        Py_ssize_t _ndim

    # 定义公共的 Python 类型实例变量
    cdef public:
        # 字符串类型的 name 属性，表示索引器的名称
        str name
        # Python 对象类型的 obj 属性，表示索引器操作的对象

    # 类的初始化方法，接受一个字符串 name 和一个对象 obj 作为参数
    def __init__(self, name: str, obj):
        # 将传入的 obj 参数赋值给实例属性 self.obj
        self.obj = obj
        # 将传入的 name 参数赋值给实例属性 self.name
        self.name = name
        # 初始化 _ndim 属性为 -1，表示维度信息尚未确定
        self._ndim = -1

    # 定义一个属性方法 ndim，返回对象的维度信息
    @property
    def ndim(self) -> int:
        # 尝试获取已缓存的维度信息
        ndim = self._ndim
        # 如果 _ndim 为 -1，即维度信息尚未确定
        if ndim == -1:
            # 从 self.obj 中获取维度信息
            ndim = self._ndim = self.obj.ndim
            # 如果获取的维度信息大于 2
            if ndim > 2:
                # 抛出 ValueError 异常，指示 NDFrameIndexer 不支持 ndim 大于 2 的 NDFrame 对象
                raise ValueError(
                    "NDFrameIndexer does not support NDFrame objects with ndim > 2"
                )
        # 返回获取到的维度信息
        return ndim
```