# `D:\src\scipysrc\scikit-learn\sklearn\utils\_bunch.py`

```
import warnings  # 导入警告模块


class Bunch(dict):
    """Container object exposing keys as attributes.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples
    --------
    >>> from sklearn.utils import Bunch
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

        # Map from deprecated key to warning message
        self.__dict__["_deprecated_key_to_warnings"] = {}  # 初始化存储废弃键和警告信息的字典

    def __getitem__(self, key):
        # 如果键在废弃键警告字典中，则发出警告
        if key in self.__dict__.get("_deprecated_key_to_warnings", {}):
            warnings.warn(
                self._deprecated_key_to_warnings[key],  # 发出警告信息
                FutureWarning,
            )
        return super().__getitem__(key)  # 调用父类的方法获取键对应的值

    def _set_deprecated(self, value, *, new_key, deprecated_key, warning_message):
        """Set key in dictionary to be deprecated with its warning message."""
        self.__dict__["_deprecated_key_to_warnings"][deprecated_key] = warning_message  # 存储废弃键和警告信息
        self[new_key] = self[deprecated_key] = value  # 设置新键和废弃键的值为相同的值

    def __setattr__(self, key, value):
        self[key] = value  # 设置属性时，调用字典的设置方法

    def __dir__(self):
        return self.keys()  # 返回所有键的列表作为对象的属性

    def __getattr__(self, key):
        try:
            return self[key]  # 获取属性时，尝试通过键获取值
        except KeyError:
            raise AttributeError(key)  # 如果键不存在，则抛出属性错误异常

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass  # 设置 __setstate__ 方法为空操作，忽略被 pickle 的 __dict__
```