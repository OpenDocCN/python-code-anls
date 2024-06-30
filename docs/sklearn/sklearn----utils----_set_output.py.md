# `D:\src\scipysrc\scikit-learn\sklearn\utils\_set_output.py`

```
# 导入模块 importlib，用于动态导入库
import importlib
# 导入 wraps 函数，用于装饰器相关操作
from functools import wraps
# 导入 Protocol 和 runtime_checkable 用于定义协议和运行时类型检查
from typing import Protocol, runtime_checkable

# 导入 numpy 库，并指定别名 np
import numpy as np
# 导入 issparse 函数，用于检查是否为稀疏矩阵
from scipy.sparse import issparse

# 导入 get_config 函数，用于获取配置信息
from .._config import get_config
# 导入 available_if 函数，用于条件判断
from ._available_if import available_if


# 定义函数 check_library_installed，检查指定库是否安装并导入
def check_library_installed(library):
    """Check library is installed."""
    try:
        return importlib.import_module(library)
    except ImportError as exc:
        # 如果导入失败，抛出 ImportError，并提示需要安装该库
        raise ImportError(
            f"Setting output container to '{library}' requires {library} to be"
            " installed"
        ) from exc


# 定义函数 get_columns，获取列名信息
def get_columns(columns):
    if callable(columns):
        try:
            # 如果 columns 是可调用对象，则调用它获取列名
            return columns()
        except Exception:
            # 调用过程中出现异常，则返回 None
            return None
    # 如果 columns 不是可调用对象，则直接返回 columns
    return columns


# 使用 runtime_checkable 装饰器定义 ContainerAdapterProtocol 协议
@runtime_checkable
class ContainerAdapterProtocol(Protocol):
    # 定义协议属性 container_lib，表示容器类型的库名
    container_lib: str

    # 定义方法 create_container，创建容器并添加元数据
    def create_container(self, X_output, X_original, columns, inplace=False):
        """Create container from `X_output` with additional metadata.

        Parameters
        ----------
        X_output : {ndarray, dataframe}
            Data to wrap.

        X_original : {ndarray, dataframe}
            Original input dataframe. This is used to extract the metadata that should
            be passed to `X_output`, e.g. pandas row index.

        columns : callable, ndarray, or None
            The column names or a callable that returns the column names. The
            callable is useful if the column names require some computation. If `None`,
            then no columns are passed to the container's constructor.

        inplace : bool, default=False
            Whether or not we intend to modify `X_output` in-place. However, it does
            not guarantee that we return the same object if the in-place operation
            is not possible.

        Returns
        -------
        wrapped_output : container_type
            `X_output` wrapped into the container type.
        """

    # 定义方法 is_supported_container，检查 X 是否为支持的容器类型
    def is_supported_container(self, X):
        """Return True if X is a supported container.

        Parameters
        ----------
        Xs: container
            Containers to be checked.

        Returns
        -------
        is_supported_container : bool
            True if X is a supported container.
        """

    # 定义方法 rename_columns，重命名容器 X 的列名
    def rename_columns(self, X, columns):
        """Rename columns in `X`.

        Parameters
        ----------
        X : container
            Container which columns is updated.

        columns : ndarray of str
            Columns to update the `X`'s columns with.

        Returns
        -------
        updated_container : container
            Container with new names.
        """

    # 定义方法 hstack，水平堆叠容器 Xs
    def hstack(self, Xs):
        """Stack containers horizontally (column-wise).

        Parameters
        ----------
        Xs : list of containers
            List of containers to stack.

        Returns
        -------
        stacked_Xs : container
            Stacked containers.
        """


# 定义 PandasAdapter 类，实现 ContainerAdapterProtocol 协议
class PandasAdapter:
    # 设置 container_lib 属性为 "pandas"
    container_lib = "pandas"
    # 创建容器函数，根据参数和条件生成新的数据容器或在原地操作
    def create_container(self, X_output, X_original, columns, inplace=True):
        # 检查并获取 pandas 库
        pd = check_library_installed("pandas")
        # 规范化列名输入
        columns = get_columns(columns)

        # 如果不是原地操作或者 X_output 不是 DataFrame 类型，需要创建新的 DataFrame

        # 如果 X_output 是 DataFrame 类型，则使用其索引
        if isinstance(X_output, pd.DataFrame):
            index = X_output.index
        # 如果 X_original 是 DataFrame 类型，则使用其索引
        elif isinstance(X_original, pd.DataFrame):
            index = X_original.index
        else:
            index = None

        # 创建新的 DataFrame，不传递列名，以免意外进行列选择而不是重命名
        X_output = pd.DataFrame(X_output, index=index, copy=not inplace)

        # 如果指定了 columns 参数，则重命名列名
        if columns is not None:
            return self.rename_columns(X_output, columns)
        # 否则直接返回 X_output
        return X_output

    # 判断传入的对象 X 是否为 pandas 的 DataFrame 类型
    def is_supported_container(self, X):
        # 检查并获取 pandas 库
        pd = check_library_installed("pandas")
        return isinstance(X, pd.DataFrame)

    # 重命名 DataFrame 的列名为指定的 columns
    def rename_columns(self, X, columns):
        # 直接赋值给 DataFrame 的 columns 属性来重命名列名
        X.columns = columns
        return X

    # 使用 pandas 库的 concat 函数将列表 Xs 中的 DataFrame 沿着列方向堆叠
    def hstack(self, Xs):
        # 检查并获取 pandas 库
        pd = check_library_installed("pandas")
        return pd.concat(Xs, axis=1)
class PolarsAdapter:
    # 定义适配器使用的容器库名称为 "polars"
    container_lib = "polars"

    # 创建容器对象的方法，根据参数决定是新建DataFrame还是重命名列名
    def create_container(self, X_output, X_original, columns, inplace=True):
        # 检查并获取 polars 库
        pl = check_library_installed("polars")
        # 获取列名列表
        columns = get_columns(columns)
        # 如果列名是 ndarray 类型，则转换为列表
        columns = columns.tolist() if isinstance(columns, np.ndarray) else columns

        # 如果不是 inplace 操作或者 X_output 不是 pl.DataFrame 类型，需要新建一个 DataFrame
        if not inplace or not isinstance(X_output, pl.DataFrame):
            return pl.DataFrame(X_output, schema=columns, orient="row")

        # 如果指定了列名，则调用重命名列名方法
        if columns is not None:
            return self.rename_columns(X_output, columns)
        # 否则直接返回 X_output
        return X_output

    # 检查传入的容器对象是否是支持的类型（pl.DataFrame）
    def is_supported_container(self, X):
        pl = check_library_installed("polars")
        return isinstance(X, pl.DataFrame)

    # 对传入的容器对象 X 重命名列名为 columns
    def rename_columns(self, X, columns):
        X.columns = columns
        return X

    # 将多个容器对象水平连接
    def hstack(self, Xs):
        pl = check_library_installed("polars")
        return pl.concat(Xs, how="horizontal")


class ContainerAdaptersManager:
    # 定义适配器管理器，初始化空字典 adapters
    def __init__(self):
        self.adapters = {}

    # 返回所有支持的输出容器类型，包括 "default" 和已注册的适配器名称
    @property
    def supported_outputs(self):
        return {"default"} | set(self.adapters)

    # 向适配器管理器注册新的适配器
    def register(self, adapter):
        self.adapters[adapter.container_lib] = adapter


# 创建全局适配器管理器对象
ADAPTERS_MANAGER = ContainerAdaptersManager()
# 向适配器管理器注册 PandasAdapter 和 PolarsAdapter 适配器
ADAPTERS_MANAGER.register(PandasAdapter())
ADAPTERS_MANAGER.register(PolarsAdapter())


def _get_adapter_from_container(container):
    """Get the adapter that knows how to handle such container.

    See :class:`sklearn.utils._set_output.ContainerAdapterProtocol` for more
    details.
    """
    # 根据容器对象的模块名获取对应的适配器
    module_name = container.__class__.__module__.split(".")[0]
    try:
        return ADAPTERS_MANAGER.adapters[module_name]
    except KeyError as exc:
        # 如果未找到对应的适配器，抛出错误并提供可用的适配器列表
        available_adapters = list(ADAPTERS_MANAGER.adapters.keys())
        raise ValueError(
            "The container does not have a registered adapter in scikit-learn. "
            f"Available adapters are: {available_adapters} while the container "
            f"provided is: {container!r}."
        ) from exc


def _get_container_adapter(method, estimator=None):
    """Get container adapter."""
    # 获取稠密输出配置
    dense_config = _get_output_config(method, estimator)["dense"]
    try:
        # 根据稠密输出配置获取对应的适配器
        return ADAPTERS_MANAGER.adapters[dense_config]
    except KeyError:
        return None


def _get_output_config(method, estimator=None):
    """Get output config based on estimator and global configuration.

    Parameters
    ----------
    method : {"transform"}
        Estimator's method for which the output container is looked up.

    estimator : estimator instance or None
        Estimator to get the output configuration from. If `None`, check global
        configuration is used.

    Returns
    -------
    """
    # 获取 estimator 对象的 "_sklearn_output_config" 属性，如果不存在则返回空字典
    est_sklearn_output_config = getattr(estimator, "_sklearn_output_config", {})
    # 检查 method 是否在 _sklearn_output_config 中，若存在则取其对应的值，否则从配置中获取 method_output 的值
    if method in est_sklearn_output_config:
        dense_config = est_sklearn_output_config[method]
    else:
        dense_config = get_config()[f"{method}_output"]

    # 获取 ADAPTERS_MANAGER 支持的输出配置列表
    supported_outputs = ADAPTERS_MANAGER.supported_outputs
    # 如果 dense_config 不在 supported_outputs 中，则抛出 ValueError 异常
    if dense_config not in supported_outputs:
        raise ValueError(
            f"output config must be in {sorted(supported_outputs)}, got {dense_config}"
        )

    # 返回包含 "dense" 键的字典，其值为 dense_config
    return {"dense": dense_config}
def _wrap_data_with_container(method, data_to_wrap, original_input, estimator):
    """Wrap output with container based on an estimator's or global config.

    Parameters
    ----------
    method : {"transform"}
        Estimator's method to get container output for.

    data_to_wrap : {ndarray, dataframe}
        Data to wrap with container.

    original_input : {ndarray, dataframe}
        Original input of function.

    estimator : estimator instance
        Estimator with to get the output configuration from.

    Returns
    -------
    output : {ndarray, dataframe}
        If the output config is "default" or the estimator is not configured
        for wrapping return `data_to_wrap` unchanged.
        If the output config is "pandas", return `data_to_wrap` as a pandas
        DataFrame.
    """
    # 获取输出配置
    output_config = _get_output_config(method, estimator)

    # 如果输出配置为"default"或估计器未配置自动包装，则返回未改变的data_to_wrap
    if output_config["dense"] == "default" or not _auto_wrap_is_configured(estimator):
        return data_to_wrap

    # 否则，根据输出配置中的dense属性处理数据
    dense_config = output_config["dense"]
    
    # 如果数据为稀疏矩阵，则抛出异常
    if issparse(data_to_wrap):
        raise ValueError(
            "The transformer outputs a scipy sparse matrix. "
            "Try to set the transformer output to a dense array or disable "
            f"{dense_config.capitalize()} output with set_output(transform='default')."
        )

    # 否则，使用适配器创建容器，包装数据
    adapter = ADAPTERS_MANAGER.adapters[dense_config]
    return adapter.create_container(
        data_to_wrap,
        original_input,
        columns=estimator.get_feature_names_out,
    )


def _wrap_method_output(f, method):
    """Wrapper used by `_SetOutputMixin` to automatically wrap methods."""

    @wraps(f)
    def wrapped(self, X, *args, **kwargs):
        # 调用被装饰方法获取数据
        data_to_wrap = f(self, X, *args, **kwargs)
        
        # 如果数据是元组，则仅对第一个输出进行包装（用于交叉分解）
        if isinstance(data_to_wrap, tuple):
            return_tuple = (
                _wrap_data_with_container(method, data_to_wrap[0], X, self),
                *data_to_wrap[1:],
            )
            # 如果数据类型有'_make'方法（如命名元组），则使用'_make'方法创建返回的元组
            if hasattr(type(data_to_wrap), "_make"):
                return type(data_to_wrap)._make(return_tuple)
            return return_tuple

        # 否则，直接对数据进行包装
        return _wrap_data_with_container(method, data_to_wrap, X, self)

    return wrapped


def _auto_wrap_is_configured(estimator):
    """Return True if estimator is configured for auto-wrapping the transform method.

    `_SetOutputMixin` sets `_sklearn_auto_wrap_output_keys` to `set()` if auto wrapping
    is manually disabled.
    """
    # 获取自动包装输出键集合
    auto_wrap_output_keys = getattr(estimator, "_sklearn_auto_wrap_output_keys", set())
    
    # 返回估计器是否具有'get_feature_names_out'方法且'transform'在自动包装输出键集合中
    return (
        hasattr(estimator, "get_feature_names_out")
        and "transform" in auto_wrap_output_keys
    )


class _SetOutputMixin:
    """Mixin that dynamically wraps methods to return container based on config.
    
    This class is a mixin that dynamically wraps methods of its subclasses to 
    return data in a container format based on the output configuration.

    It interacts with other functions (`_wrap_data_with_container`, `_wrap_method_output`, 
    and `_auto_wrap_is_configured`) to achieve this behavior, ensuring that the 
    output of certain methods (like `transform`) conforms to specified formats 
    (like Pandas DataFrame) based on the estimator's configuration.

    """
    def __init_subclass__(cls, auto_wrap_output_keys=("transform",), **kwargs):
        # 调用父类的 __init_subclass__ 方法
        super().__init_subclass__(**kwargs)

        # 根据全局配置的 set_output 来动态包装 `transform` 和 `fit_transform` 方法
        # 根据传入的 auto_wrap_output_keys 配置输出
        if not (
            isinstance(auto_wrap_output_keys, tuple) or auto_wrap_output_keys is None
        ):
            # 如果 auto_wrap_output_keys 不是元组或者为 None，则抛出异常
            raise ValueError("auto_wrap_output_keys must be None or a tuple of keys.")

        if auto_wrap_output_keys is None:
            # 如果 auto_wrap_output_keys 是 None，则初始化为空集合
            cls._sklearn_auto_wrap_output_keys = set()
            return

        # 方法到配置键的映射
        method_to_key = {
            "transform": "transform",
            "fit_transform": "transform",
        }
        # 初始化自动包装输出键的集合
        cls._sklearn_auto_wrap_output_keys = set()

        # 遍历方法到键的映射
        for method, key in method_to_key.items():
            # 如果类没有定义 method 方法或者 key 不在 auto_wrap_output_keys 中，则继续
            if not hasattr(cls, method) or key not in auto_wrap_output_keys:
                continue
            # 添加 key 到自动包装输出键的集合中
            cls._sklearn_auto_wrap_output_keys.add(key)

            # 只包装类自身定义的方法
            if method not in cls.__dict__:
                continue
            # 获取包装后的方法，并设置为该方法的新实现
            wrapped_method = _wrap_method_output(getattr(cls, method), key)
            setattr(cls, method, wrapped_method)

    @available_if(_auto_wrap_is_configured)
    def set_output(self, *, transform=None):
        """Set output container.

        设置输出容器。

        See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
        for an example on how to use the API.
        查看 :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
        了解如何使用 API 的示例。

        Parameters
        ----------
        transform : {"default", "pandas", "polars"}, default=None
            Configure output of `transform` and `fit_transform`.
            配置 `transform` 和 `fit_transform` 的输出。

            - `"default"`: Default output format of a transformer
              默认的转换器输出格式
            - `"pandas"`: DataFrame output
              返回 DataFrame 格式的输出
            - `"polars"`: Polars output
              返回 Polars 格式的输出
            - `None`: Transform configuration is unchanged
              `None`: 转换配置不变

            .. versionadded:: 1.4
                `"polars"` option was added.
                添加了 `"polars"` 选项。

        Returns
        -------
        self : estimator instance
            Estimator instance.
            估计器实例。
        """
        if transform is None:
            return self

        # 如果实例没有 _sklearn_output_config 属性，则初始化为空字典
        if not hasattr(self, "_sklearn_output_config"):
            self._sklearn_output_config = {}

        # 设置 transform 对应的输出配置
        self._sklearn_output_config["transform"] = transform
        return self
def _safe_set_output(estimator, *, transform=None):
    """Safely call estimator.set_output and error if it not available.

    This is used by meta-estimators to set the output for child estimators.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance.

    transform : {"default", "pandas", "polars"}, default=None
        Configure output of the following estimator's methods:

        - `"transform"`
        - `"fit_transform"`

        If `None`, this operation is a no-op.

    Returns
    -------
    estimator : estimator instance
        Estimator instance.
    """
    # 检查是否需要设置转换输出
    set_output_for_transform = (
        hasattr(estimator, "transform")
        or hasattr(estimator, "fit_transform")
        and transform is not None
    )
    # 如果不需要设置转换输出，则直接返回
    if not set_output_for_transform:
        # 如果估算器不能进行转换，则不需要调用 `set_output`
        return

    # 检查估算器是否具有 `set_output` 方法，若没有则抛出异常
    if not hasattr(estimator, "set_output"):
        raise ValueError(
            f"Unable to configure output for {estimator} because `set_output` "
            "is not available."
        )
    
    # 调用估算器的 `set_output` 方法设置输出
    return estimator.set_output(transform=transform)
```