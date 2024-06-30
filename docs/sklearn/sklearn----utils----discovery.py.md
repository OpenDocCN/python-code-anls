# `D:\src\scipysrc\scikit-learn\sklearn\utils\discovery.py`

```
    # 导入必要的模块和函数
    """Utilities to discover scikit-learn objects."""
    import inspect  # 导入用于检查对象的模块
    import pkgutil  # 导入用于包的工具模块
    from importlib import import_module  # 从标准库导入模块导入函数
    from operator import itemgetter  # 导入用于操作对象的模块
    from pathlib import Path  # 导入处理文件路径的模块

    _MODULE_TO_IGNORE = {  # 定义需要忽略的模块集合
        "tests",
        "externals",
        "setup",
        "conftest",
        "experimental",
        "estimator_checks",
    }


    def all_estimators(type_filter=None):
        """Get a list of all estimators from `sklearn`.

        This function crawls the module and gets all classes that inherit
        from BaseEstimator. Classes that are defined in test-modules are not
        included.

        Parameters
        ----------
        type_filter : {"classifier", "regressor", "cluster", "transformer"} \
                or list of such str, default=None
            Which kind of estimators should be returned. If None, no filter is
            applied and all estimators are returned.  Possible values are
            'classifier', 'regressor', 'cluster' and 'transformer' to get
            estimators only of these specific types, or a list of these to
            get the estimators that fit at least one of the types.

        Returns
        -------
        estimators : list of tuples
            List of (name, class), where ``name`` is the class name as string
            and ``class`` is the actual type of the class.

        Examples
        --------
        >>> from sklearn.utils.discovery import all_estimators
        >>> estimators = all_estimators()
        >>> type(estimators)
        <class 'list'>
        >>> type(estimators[0])
        <class 'tuple'>
        >>> estimators[:2]
        [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>),
         ('AdaBoostClassifier',
          <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)]
        >>> classifiers = all_estimators(type_filter="classifier")
        >>> classifiers[:2]
        [('AdaBoostClassifier',
          <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),
         ('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>)]
        >>> regressors = all_estimators(type_filter="regressor")
        >>> regressors[:2]
        [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>),
         ('AdaBoostRegressor',
          <class 'sklearn.ensemble._weight_boosting.AdaBoostRegressor'>)]
        >>> both = all_estimators(type_filter=["classifier", "regressor"])
        >>> both[:2]
        [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>),
         ('AdaBoostClassifier',
          <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)]
        """
        # lazy import to avoid circular imports from sklearn.base
        from ..base import (  # 导入基础模块，避免循环导入
            BaseEstimator,
            ClassifierMixin,
            ClusterMixin,
            RegressorMixin,
            TransformerMixin,
        )
        from ._testing import ignore_warnings  # 导入用于忽略警告的函数

        def is_abstract(c):
            """Check if a class is abstract."""
            if not (hasattr(c, "__abstractmethods__")):
                return False
            if not len(c.__abstractmethods__):
                return False
            return True

        all_classes = []  # 初始化空列表，用于存储所有的类
        root = str(Path(__file__).parent.parent)  # 获取根目录路径字符串，即sklearn包的路径
    # 忽略导入时和遍历包时触发的弃用警告
    with ignore_warnings(category=FutureWarning):
        # 使用 pkgutil.walk_packages 遍历指定路径下以 'sklearn.' 开头的所有模块
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="sklearn."):
            # 将模块名按 '.' 分割成部分
            module_parts = module_name.split(".")
            # 如果模块名中的任意部分在 _MODULE_TO_IGNORE 中，或者模块名以 '._' 开头，则跳过该模块
            if (
                any(part in _MODULE_TO_IGNORE for part in module_parts)
                or "._" in module_name
            ):
                continue
            # 动态导入模块
            module = import_module(module_name)
            # 获取模块中的所有类
            classes = inspect.getmembers(module, inspect.isclass)
            # 过滤掉以 '_' 开头的类，将类名和类对象组成的元组列表
            classes = [
                (name, est_cls) for name, est_cls in classes if not name.startswith("_")
            ]

            # 将获取的类列表扩展到 all_classes 列表中
            all_classes.extend(classes)

    # 将 all_classes 转换为集合，去除重复类，以确保结果的可复现性
    all_classes = set(all_classes)

    # 筛选出是 BaseEstimator 的子类且不是 BaseEstimator 自身的类
    estimators = [
        c
        for c in all_classes
        if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")
    ]
    
    # 去除抽象基类
    estimators = [c for c in estimators if not is_abstract(c[1])]

    # 如果 type_filter 参数不为 None，则进行类型过滤
    if type_filter is not None:
        # 如果 type_filter 不是列表，则转换为单元素列表
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # 复制一份列表

        # 筛选出符合过滤条件的估算器
        filtered_estimators = []
        filters = {
            "classifier": ClassifierMixin,
            "regressor": RegressorMixin,
            "transformer": TransformerMixin,
            "cluster": ClusterMixin,
        }
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                # 将是 mixin 子类的估算器类加入筛选结果中
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)]
                )
        estimators = filtered_estimators
        # 如果 type_filter 列表还有剩余项，则抛出异常
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'classifier', "
                "'regressor', 'transformer', 'cluster' or "
                "None, got"
                f" {repr(type_filter)}."
            )

    # 去除重复项，并按类名排序，以确保可复现性
    # 使用 itemgetter 来确保排序仅依据元组的第一个项
    return sorted(set(estimators), key=itemgetter(0))
# 从`sklearn`中获取所有显示类的列表
def all_displays():
    """Get a list of all displays from `sklearn`.

    Returns
    -------
    displays : list of tuples
        List of (name, class), where ``name`` is the display class name as
        string and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from sklearn.utils.discovery import all_displays
    >>> displays = all_displays()
    >>> displays[0]
    ('CalibrationDisplay', <class 'sklearn.calibration.CalibrationDisplay'>)
    """
    # lazy import to avoid circular imports from sklearn.base
    from ._testing import ignore_warnings

    # 存储所有的类列表
    all_classes = []
    root = str(Path(__file__).parent.parent)  # 获取sklearn包的根目录路径
    # 忽略导入时和遍历包时触发的过时警告
    with ignore_warnings(category=FutureWarning):
        # 遍历sklearn包及其子包中的所有模块
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="sklearn."):
            module_parts = module_name.split(".")
            # 如果模块的任何部分包含在要忽略的模块列表中或者模块名以._开头，则跳过
            if (
                any(part in _MODULE_TO_IGNORE for part in module_parts)
                or "._" in module_name
            ):
                continue
            # 导入模块
            module = import_module(module_name)
            # 获取模块中的所有类
            classes = inspect.getmembers(module, inspect.isclass)
            # 过滤出不以下划线开头且以Display结尾的类，并存入列表
            classes = [
                (name, display_class)
                for name, display_class in classes
                if not name.startswith("_") and name.endswith("Display")
            ]
            # 将符合条件的类列表扩展到all_classes中
            all_classes.extend(classes)

    # 返回按类名排序去重后的所有类列表
    return sorted(set(all_classes), key=itemgetter(0))


def _is_checked_function(item):
    if not inspect.isfunction(item):
        return False

    if item.__name__.startswith("_"):
        return False

    mod = item.__module__
    if not mod.startswith("sklearn.") or mod.endswith("estimator_checks"):
        return False

    return True


# 获取`sklearn`中所有函数的列表
def all_functions():
    """Get a list of all functions from `sklearn`.

    Returns
    -------
    functions : list of tuples
        List of (name, function), where ``name`` is the function name as
        string and ``function`` is the actual function.

    Examples
    --------
    >>> from sklearn.utils.discovery import all_functions
    >>> functions = all_functions()
    >>> name, function = functions[0]
    >>> name
    'accuracy_score'
    """
    # lazy import to avoid circular imports from sklearn.base
    from ._testing import ignore_warnings

    # 存储所有函数的列表
    all_functions = []
    root = str(Path(__file__).parent.parent)  # 获取sklearn包的根目录路径
    # 忽略导入时和遍历包时触发的过时警告
    ```
    # 使用 ignore_warnings 上下文管理器，忽略 FutureWarning 类别的警告
    with ignore_warnings(category=FutureWarning):
        # 遍历指定路径下以 'sklearn.' 开头的所有包和模块
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="sklearn."):
            # 将模块名按 '.' 分割成部分
            module_parts = module_name.split(".")
            # 如果模块的任意部分在 _MODULE_TO_IGNORE 中，或者模块名包含 '._'，则跳过
            if (
                any(part in _MODULE_TO_IGNORE for part in module_parts)
                or "._" in module_name
            ):
                continue

            # 动态导入当前模块
            module = import_module(module_name)
            # 获取模块中所有满足 _is_checked_function 条件的函数和方法
            functions = inspect.getmembers(module, _is_checked_function)
            # 筛选出不以下划线开头的函数，并构建成 (函数名, 函数对象) 的列表
            functions = [
                (func.__name__, func)
                for name, func in functions
                if not name.startswith("_")
            ]
            # 将筛选得到的函数列表添加到 all_functions 列表中
            all_functions.extend(functions)

    # 去除重复项，并按照第一个元素（函数名）排序，以确保可重现性
    return sorted(set(all_functions), key=itemgetter(0))
```