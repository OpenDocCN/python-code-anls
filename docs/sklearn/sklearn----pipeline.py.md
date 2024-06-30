# `D:\src\scipysrc\scikit-learn\sklearn\pipeline.py`

```
# 导入所需的库和模块
from collections import Counter, defaultdict  # 导入计数器和默认字典
from itertools import chain, islice  # 导入链式迭代工具和切片工具

import numpy as np  # 导入NumPy库
from scipy import sparse  # 导入SciPy稀疏矩阵模块

from .base import TransformerMixin, _fit_context, clone  # 从当前包中导入基础类和函数
from .exceptions import NotFittedError  # 导入未拟合错误异常
from .preprocessing import FunctionTransformer  # 导入函数转换器
from .utils import Bunch  # 导入工具类Bunch
from .utils._estimator_html_repr import _VisualBlock  # 导入HTML表达的估计器
from .utils._metadata_requests import METHODS  # 导入方法列表
from .utils._param_validation import HasMethods, Hidden  # 导入参数验证工具类
from .utils._set_output import (
    _get_container_adapter,  # 导入获取容器适配器函数
    _safe_set_output,  # 导入安全设置输出函数
)
from .utils._tags import _safe_tags  # 导入安全标签工具
from .utils._user_interface import _print_elapsed_time  # 导入打印经过时间的函数
from .utils.deprecation import _deprecate_Xt_in_inverse_transform  # 导入反向转换中的Xt弃用函数
from .utils.metadata_routing import (
    MetadataRouter,  # 导入元数据路由器
    MethodMapping,  # 导入方法映射
    _raise_for_params,  # 导入参数异常处理函数
    _routing_enabled,  # 导入路由使能函数
    process_routing,  # 导入处理路由函数
)
from .utils.metaestimators import _BaseComposition, available_if  # 导入基本组合类和可用条件
from .utils.parallel import Parallel, delayed  # 导入并行处理类和延迟函数
from .utils.validation import check_is_fitted, check_memory  # 导入拟合检查和内存检查函数
    Read more in the :ref:`User Guide <pipeline>`.



# 在文档中引用用户指南中的链接
    .. versionadded:: 0.5



# 参数部分开始
    Parameters
    ----------
    steps : list of tuples
        List of (name of step, estimator) tuples that are to be chained in
        sequential order. To be compatible with the scikit-learn API, all steps
        must define `fit`. All non-last steps must also define `transform`. See
        :ref:`Combining Estimators <combining_estimators>` for more details.



# memory 参数的说明
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. The last step
        will never be cached, even if it is a transformer. By default, no
        caching is performed. If a string is given, it is the path to the
        caching directory. Enabling caching triggers a clone of the transformers
        before fitting. Therefore, the transformer instance given to the
        pipeline cannot be inspected directly. Use the attribute ``named_steps``
        or ``steps`` to inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.



# verbose 参数的说明
    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.



# Attributes 部分开始
    Attributes
    ----------



# named_steps 属性的说明
    named_steps : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.



# classes_ 属性的说明
    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only exist if the last step of the pipeline is a
        classifier.



# n_features_in_ 属性的说明
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying first estimator in `steps` exposes such an attribute
        when fit.

        .. versionadded:: 0.24



# feature_names_in_ 属性的说明
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0



# See Also 部分开始
    See Also
    --------



# make_pipeline 函数的说明
    make_pipeline : Convenience function for simplified pipeline construction.



# Examples 部分开始
    Examples
    --------



# 示例代码的说明
    >>> from sklearn.svm import SVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.pipeline import Pipeline
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    >>> # The pipeline can be used as any other estimator
    >>> # and avoids leaking the test set into the train set
    >>> pipe.fit(X_train, y_train).score(X_test, y_test)
    0.88
    """
    # An estimator's parameter can be set using '__' syntax
    # 使用 '__' 语法可以设置估计器的参数

    # BaseEstimator interface
    # BaseEstimator 接口

    # Required parameters for the estimator
    _required_parameters = ["steps"]
    # 估计器的必需参数列表，必须包含 "steps"

    # Parameter constraints for the estimator
    _parameter_constraints: dict = {
        "steps": [list, Hidden(tuple)],
        "memory": [None, str, HasMethods(["cache"])],
        "verbose": ["boolean"],
    }
    # 估计器的参数约束条件，包括对 "steps"、"memory" 和 "verbose" 参数的限制

    def __init__(self, steps, *, memory=None, verbose=False):
        """
        Initialize the estimator with required parameters.

        Parameters
        ----------
        steps : list
            List of tuples where each tuple contains a name and an estimator object.
            每个元组包含一个名称和一个估计器对象的列表。
        memory : str or None, default=None
            Optional string to indicate caching options.
            可选的字符串，用于指定缓存选项。
        verbose : bool, default=False
            Optional boolean flag indicating verbosity.
            可选的布尔标志，指示详细程度。
        """
        self.steps = steps
        self.memory = memory
        self.verbose = verbose

    def set_output(self, *, transform=None):
        """
        Set the output container when `"transform"` and `"fit_transform"` are called.

        Parameters
        ----------
        transform : {"default", "pandas", "polars"}, default=None
            Configure output of `transform` and `fit_transform`.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        # Iterate over steps and set the output container for each estimator
        for _, _, step in self._iter():
            _safe_set_output(step, transform=transform)
        return self

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return parameters for this estimator and contained subobjects.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("steps", deep=deep)

    def set_params(self, **kwargs):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **kwargs : dict
            Parameters of this estimator or parameters of estimators contained in `steps`.

        Returns
        -------
        self : object
            Pipeline class instance.
        """
        self._set_params("steps", **kwargs)
        return self
    ```
    def _validate_steps(self):
        # 将步骤列表分离为步骤名称和步骤对象两个元组
        names, estimators = zip(*self.steps)

        # 验证步骤名称的有效性
        self._validate_names(names)

        # 验证步骤对象的有效性
        transformers = estimators[:-1]  # 获取除最后一个步骤之外的所有步骤对象
        estimator = estimators[-1]      # 获取最后一个步骤对象

        for t in transformers:
            # 跳过值为 None 或 "passthrough" 的步骤对象
            if t is None or t == "passthrough":
                continue
            # 检查中间步骤对象是否实现了 fit 或 fit_transform 方法，并且是否实现了 transform 方法
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
                t, "transform"
            ):
                raise TypeError(
                    "All intermediate steps should be "
                    "transformers and implement fit and transform "
                    "or be the string 'passthrough'. "
                    "'%s' (type %s) doesn't" % (t, type(t))
                )

        # 允许最后一个步骤对象为 None，作为一个恒等变换
        if (
            estimator is not None
            and estimator != "passthrough"
            and not hasattr(estimator, "fit")
        ):
            raise TypeError(
                "Last step of Pipeline should implement fit "
                "or be the string 'passthrough'. "
                "'%s' (type %s) doesn't" % (estimator, type(estimator))
            )

    def _iter(self, with_final=True, filter_passthrough=True):
        """
        Generate (idx, (name, trans)) tuples from self.steps

        When filter_passthrough is True, 'passthrough' and None transformers
        are filtered out.
        """
        # 计算迭代终止条件
        stop = len(self.steps)
        if not with_final:
            stop -= 1

        # 遍历步骤列表，生成 (索引, (步骤名称, 步骤对象)) 元组
        for idx, (name, trans) in enumerate(islice(self.steps, 0, stop)):
            if not filter_passthrough:
                # 如果 filter_passthrough 为 False，则不过滤 'passthrough' 和 None 的步骤对象
                yield idx, name, trans
            elif trans is not None and trans != "passthrough":
                # 如果 filter_passthrough 为 True，则过滤掉 'passthrough' 和 None 的步骤对象
                yield idx, name, trans

    def __len__(self):
        """
        Returns the length of the Pipeline
        """
        # 返回 Pipeline 的步骤数量
        return len(self.steps)

    def __getitem__(self, ind):
        """Returns a sub-pipeline or a single estimator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.
        """
        if isinstance(ind, slice):
            # 如果使用切片索引，则返回一个新的 Pipeline 实例，其中包含 Pipeline 的切片副本
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            return self.__class__(
                self.steps[ind], memory=self.memory, verbose=self.verbose
            )
        try:
            # 如果使用整数索引，则返回该索引位置的步骤对象
            name, est = self.steps[ind]
        except TypeError:
            # 如果无法使用整数索引，则尝试使用名称获取步骤对象
            return self.named_steps[ind]
        return est

    @property
    def _estimator_type(self):
        # 返回 Pipeline 中最后一个步骤对象的 _estimator_type 属性
        return self.steps[-1][1]._estimator_type

    @property
    # 返回一个包含步骤名称和步骤对象的只读字典对象
    def named_steps(self):
        """Access the steps by name.

        Read-only attribute to access any step by given name.
        Keys are steps names and values are the steps objects.
        """
        # 使用 Bunch 对象来改善自动补全功能
        return Bunch(**dict(self.steps))

    @property
    # 返回管道中的最终估算器对象或者字符串 "passthrough"
    def _final_estimator(self):
        try:
            estimator = self.steps[-1][1]
            return "passthrough" if estimator is None else estimator
        except (ValueError, AttributeError, TypeError):
            # 在首次调用方法时调用 `_available_if` 和 `fit` 尚未验证 `steps` 时会发生这种情况。
            # 返回 `None`，然后会在稍后引发 `InvalidParameterError`。
            return None

    # 生成一个日志消息，指示当前处理的步骤和其索引
    def _log_message(self, step_idx):
        if not self.verbose:
            return None
        name, _ = self.steps[step_idx]

        return "(step %d of %d) Processing %s" % (step_idx + 1, len(self.steps), name)

    # 检查方法参数并根据路由情况处理它们
    def _check_method_params(self, method, props, **kwargs):
        if _routing_enabled():
            routed_params = process_routing(self, method, **props, **kwargs)
            return routed_params
        else:
            # 准备一个 Bunch 对象来存储每个步骤的方法参数
            fit_params_steps = Bunch(
                **{
                    name: Bunch(**{method: {} for method in METHODS})
                    for name, step in self.steps
                    if step is not None
                }
            )
            for pname, pval in props.items():
                if "__" not in pname:
                    # 如果参数名中不包含 "__"，则抛出 ValueError
                    raise ValueError(
                        "Pipeline.fit does not accept the {} parameter. "
                        "You can pass parameters to specific steps of your "
                        "pipeline using the stepname__parameter format, e.g. "
                        "`Pipeline.fit(X, y, logisticregression__sample_weight"
                        "=sample_weight)`.".format(pname)
                    )
                step, param = pname.split("__", 1)
                # 将参数值分配给相应步骤和方法的参数
                fit_params_steps[step]["fit"][param] = pval
                # 在没有元数据路由的情况下，fit_transform 和 fit_predict
                # 使用相同的参数并传递给最后的 fit 方法
                fit_params_steps[step]["fit_transform"][param] = pval
                fit_params_steps[step]["fit_predict"][param] = pval
            return fit_params_steps

    # 估算器接口
    # 定义私有方法 `_fit`，用于拟合管道模型
    def _fit(self, X, y=None, routed_params=None):
        # 对步骤进行浅复制，应当使用 `steps_` 而不是 `steps`
        self.steps = list(self.steps)
        # 验证管道步骤的有效性
        self._validate_steps()
        
        # 设置内存缓存
        memory = check_memory(self.memory)
        
        # 通过内存缓存装饰 `_fit_transform_one` 方法
        fit_transform_one_cached = memory.cache(_fit_transform_one)

        # 迭代管道的步骤
        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            # 如果转换器为空或者为 "passthrough"，则跳过当前步骤
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue
            
            # 如果内存缓存具有 `location` 属性且为 `None`，则不克隆转换器以保持向后兼容性
            if hasattr(memory, "location") and memory.location is None:
                cloned_transformer = transformer
            else:
                # 克隆当前转换器
                cloned_transformer = clone(transformer)
            
            # 拟合或从缓存中加载当前转换器
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                params=routed_params[name],
            )
            
            # 将当前步骤的转换器替换为已拟合的转换器
            self.steps[step_idx] = (name, fitted_transformer)
        
        # 返回拟合后的数据 X
        return X

    @_fit_context(
        # Pipeline.steps 中的估计器尚未验证
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **params):
        """Fit the model.

        Fit all the transformers one after the other and sequentially transform the
        data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True` is set via
                :func:`~sklearn.set_config`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        routed_params = self._check_method_params(method="fit", props=params)
        # 检查并处理参数，确保其正确路由到每个步骤
        Xt = self._fit(X, y, routed_params)
        # 执行管道中的所有变换器以及最终估算器的拟合过程
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                # 如果最终估算器不是"passthrough"，则使用其fit方法拟合最终转换后的数据
                last_step_params = routed_params[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **last_step_params["fit"])

        return self

    def _can_fit_transform(self):
        # 检查最终估算器是否支持fit_transform方法或者transform方法，或者是"passthrough"
        return (
            self._final_estimator == "passthrough"
            or hasattr(self._final_estimator, "transform")
            or hasattr(self._final_estimator, "fit_transform")
        )

    @available_if(_can_fit_transform)
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    # 定义一个方法，用于在 pipeline 中拟合模型并进行数据转换
    def fit_transform(self, X, y=None, **params):
        """Fit the model and transform with the final estimator.

        Fit all the transformers one after the other and sequentially transform
        the data. Only valid if the final estimator either implements
        `fit_transform` or `fit` and `transform`.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        # 根据参数检查方法的参数并获取已路由的参数
        routed_params = self._check_method_params(method="fit_transform", props=params)
        # 调用内部方法 _fit 进行拟合操作
        Xt = self._fit(X, y, routed_params)

        # 获取最后一个步骤的估算器
        last_step = self._final_estimator
        # 打印 pipeline 的运行时间
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            # 如果最后一步是 "passthrough"，直接返回 Xt
            if last_step == "passthrough":
                return Xt
            # 获取最后一步的参数
            last_step_params = routed_params[self.steps[-1][0]]
            # 如果最后一步实现了 fit_transform 方法，则调用它
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(
                    Xt, y, **last_step_params["fit_transform"]
                )
            # 否则，先调用 fit 方法，再调用 transform 方法
            else:
                return last_step.fit(Xt, y, **last_step_params["fit"]).transform(
                    Xt, **last_step_params["transform"]
                )

    # 仅在最终估算器具有 "predict" 方法时有效
    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **params):
        """
        Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters to the ``predict`` called at the end of all
                transformations in the pipeline.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionadded:: 0.20

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True` is set via
                :func:`~sklearn.set_config`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

            Note that while this may be used to return uncertainties from some
            models with ``return_std`` or ``return_cov``, uncertainties that are
            generated by the transformations in the pipeline are not propagated
            to the final estimator.

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        # Initialize Xt with input data X
        Xt = X

        # Check if metadata routing is not enabled
        if not _routing_enabled():
            # Iterate over all pipeline steps except the final estimator
            for _, name, transform in self._iter(with_final=False):
                # Apply transform to Xt for each step
                Xt = transform.transform(Xt)
            # Return the prediction from the final estimator using transformed Xt and parameters
            return self.steps[-1][1].predict(Xt, **params)

        # Metadata routing is enabled
        # Process and obtain routed parameters for prediction
        routed_params = process_routing(self, "predict", **params)
        
        # Iterate over all pipeline steps except the final estimator
        for _, name, transform in self._iter(with_final=False):
            # Apply transform to Xt with routed parameters for each step
            Xt = transform.transform(Xt, **routed_params[name].transform)
        
        # Return the prediction from the final estimator using transformed Xt and routed prediction parameters
        return self.steps[-1][1].predict(Xt, **routed_params[self.steps[-1][0]].predict)

    @available_if(_final_estimator_has("fit_predict"))
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_predict(self, X, y=None, **params):
        """
        Transform the data, and apply `fit_predict` with the final estimator.

        Call `fit_transform` of each transformer in the pipeline. The
        transformed data are finally passed to the final estimator that calls
        `fit_predict` method. Only valid if the final estimator implements
        `fit_predict`.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters to the ``predict`` called at the end of all
                transformations in the pipeline.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionadded:: 0.20

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

            Note that while this may be used to return uncertainties from some
            models with ``return_std`` or ``return_cov``, uncertainties that are
            generated by the transformations in the pipeline are not propagated
            to the final estimator.

        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the final estimator.
        """
        # 检查并获取方法参数
        routed_params = self._check_method_params(method="fit_predict", props=params)
        # 对输入数据进行拟合和转换
        Xt = self._fit(X, y, routed_params)

        # 获取最后一步的参数
        params_last_step = routed_params[self.steps[-1][0]]
        # 调用最后一步估算器的fit_predict方法，并记录执行时间
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][1].fit_predict(
                Xt, y, **params_last_step.get("fit_predict", {})
            )
        # 返回预测结果
        return y_pred

    @available_if(_final_estimator_has("predict_proba"))


这段代码是一个类方法的定义，用于在机器学习流水线中执行数据转换和最终估算器的`fit_predict`方法。
    def predict_proba(self, X, **params):
        """
        Transform the data, and apply `predict_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_proba` method. Only valid if the final estimator implements
        `predict_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters to the `predict_proba` called at the end of all
                transformations in the pipeline.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionadded:: 0.20

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_proba` on the final estimator.
        """
        # Initialize Xt with the input data X
        Xt = X

        # Check if metadata routing is not enabled
        if not _routing_enabled():
            # Iterate over transformers excluding the final estimator
            for _, name, transform in self._iter(with_final=False):
                # Transform data using each transformer
                Xt = transform.transform(Xt)
            # Return the result of predict_proba from the final estimator
            return self.steps[-1][1].predict_proba(Xt, **params)

        # metadata routing enabled
        # Process parameters with routing logic
        routed_params = process_routing(self, "predict_proba", **params)
        
        # Iterate over transformers excluding the final estimator
        for _, name, transform in self._iter(with_final=False):
            # Transform data using each transformer with routed parameters
            Xt = transform.transform(Xt, **routed_params[name].transform)
        
        # Return the result of predict_proba from the final estimator with routed parameters
        return self.steps[-1][1].predict_proba(
            Xt, **routed_params[self.steps[-1][0]].predict_proba
        )

    @available_if(_final_estimator_has("decision_function"))


这段代码定义了一个 `predict_proba` 方法，用于在管道中进行数据转换并应用最终估算器的 `predict_proba` 方法。根据是否启用元数据路由，它决定如何传递和处理参数，并确保最终估算器实现了 `predict_proba` 方法。
    def decision_function(self, X, **params):
        """
        Transform the data, and apply `decision_function` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `decision_function` method. Only valid if the final estimator
        implements `decision_function`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of string -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        y_score : ndarray of shape (n_samples, n_classes)
            Result of calling `decision_function` on the final estimator.
        """
        # Validate and raise an error for any unsupported parameters
        _raise_for_params(params, self, "decision_function")

        # Process and route metadata if `enable_metadata_routing=True`
        # If `enable_metadata_routing` is not set or False, `params` will be empty
        routed_params = process_routing(self, "decision_function", **params)

        # Initialize Xt with the input data X
        Xt = X

        # Iterate over each step (excluding the final estimator) in the pipeline
        for _, name, transform in self._iter(with_final=False):
            # Apply transform.transform to Xt with specific routed parameters
            Xt = transform.transform(
                Xt, **routed_params.get(name, {}).get("transform", {})
            )

        # Apply decision_function of the final estimator on the transformed data Xt
        return self.steps[-1][1].decision_function(
            Xt, **routed_params.get(self.steps[-1][0], {}).get("decision_function", {})
        )

    @available_if(_final_estimator_has("score_samples"))
    def score_samples(self, X):
        """
        Transform the data, and apply `score_samples` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score_samples` method. Only valid if the final estimator implements
        `score_samples`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            Result of calling `score_samples` on the final estimator.
        """
        # Initialize Xt with the input data X
        Xt = X

        # Iterate over each transformer in the pipeline (excluding the final estimator)
        for _, _, transformer in self._iter(with_final=False):
            # Apply transformer.transform to Xt
            Xt = transformer.transform(Xt)

        # Apply score_samples method of the final estimator on transformed data Xt
        return self.steps[-1][1].score_samples(Xt)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **params):
        """
        Transform the data, and apply `predict_log_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_log_proba` method. Only valid if the final estimator
        implements `predict_log_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters to the `predict_log_proba` called at the end of all
                transformations in the pipeline.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionadded:: 0.20

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_log_proba` on the final estimator.
        """
        # Initialize the transformed data as the input X
        Xt = X

        # Check if metadata routing is disabled
        if not _routing_enabled():
            # Iterate over each step (excluding the final estimator) in the pipeline
            for _, name, transform in self._iter(with_final=False):
                # Transform the data using each transformer in the pipeline
                Xt = transform.transform(Xt)
            # Call `predict_log_proba` on the final estimator with transformed data and parameters
            return self.steps[-1][1].predict_log_proba(Xt, **params)

        # metadata routing enabled
        # Process routing parameters based on the pipeline and method name
        routed_params = process_routing(self, "predict_log_proba", **params)
        
        # Iterate over each step (excluding the final estimator) in the pipeline
        for _, name, transform in self._iter(with_final=False):
            # Transform the data using each transformer and routed parameters
            Xt = transform.transform(Xt, **routed_params[name].transform)
        
        # Call `predict_log_proba` on the final estimator with transformed data and routed parameters
        return self.steps[-1][1].predict_log_proba(
            Xt, **routed_params[self.steps[-1][0]].predict_log_proba
        )

    def _can_transform(self):
        # Check if the final estimator supports transformation or is set to "passthrough"
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "transform"
        )

    @available_if(_can_transform)
    def transform(self, X, **params):
        """Transform the data, and apply `transform` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        _raise_for_params(params, self, "transform")  # 检查参数是否正确，若不正确则抛出异常

        # not branching here since params is only available if
        # enable_metadata_routing=True
        routed_params = process_routing(self, "transform", **params)  # 处理路由参数，如果启用了 metadata routing，则传递参数给各步骤
        Xt = X  # 初始化转换后的数据为输入数据 X
        for _, name, transform in self._iter():
            Xt = transform.transform(Xt, **routed_params[name].transform)  # 遍历管道中的每个转换器，并应用其 transform 方法到 Xt 上
        return Xt  # 返回转换后的数据

    def _can_inverse_transform(self):
        return all(hasattr(t, "inverse_transform") for _, _, t in self._iter())  # 检查管道中所有转换器是否具有 inverse_transform 方法

    @available_if(_can_inverse_transform)
    def inverse_transform(self, X=None, *, Xt=None, **params):
        """
        Apply `inverse_transform` for each step in a reverse order.

        All estimators in the pipeline must support `inverse_transform`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

        Xt : array-like of shape (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

            .. deprecated:: 1.5
                `Xt` was deprecated in 1.5 and will be removed in 1.7. Use `X` instead.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Inverse transformed data, that is, data in the original feature
            space.
        """
        # Validate parameters against the function signature and current object
        _raise_for_params(params, self, "inverse_transform")

        # Handle deprecated `Xt` parameter by converting it to `X` if provided
        X = _deprecate_Xt_in_inverse_transform(X, Xt)

        # Process metadata routing if enabled, otherwise `params` will be empty
        routed_params = process_routing(self, "inverse_transform", **params)

        # Iterate over pipeline steps in reverse order and apply inverse_transform
        reverse_iter = reversed(list(self._iter()))
        for _, name, transform in reverse_iter:
            # Apply inverse_transform of each step to the data `X`
            X = transform.inverse_transform(X, **routed_params[name].inverse_transform)

        # Return the final inverse transformed data
        return X

    @available_if(_final_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None, **params):
        """
        Transform the data, and apply `score` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score` method. Only valid if the final estimator implements `score`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        score : float
            Result of calling `score` on the final estimator.
        """
        # 初始化变量 Xt 为输入数据 X
        Xt = X

        # 如果元数据路由未启用
        if not _routing_enabled():
            # 遍历管道中的每个转换器（除了最后一个评估器）
            for _, name, transform in self._iter(with_final=False):
                # 对数据 Xt 应用当前转换器的 transform 方法
                Xt = transform.transform(Xt)
            
            # 准备用于评估的参数
            score_params = {}
            if sample_weight is not None:
                score_params["sample_weight"] = sample_weight
            
            # 返回最后一个评估器的 score 方法的结果
            return self.steps[-1][1].score(Xt, y, **score_params)

        # 如果启用了元数据路由
        # 对参数进行路由处理，使用 process_routing 函数
        routed_params = process_routing(
            self, "score", sample_weight=sample_weight, **params
        )

        # 将 Xt 重新设置为输入数据 X
        Xt = X

        # 遍历管道中的每个转换器（除了最后一个评估器）
        for _, name, transform in self._iter(with_final=False):
            # 对数据 Xt 应用当前转换器的 transform 方法，使用路由参数
            Xt = transform.transform(Xt, **routed_params[name].transform)
        
        # 返回最后一个评估器的 score 方法的结果，使用路由参数
        return self.steps[-1][1].score(Xt, y, **routed_params[self.steps[-1][0]].score)

    @property
    def classes_(self):
        """The classes labels. Only exist if the last step is a classifier."""
        # 返回管道中最后一个步骤的评估器的 classes_ 属性
        return self.steps[-1][1].classes_
    def _more_tags(self):
        # 定义一个包含测试标签的字典，用于测试中的特定标记
        tags = {
            "_xfail_checks": {
                "check_dont_overwrite_parameters": (
                    "Pipeline changes the `steps` parameter, which it shouldn't."
                    "Therefore this test is x-fail until we fix this."
                ),
                "check_estimators_overwrite_params": (
                    "Pipeline changes the `steps` parameter, which it shouldn't."
                    "Therefore this test is x-fail until we fix this."
                ),
            }
        }

        try:
            # 尝试获取第一个步骤的安全标签
            tags["pairwise"] = _safe_tags(self.steps[0][1], "pairwise")
        except (ValueError, AttributeError, TypeError):
            # 捕获当 `steps` 不是 (name, estimator) 元组列表或者尚未调用 `fit` 验证步骤时的异常
            # 这种情况发生在 `steps` 不是 (name, estimator) 元组列表，或者尚未调用 `fit` 验证步骤时
            pass

        try:
            # 尝试获取最后一个步骤的安全标签
            tags["multioutput"] = _safe_tags(self.steps[-1][1], "multioutput")
        except (ValueError, AttributeError, TypeError):
            # 捕获当 `steps` 不是 (name, estimator) 元组列表或者尚未调用 `fit` 验证步骤时的异常
            # 这种情况发生在 `steps` 不是 (name, estimator) 元组列表，或者尚未调用 `fit` 验证步骤时
            pass

        return tags

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Transform input features using the pipeline.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        # 将输入特征赋给输出特征名
        feature_names_out = input_features
        for _, name, transform in self._iter():
            # 检查当前步骤是否具有 `get_feature_names_out` 方法
            if not hasattr(transform, "get_feature_names_out"):
                raise AttributeError(
                    "Estimator {} does not provide get_feature_names_out. "
                    "Did you mean to call pipeline[:-1].get_feature_names_out"
                    "()?".format(name)
                )
            # 使用当前步骤的 `get_feature_names_out` 方法进行特征名转换
            feature_names_out = transform.get_feature_names_out(feature_names_out)
        return feature_names_out

    @property
    def n_features_in_(self):
        """Number of features seen during first step `fit` method."""
        # 委托给第一个步骤的 `n_features_in_` 属性（将调用 `_check_is_fitted`）
        return self.steps[0][1].n_features_in_

    @property
    def feature_names_in_(self):
        """Names of features seen during first step `fit` method."""
        # 委托给第一个步骤的 `feature_names_in_` 属性（将调用 `_check_is_fitted`）
        return self.steps[0][1].feature_names_in_
    # 检查管道是否已经拟合（即是否已经训练过）
    def __sklearn_is_fitted__(self):
        """Indicate whether pipeline has been fit."""
        try:
            # 检查管道中最后一步是否已经拟合
            # 我们只检查最后一步，因为如果最后一步已经拟合，
            # 则意味着前面的步骤也应该已经拟合。这比检查管道中的每一步都拟合要快。
            check_is_fitted(self.steps[-1][1])
            # 如果最后一步已经拟合，则返回 True
            return True
        except NotFittedError:
            # 如果有未拟合的步骤，则返回 False
            return False

    def _sk_visual_block_(self):
        # 解构 self.steps 元组，分别获取步骤名称和估算器对象
        _, estimators = zip(*self.steps)

        def _get_name(name, est):
            if est is None or est == "passthrough":
                # 如果估算器为 None 或者为 "passthrough"，则返回简单的标识字符串
                return f"{name}: passthrough"
            # 如果是一个有效的估算器，则返回格式化的名称字符串
            return f"{name}: {est.__class__.__name__}"

        # 生成步骤名称列表，每个名称格式为 "步骤名: 估算器类名"
        names = [_get_name(name, est) for name, est in self.steps]
        # 获取估算器的详细描述信息列表，这些信息通常是估算器的参数设置等
        name_details = [str(est) for est in estimators]
        # 返回一个 _VisualBlock 对象，用于可视化展示管道的结构
        return _VisualBlock(
            "serial",
            estimators,
            names=names,
            name_details=name_details,
            dash_wrapped=False,
        )
# 为给定的估计器生成名称。
def _name_estimators(estimators):
    # 创建一个列表，其中包含估计器的名称或类名的小写形式（如果未提供名称）。
    names = [
        estimator if isinstance(estimator, str) else type(estimator).__name__.lower()
        for estimator in estimators
    ]
    # 使用 defaultdict 统计每个名称的出现次数。
    namecount = defaultdict(int)
    # 遍历估计器列表和名称列表，更新名称计数。
    for est, name in zip(estimators, names):
        namecount[name] += 1

    # 移除只出现一次的名称。
    for k, v in list(namecount.items()):
        if v == 1:
            del namecount[k]

    # 反向遍历估计器列表，更新重复名称的估计器名称。
    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    # 返回名称和估计器的元组列表。
    return list(zip(names, estimators))


# 构造一个 Pipeline 对象，由给定的估计器组成。
def make_pipeline(*steps, memory=None, verbose=False):
    """Construct a :class:`Pipeline` from the given estimators.

    This is a shorthand for the :class:`Pipeline` constructor; it does not
    require, and does not permit, naming the estimators. Instead, their names
    will be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of Estimator objects
        List of the scikit-learn estimators that are chained together.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. The last step
        will never be cached, even if it is a transformer. By default, no
        caching is performed. If a string is given, it is the path to the
        caching directory. Enabling caching triggers a clone of the transformers
        before fitting. Therefore, the transformer instance given to the
        pipeline cannot be inspected directly. Use the attribute ``named_steps``
        or ``steps`` to inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Returns
    -------
    p : Pipeline
        Returns a scikit-learn :class:`Pipeline` object.

    See Also
    --------
    Pipeline : Class for creating a pipeline of transforms with a final
        estimator.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('gaussiannb', GaussianNB())])
    """
    # 使用 _name_estimators 函数为估计器生成名称，并构造 Pipeline 对象返回。
    return Pipeline(_name_estimators(steps), memory=memory, verbose=verbose)


# 调用 transform 方法，并对输出应用权重。
def _transform_one(transformer, X, y, weight, params=None):
    """Call transform and apply weight to output.

    Parameters
    ----------
    transformer : estimator
        Estimator to be used for transformation.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data to be transformed.

    y : ndarray
        Ignored.
    """
    # 定义函数的参数 weight，表示要应用于转换输出的权重，类型为 float
    weight : float
    # 定义函数的参数 params，表示传递给转换器的参数，应当是形式为 process_routing()["step_name"] 的字典
    params : dict
        Parameters to be passed to the transformer's ``transform`` method.
        
        This should be of the form ``process_routing()["step_name"]``.
    """
    # 调用 transformer 对象的 transform 方法，传递 X 和 params.transform 作为参数，获取转换后的结果
    res = transformer.transform(X, **params.transform)
    # 如果为该转换器定义了权重 weight，则将结果乘以权重
    if weight is None:
        return res
    # 返回乘以权重后的结果
    return res * weight
def _fit_transform_one(
    transformer, X, y, weight, message_clsname="", message=None, params=None
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.

    ``params`` needs to be of the form ``process_routing()["step_name"]``.
    """
    # 如果参数 params 为 None，则设为空字典
    params = params or {}
    # 使用 _print_elapsed_time 函数打印执行时间信息，传入的参数为类名和消息
    with _print_elapsed_time(message_clsname, message):
        # 如果 transformer 对象有 fit_transform 方法，则调用它
        if hasattr(transformer, "fit_transform"):
            # 调用 fit_transform 方法，传入 X, y 和 params 中的 fit_transform 参数
            res = transformer.fit_transform(X, y, **params.get("fit_transform", {}))
        else:
            # 否则，调用 fit 方法进行拟合，然后再调用 transform 方法进行变换
            res = transformer.fit(X, y, **params.get("fit", {})).transform(
                X, **params.get("transform", {})
            )

    # 如果 weight 为 None，则返回变换后的结果和拟合后的 transformer 对象
    if weight is None:
        return res, transformer
    # 否则，返回变换后的结果乘以 weight 和拟合后的 transformer 对象
    return res * weight, transformer


def _fit_one(transformer, X, y, weight, message_clsname="", message=None, params=None):
    """
    Fits ``transformer`` to ``X`` and ``y``.
    """
    # 使用 _print_elapsed_time 函数打印执行时间信息，传入的参数为类名和消息
    with _print_elapsed_time(message_clsname, message):
        # 调用 transformer 对象的 fit 方法进行拟合，传入 X, y 和 params 中的 fit 参数
        return transformer.fit(X, y, **params["fit"])


class FeatureUnion(TransformerMixin, _BaseComposition):
    """Concatenates results of multiple transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer, removed by
    setting to 'drop' or disabled by setting to 'passthrough' (features are
    passed without transformation).

    Read more in the :ref:`User Guide <feature_union>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    transformer_list : list of (str, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer. The transformer can
        be 'drop' for it to be ignored or can be 'passthrough' for features to
        be passed unchanged.

        .. versionadded:: 1.1
           Added the option `"passthrough"`.

        .. versionchanged:: 0.22
           Deprecated `None` as a transformer in favor of 'drop'.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
        Raises ValueError if key not present in ``transformer_list``.
    """
    pass
    verbose : bool, default=False
        # 控制是否打印每个转换器拟合时的耗时信息，默认为False，即不打印。

    verbose_feature_names_out : bool, default=True
        # 控制是否在生成的特征名称前加上生成该特征的转换器的名称，默认为True。
        # 如果为True，:meth:`get_feature_names_out` 将在每个特征名前加上转换器名称。
        # 如果为False，:meth:`get_feature_names_out` 将不会在特征名前加上任何前缀，并且如果特征名不唯一则会报错。
        # 添加于版本 1.5。

    Attributes
    ----------
    named_transformers : :class:`~sklearn.utils.Bunch`
        # 类似字典的对象，具有以下属性。
        # 只读属性，通过用户给定的名称访问任何转换器参数。
        # 键是转换器名称，值是转换器参数。
        # 添加于版本 1.2。

    n_features_in_ : int
        # 在 :term:`fit` 过程中看到的特征数量。
        # 仅在 `transformer_list` 中的第一个转换器在拟合时公开这样的属性时才定义。
        # 添加于版本 0.24。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 :term:`fit` 过程中看到的特征的名称。
        # 仅当 `X` 具有全部为字符串的特征名称时才定义。
        # 添加于版本 1.3。

    See Also
    --------
    make_union : 简化特征合并构造的便捷函数。

    Examples
    --------
    >>> from sklearn.pipeline import FeatureUnion
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> union = FeatureUnion([("pca", PCA(n_components=1)),
    ...                       ("svd", TruncatedSVD(n_components=2))])
    >>> X = [[0., 1., 3], [2., 2., 5]]
    >>> union.fit_transform(X)
    array([[-1.5       ,  3.0..., -0.8...],
           [ 1.5       ,  5.7...,  0.4...]])
    >>> # 使用 '__' 语法可以设置估算器的参数
    >>> union.set_params(svd__n_components=1).fit_transform(X)
    array([[-1.5       ,  3.0...],
           [ 1.5       ,  5.7...]])

    For a more detailed example of usage, see
    :ref:`sphx_glr_auto_examples_compose_plot_feature_union.py`.
    """

    _required_parameters = ["transformer_list"]

    def __init__(
        self,
        transformer_list,
        *,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
    ):
        # 初始化方法，用于设置对象的初始状态和属性。
        self.transformer_list = transformer_list  # 设置转换器列表
        self.n_jobs = n_jobs  # 设置并行执行的作业数
        self.transformer_weights = transformer_weights  # 设置转换器权重
        self.verbose = verbose  # 是否打印拟合每个转换器时的耗时信息
        self.verbose_feature_names_out = verbose_feature_names_out  # 是否在特征名称前加上转换器名称
    def set_output(self, *, transform=None):
        """
        设置当调用 `"transform"` 和 `"fit_transform"` 时的输出容器。

        `set_output` 方法会设置 `transformer_list` 中所有估计器的输出。

        Parameters
        ----------
        transform : {"default", "pandas", "polars"}, default=None
            配置 `transform` 和 `fit_transform` 的输出格式。

            - `"default"`: 估计器的默认输出格式
            - `"pandas"`: DataFrame 输出
            - `"polars"`: Polars 输出
            - `None`: 不改变转换配置

        Returns
        -------
        self : estimator instance
            估计器实例。
        """
        super().set_output(transform=transform)  # 调用父类的 set_output 方法
        for _, step, _ in self._iter():
            _safe_set_output(step, transform=transform)  # 对每个步骤设置输出
        return self

    @property
    def named_transformers(self):
        """
        使用 Bunch 对象改进自动补全功能。
        返回一个将 `transformer_list` 转换为字典的 Bunch 对象。
        """
        return Bunch(**dict(self.transformer_list))

    def get_params(self, deep=True):
        """
        获取此估计器的参数。

        返回构造函数中给定的参数以及 `FeatureUnion` 的 `transformer_list` 中包含的估计器的参数。

        Parameters
        ----------
        deep : bool, default=True
            如果为 True，将返回此估计器及其包含的子对象（即估计器）的参数。

        Returns
        -------
        params : mapping of string to any
            参数名称映射到其值的字典。
        """
        return self._get_params("transformer_list", deep=deep)

    def set_params(self, **kwargs):
        """
        设置此估计器的参数。

        可以使用 ``get_params()`` 列出有效的参数键。注意，可以直接设置 `transformer_list` 中包含的估计器的参数。

        Parameters
        ----------
        **kwargs : dict
            此估计器的参数或包含在 `transform_list` 中的估计器的参数。可以使用估计器的名称和参数名称之间的 '__' 分隔符来设置转换器的参数。

        Returns
        -------
        self : object
            FeatureUnion 类的实例。
        """
        self._set_params("transformer_list", **kwargs)  # 设置 `transformer_list` 的参数
        return self
    def _validate_transformers(self):
        # 从 self.transformer_list 中解压出 names 和 transformers
        names, transformers = zip(*self.transformer_list)

        # 调用 _validate_names 方法验证 names
        self._validate_names(names)

        # 验证每个 transformer 是否符合要求
        for t in transformers:
            # 如果 transformer 是 "drop" 或 "passthrough"，跳过验证
            if t in ("drop", "passthrough"):
                continue
            # 如果 transformer 没有 fit 或 fit_transform 方法，或者没有 transform 方法，则抛出 TypeError
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
                t, "transform"
            ):
                raise TypeError(
                    "All estimators should implement fit and "
                    "transform. '%s' (type %s) doesn't" % (t, type(t))
                )

    def _validate_transformer_weights(self):
        # 如果 transformer_weights 为空，则直接返回
        if not self.transformer_weights:
            return

        # 获取 transformer_list 中的所有名字构成集合 transformer_names
        transformer_names = set(name for name, _ in self.transformer_list)
        # 检查每个 transformer_weights 中的名字是否在 transformer_names 中，否则抛出 ValueError
        for name in self.transformer_weights:
            if name not in transformer_names:
                raise ValueError(
                    f'Attempting to weight transformer "{name}", '
                    "but it is not present in transformer_list."
                )

    def _iter(self):
        """
        Generate (name, trans, weight) tuples excluding None and
        'drop' transformers.
        """
        # 获取权重信息的函数
        get_weight = (self.transformer_weights or {}).get

        # 遍历 transformer_list，生成 (name, trans, weight) 的元组，但排除 None 和 'drop' transformer
        for name, trans in self.transformer_list:
            if trans == "drop":
                continue
            # 如果 trans 是 "passthrough"，则使用 FunctionTransformer 创建一个新的 trans
            if trans == "passthrough":
                trans = FunctionTransformer(feature_names_out="one-to-one")
            yield (name, trans, get_weight(name))

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        # 存储 (name, feature_names_out) 元组的列表
        transformer_with_feature_names_out = []
        # 遍历 _iter() 生成的迭代器
        for name, trans, _ in self._iter():
            # 如果 trans 没有 get_feature_names_out 方法，则抛出 AttributeError
            if not hasattr(trans, "get_feature_names_out"):
                raise AttributeError(
                    "Transformer %s (type %s) does not provide get_feature_names_out."
                    % (str(name), type(trans).__name__)
                )
            # 获取 trans 的输出特征名字，并存储为 (name, feature_names_out) 元组
            feature_names_out = trans.get_feature_names_out(input_features)
            transformer_with_feature_names_out.append((name, feature_names_out))

        # 调用 _add_prefix_for_feature_names_out 方法处理特征名字的前缀，并返回结果
        return self._add_prefix_for_feature_names_out(
            transformer_with_feature_names_out
        )
    def _add_prefix_for_feature_names_out(self, transformer_with_feature_names_out):
        """Add prefix for feature names out that includes the transformer names.

        Parameters
        ----------
        transformer_with_feature_names_out : list of tuples of (str, array-like of str)
            The tuple consistent of the transformer's name and its feature names out.

        Returns
        -------
        feature_names_out : ndarray of shape (n_features,), dtype=str
            Transformed feature names.
        """
        # 如果设置了 verbose_feature_names_out 为 True
        if self.verbose_feature_names_out:
            # 为每个特征名添加转换器名称作为前缀
            names = list(
                chain.from_iterable(
                    (f"{name}__{i}" for i in feature_names_out)
                    for name, feature_names_out in transformer_with_feature_names_out
                )
            )
            return np.asarray(names, dtype=object)

        # verbose_feature_names_out 为 False 的情况下
        # 检查特征名是否全部唯一且没有前缀
        feature_names_count = Counter(
            chain.from_iterable(s for _, s in transformer_with_feature_names_out)
        )
        # 获取出现次数最多的前 6 个重复的特征名
        top_6_overlap = [
            name for name, count in feature_names_count.most_common(6) if count > 1
        ]
        # 对这些重复的特征名进行排序
        top_6_overlap.sort()
        if top_6_overlap:
            if len(top_6_overlap) == 6:
                # 如果超过 5 个重复的特征名，只显示前 5 个以及省略号
                names_repr = str(top_6_overlap[:5])[:-1] + ", ...]"
            else:
                names_repr = str(top_6_overlap)
            # 抛出数值错误，指示输出特征名不唯一，并建议设置 verbose_feature_names_out=True
            raise ValueError(
                f"Output feature names: {names_repr} are not unique. Please set "
                "verbose_feature_names_out=True to add prefixes to feature names"
            )

        # 如果没有重复的特征名，则直接返回转换器的特征名数组
        return np.concatenate(
            [name for _, name in transformer_with_feature_names_out],
        )
    def fit(self, X, y=None, **fit_params):
        """Fit all transformers using X.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **fit_params : dict, default=None
            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `fit` methods of the
              sub-transformers.

            - If `enable_metadata_routing=True`:
              Parameters safely routed to the `fit` methods of the
              sub-transformers. See :ref:`Metadata Routing User Guide
              <metadata_routing>` for more details.

            .. versionchanged:: 1.5
                `**fit_params` can be routed via metadata routing API.

        Returns
        -------
        self : object
            FeatureUnion class instance.
        """
        # 如果启用了元数据路由，对参数进行处理以进行路由
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            # 否则，使用旧的方法处理参数
            # TODO(SLEP6): 当元数据路由不可禁用时删除此处
            routed_params = Bunch()
            for name, _ in self.transformer_list:
                routed_params[name] = Bunch(fit={})
                routed_params[name].fit = fit_params

        # 并行处理每个转换器的拟合过程
        transformers = self._parallel_func(X, y, _fit_one, routed_params)

        # 如果没有任何转换器返回，则直接返回 self
        if not transformers:
            # 所有转换器均为 None
            return self

        # 更新转换器列表
        self._update_transformer_list(transformers)
        return self
    # 定义一个方法用于拟合和转换数据，同时连接结果
    def fit_transform(self, X, y=None, **params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **params : dict, default=None
            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `fit` methods of the
              sub-transformers.

            - If `enable_metadata_routing=True`:
              Parameters safely routed to the `fit` methods of the
              sub-transformers. See :ref:`Metadata Routing User Guide
              <metadata_routing>` for more details.

            .. versionchanged:: 1.5
                `**params` can now be routed via metadata routing API.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        # 如果启用了路由功能，则处理参数以进行路由
        if _routing_enabled():
            routed_params = process_routing(self, "fit_transform", **params)
        else:
            # 否则，设置一个空的 Bunch 对象
            routed_params = Bunch()
            # 遍历 transformer_list 中的每个子转换器
            for name, obj in self.transformer_list:
                if hasattr(obj, "fit_transform"):
                    # 如果子转换器具有 fit_transform 方法，将参数传递给它
                    routed_params[name] = Bunch(fit_transform={})
                    routed_params[name].fit_transform = params
                else:
                    # 如果没有 fit_transform 方法，则设置 fit 和 transform 方法的参数
                    routed_params[name] = Bunch(fit={})
                    routed_params[name] = Bunch(transform={})
                    routed_params[name].fit = params

        # 使用并行处理函数调用 _fit_transform_one 方法
        results = self._parallel_func(X, y, _fit_transform_one, routed_params)
        # 如果结果为空，则返回一个全为零的数组
        if not results:
            return np.zeros((X.shape[0], 0))

        # 将结果解压缩为 Xs 和 transformers
        Xs, transformers = zip(*results)
        # 更新转换器列表
        self._update_transformer_list(transformers)

        # 返回连接后的结果
        return self._hstack(Xs)

    # 定义一个方法用于生成日志消息
    def _log_message(self, name, idx, total):
        # 如果不需要详细信息，则返回 None
        if not self.verbose:
            return None
        # 否则返回格式化后的日志消息字符串
        return "(step %d of %d) Processing %s" % (idx, total, name)
    # 定义一个并行执行函数，用于在输入数据 X 和 y 上并行运行指定的 func 函数
    def _parallel_func(self, X, y, func, routed_params):
        # 将 transformer_list 转换为列表形式
        self.transformer_list = list(self.transformer_list)
        # 验证 transformer_list 中的转换器是否有效
        self._validate_transformers()
        # 验证转换器权重的有效性
        self._validate_transformer_weights()
        # 获取转换器的迭代器
        transformers = list(self._iter())

        # 使用 Parallel 对象并行执行 func 函数，传入每个转换器及其相关参数
        return Parallel(n_jobs=self.n_jobs)(
            delayed(func)(
                transformer,
                X,
                y,
                weight,
                message_clsname="FeatureUnion",  # 日志消息的类名
                message=self._log_message(name, idx, len(transformers)),  # 记录日志消息
                params=routed_params[name],  # 通过路由参数传递给 func 函数的参数
            )
            for idx, (name, transformer, weight) in enumerate(transformers, 1)
        )

    # 对输入数据 X 进行转换，由每个转换器单独进行转换，然后将结果连接起来
    def transform(self, X, **params):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        **params : dict, default=None

            Parameters routed to the `transform` method of the sub-transformers via the
            metadata routing API. See :ref:`Metadata Routing User Guide
            <metadata_routing>` for more details.

            .. versionadded:: 1.5

        Returns
        -------
        X_t : array-like or sparse matrix of shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        # 检查参数的有效性
        _raise_for_params(params, self, "transform")

        # 如果启用了路由功能，则对参数进行处理
        if _routing_enabled():
            routed_params = process_routing(self, "transform", **params)
        else:
            # TODO(SLEP6): 当不能禁用元数据路由时，删除此处代码
            # 创建一个空的 Bunch 对象来存储路由参数
            routed_params = Bunch()
            # 遍历 transformer_list，为每个转换器创建一个空的 Bunch 对象
            for name, _ in self.transformer_list:
                routed_params[name] = Bunch(transform={})

        # 使用 Parallel 对象并行执行 _transform_one 函数，传入每个转换器及其相关参数
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight, params=routed_params[name])
            for name, trans, weight in self._iter()
        )
        # 如果 Xs 为空，则返回一个全零数组，其形状为 (X 的行数, 0)
        if not Xs:
            return np.zeros((X.shape[0], 0))

        # 调用内部方法 _hstack 对 Xs 中的结果进行水平堆叠
        return self._hstack(Xs)

    # 对输入的 Xs 中的数据进行水平堆叠
    def _hstack(self, Xs):
        # 获取容器适配器，用于检查是否支持容器类型
        adapter = _get_container_adapter("transform", self)
        # 如果适配器存在且所有 Xs 中的数据都是支持的容器类型，则调用适配器的 hstack 方法
        if adapter and all(adapter.is_supported_container(X) for X in Xs):
            return adapter.hstack(Xs)

        # 如果 Xs 中有稀疏矩阵，则使用 sparse.hstack 进行稀疏矩阵的水平堆叠
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            # 否则使用 np.hstack 进行普通矩阵的水平堆叠
            Xs = np.hstack(Xs)
        return Xs

    # 更新 transformer_list 中的转换器列表
    def _update_transformer_list(self, transformers):
        # 创建转换器的迭代器
        transformers = iter(transformers)
        # 更新 transformer_list 中每个转换器的定义
        self.transformer_list[:] = [
            (name, old if old == "drop" else next(transformers))
            for name, old in self.transformer_list
        ]
    # 返回第一个转换器的特征数量，作为整个特征联合的特征数量
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""

        # 将 X 传递给所有的转换器，因此我们委托给第一个转换器
        return self.transformer_list[0][1].n_features_in_

    @property
    # 返回第一个转换器的特征名称列表，作为整个特征联合的特征名称列表
    def feature_names_in_(self):
        """Names of features seen during :term:`fit`."""
        # X 被传递给所有的转换器 -- 委托给第一个转换器
        return self.transformer_list[0][1].feature_names_in_

    # 检查特征联合是否已经被拟合
    def __sklearn_is_fitted__(self):
        # 委托检查每个转换器是否已经被拟合
        for _, transformer, _ in self._iter():
            check_is_fitted(transformer)
        return True

    # 返回一个视觉化的并行转换块对象
    def _sk_visual_block_(self):
        names, transformers = zip(*self.transformer_list)
        return _VisualBlock("parallel", transformers, names=names)

    # 根据名称获取特定的转换器对象
    def __getitem__(self, name):
        """Return transformer with name."""
        if not isinstance(name, str):
            raise KeyError("Only string keys are supported")
        return self.named_transformers[name]

    # 获取对象的元数据路由信息
    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.5

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个元数据路由器对象，并配置每个转换器的调用映射
        router = MetadataRouter(owner=self.__class__.__name__)

        for name, transformer in self.transformer_list:
            router.add(
                **{name: transformer},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit_transform", callee="fit_transform")
                .add(caller="fit_transform", callee="fit")
                .add(caller="fit_transform", callee="transform")
                .add(caller="transform", callee="transform"),
            )

        return router
def make_union(*transformers, n_jobs=None, verbose=False):
    """
    Construct a :class:`FeatureUnion` from the given transformers.

    This is a shorthand for the :class:`FeatureUnion` constructor; it does not
    require, and does not permit, naming the transformers. Instead, they will
    be given names automatically based on their types. It also does not allow
    weighting.

    Parameters
    ----------
    *transformers : list of estimators
        One or more estimators.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Returns
    -------
    f : FeatureUnion
        A :class:`FeatureUnion` object for concatenating the results of multiple
        transformer objects.

    See Also
    --------
    FeatureUnion : Class for concatenating the results of multiple transformer
        objects.

    Examples
    --------
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> from sklearn.pipeline import make_union
    >>> make_union(PCA(), TruncatedSVD())
     FeatureUnion(transformer_list=[('pca', PCA()),
                                   ('truncatedsvd', TruncatedSVD())])
    """
    # 使用内部函数 `_name_estimators` 来自动为每个转换器生成名称，并创建 FeatureUnion 对象
    return FeatureUnion(_name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)
```