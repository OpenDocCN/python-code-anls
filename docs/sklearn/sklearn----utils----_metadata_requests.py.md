# `D:\src\scipysrc\scikit-learn\sklearn\utils\_metadata_requests.py`

```
"""
Metadata Routing Utility

In order to better understand the components implemented in this file, one
needs to understand their relationship to one another.

The only relevant public API for end users are the ``set_{method}_request``,
e.g. ``estimator.set_fit_request(sample_weight=True)``. However, third-party
developers and users who implement custom meta-estimators, need to deal with
the objects implemented in this file.

All estimators (should) implement a ``get_metadata_routing`` method, returning
the routing requests set for the estimator. This method is automatically
implemented via ``BaseEstimator`` for all simple estimators, but needs a custom
implementation for meta-estimators.

In non-routing consumers, i.e. the simplest case, e.g. ``SVM``,
``get_metadata_routing`` returns a ``MetadataRequest`` object.

In routers, e.g. meta-estimators and a multi metric scorer,
``get_metadata_routing`` returns a ``MetadataRouter`` object.

An object which is both a router and a consumer, e.g. a meta-estimator which
consumes ``sample_weight`` and routes ``sample_weight`` to its sub-estimators,
routing information includes both information about the object itself (added
via ``MetadataRouter.add_self_request``), as well as the routing information
for its sub-estimators.

A ``MetadataRequest`` instance includes one ``MethodMetadataRequest`` per
method in ``METHODS``, which includes ``fit``, ``score``, etc.

Request values are added to the routing mechanism by adding them to
``MethodMetadataRequest`` instances, e.g.
``metadatarequest.fit.add(param="sample_weight", alias="my_weights")``. This is
used in ``set_{method}_request`` which are automatically generated, so users
and developers almost never need to directly call methods on a
``MethodMetadataRequest``.

The ``alias`` above in the ``add`` method has to be either a string (an alias),
or a {True (requested), False (unrequested), None (error if passed)}``. There
are some other special values such as ``UNUSED`` and ``WARN`` which are used
for purposes such as warning of removing a metadata in a child class, but not
used by the end users.

``MetadataRouter`` includes information about sub-objects' routing and how
methods are mapped together. For instance, the information about which methods
of a sub-estimator are called in which methods of the meta-estimator are all
stored here. Conceptually, this information looks like:

{
    "sub_estimator1": (
        mapping=[(caller="fit", callee="transform"), ...],
        router=MetadataRequest(...),  # or another MetadataRouter
    ),
    ...
}

To give the above representation some structure, we use the following objects:

- ``(caller, callee)`` is a namedtuple called ``MethodPair``

- The list of ``MethodPair`` stored in the ``mapping`` field is a
  ``MethodMapping`` object

- ``(mapping=..., router=...)`` is a namedtuple called ``RouterMappingPair``

The ``set_{method}_request`` methods are dynamically generated for estimators

"""

# 声明 Metadata Routing Utility 的类或模块开始
class MetadataRoutingUtility:
    """
    Utility class for managing metadata routing in estimators and meta-estimators.
    """

    def __init__(self):
        """
        Initialize the MetadataRoutingUtility.
        """

    def set_fit_request(self, **kwargs):
        """
        Dynamically sets the fit request parameters for the estimator.

        Keyword arguments:
        **kwargs -- parameters to be set for the fit request
        """

    def set_score_request(self, **kwargs):
        """
        Dynamically sets the score request parameters for the estimator.

        Keyword arguments:
        **kwargs -- parameters to be set for the score request
        """

    def get_metadata_routing(self):
        """
        Returns the metadata routing information for the estimator.

        This method should be implemented in all estimators and meta-estimators.

        Returns:
        metadata -- routing information encapsulated in a MetadataRequest or MetadataRouter object
        """
        pass

    def add_self_request(self, **kwargs):
        """
        Adds routing information about the object itself.

        Keyword arguments:
        **kwargs -- parameters to be added for self-routing
        """

    def add_sub_estimator_routing(self, sub_estimator, mapping, router):
        """
        Adds routing information about a sub-estimator.

        Arguments:
        sub_estimator -- name or reference to the sub-estimator
        mapping -- method mapping specifying caller and callee relationships
        router -- MetadataRequest or MetadataRouter for the sub-estimator's routing
        """
"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import inspect  # 导入 inspect 模块，用于获取对象信息
from collections import namedtuple  # 导入 namedtuple 类型，用于创建命名元组
from copy import deepcopy  # 导入 deepcopy 函数，用于深拷贝对象
from typing import TYPE_CHECKING, Optional, Union  # 导入类型提示相关模块
from warnings import warn  # 导入 warn 函数，用于发出警告

from .. import get_config  # 从上级目录导入 get_config 函数
from ..exceptions import UnsetMetadataPassedError  # 导入异常类 UnsetMetadataPassedError
from ._bunch import Bunch  # 从当前目录导入 Bunch 类

# Only the following methods are supported in the routing mechanism. Adding new
# methods at the moment involves monkeypatching this list.
# Note that if this list is changed or monkeypatched, the corresponding method
# needs to be added under a TYPE_CHECKING condition like the one done here in
# _MetadataRequester
SIMPLE_METHODS = [
    "fit",
    "partial_fit",
    "predict",
    "predict_proba",
    "predict_log_proba",
    "decision_function",
    "score",
    "split",
    "transform",
    "inverse_transform",
]

# These methods are a composite of other methods and one cannot set their
# requests directly. Instead they should be set by setting the requests of the
# simple methods which make the composite ones.
COMPOSITE_METHODS = {
    "fit_transform": ["fit", "transform"],
    "fit_predict": ["fit", "predict"],
}

METHODS = SIMPLE_METHODS + list(COMPOSITE_METHODS.keys())


def _routing_enabled():
    """Return whether metadata routing is enabled.

    .. versionadded:: 1.3

    Returns
    -------
    enabled : bool
        Whether metadata routing is enabled. If the config is not set, it
        defaults to False.
    """
    return get_config().get("enable_metadata_routing", False)  # 返回是否启用元数据路由的配置值


def _raise_for_params(params, owner, method):
    """Raise an error if metadata routing is not enabled and params are passed.

    .. versionadded:: 1.4

    Parameters
    ----------
    params : dict
        The metadata passed to a method.

    owner : object
        The object to which the method belongs.

    method : str
        The name of the method, e.g. "fit".

    Raises
    ------
    ValueError
        If metadata routing is not enabled and params are passed.
    """
    caller = (
        f"{owner.__class__.__name__}.{method}" if method else owner.__class__.__name__
    )  # 获取调用者的名称和方法名（如果存在）
    if not _routing_enabled() and params:  # 如果未启用元数据路由且存在传入参数
        raise ValueError(
            f"Passing extra keyword arguments to {caller} is only supported if"
            " enable_metadata_routing=True, which you can set using"
            " `sklearn.set_config`. See the User Guide"
            " <https://scikit-learn.org/stable/metadata_routing.html> for more"
            f" details. Extra parameters passed are: {set(params)}"
        )  # 抛出数值错误，指出只有在启用元数据路由时才支持传递额外关键字参数


def _raise_for_unsupported_routing(obj, method, **kwargs):
    """Raise when metadata routing is enabled and metadata is passed.

    This is used in meta-estimators which have not implemented metadata routing
    to prevent silent bugs. There is no need to use this function if the
    meta-estimator is not accepting any metadata, especially in `fit`, since
    if a meta-estimator accepts any metadata, they would do that in `fit` as
    well.

    Parameters
    ----------
    obj : estimator
        The estimator for which we're raising the error.

    method : str
        The method where the error is raised.

    **kwargs : dict
        The metadata passed to the method.
    """
    # Filter out any None values from kwargs
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    
    # Check if metadata routing is enabled and if kwargs is not empty
    if _routing_enabled() and kwargs:
        # Get the name of the class of the estimator
        cls_name = obj.__class__.__name__
        
        # Raise a NotImplementedError with a descriptive message
        raise NotImplementedError(
            f"{cls_name}.{method} cannot accept given metadata ({set(kwargs.keys())})"
            f" since metadata routing is not yet implemented for {cls_name}."
        )
class _RoutingNotSupportedMixin:
    """A mixin to be used to remove the default `get_metadata_routing`.

    This is used in meta-estimators where metadata routing is not yet
    implemented.

    This also makes it clear in our rendered documentation that this method
    cannot be used.
    """

    def get_metadata_routing(self):
        """Raise `NotImplementedError`.

        This estimator does not support metadata routing yet."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented metadata routing yet."
        )
    def __init__(self, owner, method, requests=None):
        """
        Initialize the RequestHandler object.

        Parameters
        ----------
        owner : object
            The owner of the request handler.
        method : object
            The method associated with the request handler.
        requests : dict, optional
            Dictionary to store requests, defaults to an empty dictionary.
        """
        self._requests = requests or dict()  # Initialize _requests with provided dictionary or empty if None
        self.owner = owner  # Assign owner attribute
        self.method = method  # Assign method attribute

    @property
    def requests(self):
        """
        Dictionary of requests associated with the handler.

        Returns
        -------
        dict
            Dictionary containing request information.
        """
        return self._requests  # Return the requests dictionary

    def add_request(
        self,
        *,
        param,
        alias,
    ):
        """
        Add request information for a metadata parameter.

        Parameters
        ----------
        param : str
            The parameter for which a request is being set.
        alias : str or {True, False, None}
            Specifies metadata routing:

            - str: Alias of metadata routed to `param`.
            - True: Indicates request is True.
            - False: Indicates request is False.
            - None: Raises error if passed.
        
        Returns
        -------
        self
            Returns the instance of the object for method chaining.
        """
        if not request_is_alias(alias) and not request_is_valid(alias):
            raise ValueError(
                f"The alias you're setting for `{param}` should be either a "
                "valid identifier or one of {None, True, False}, but given "
                f"value is: `{alias}`"
            )

        if alias == param:
            alias = True

        if alias == UNUSED:
            if param in self._requests:
                del self._requests[param]
            else:
                raise ValueError(
                    f"Trying to remove parameter {param} with UNUSED which doesn't"
                    " exist."
                )
        else:
            self._requests[param] = alias

        return self  # Return self for method chaining

    def _get_param_names(self, return_alias):
        """
        Get names of all metadata that can be consumed or routed by this method.

        This method returns names of all metadata, including `False`.

        Parameters
        ----------
        return_alias : bool
            Controls whether original or aliased names should be returned. If
            `False`, aliases are ignored and original names are returned.

        Returns
        -------
        set of str
            A set of strings with the names of all parameters.
        """
        return set(
            alias if return_alias and not request_is_valid(alias) else prop
            for prop, alias in self._requests.items()
            if not request_is_valid(alias) or alias is not False
        )
    # 检查传入的参数中是否包含标记为 WARN 的元数据，如果有则发出警告。
    def _check_warnings(self, *, params):
        """Check whether metadata is passed which is marked as WARN.

        If any metadata is passed which is marked as WARN, a warning is raised.

        Parameters
        ----------
        params : dict
            The metadata passed to a method.
        """
        # 如果 params 为 None，则将其设为一个空字典
        params = {} if params is None else params
        # 从 self._requests 字典中筛选出所有标记为 WARN 并且在 params 中存在的属性
        warn_params = {
            prop
            for prop, alias in self._requests.items()
            if alias == WARN and prop in params
        }
        # 遍历所有标记为 WARN 的参数，并发出警告
        for param in warn_params:
            warn(
                f"Support for {param} has recently been added to this class. "
                "To maintain backward compatibility, it is ignored now. "
                f"Using `set_{self.method}_request({param}={{True, False}})` "
                "on this method of the class, you can set the request value "
                "to False to silence this warning, or to True to consume and "
                "use the metadata."
            )
    def _route_params(self, params, parent, caller):
        """准备给定的参数以便传递给方法。

        此方法的输出可以直接用作额外的属性传递给对应的方法。

        Parameters
        ----------
        params : dict
            提供的元数据字典。

        parent : object
            父类对象，路由元数据的来源。

        caller : str
            调用者的方法名，元数据从这里路由出去。

        Returns
        -------
        params : Bunch
            一个:class:`~sklearn.utils.Bunch`，包含{prop: value}，可以传递给对应的方法。
        """
        self._check_warnings(params=params)  # 调用本对象的 _check_warnings 方法，检查警告
        unrequested = dict()  # 初始化一个空字典用于存储未请求的参数
        args = {arg: value for arg, value in params.items() if value is not None}  # 过滤掉值为 None 的参数，构建参数字典 args
        res = Bunch()  # 创建一个 Bunch 对象，用于存储结果
        for prop, alias in self._requests.items():
            if alias is False or alias == WARN:
                continue  # 如果 alias 为 False 或 WARN，跳过当前迭代
            elif alias is True and prop in args:
                res[prop] = args[prop]  # 如果 alias 为 True 并且 prop 在 args 中，则将其加入结果 Bunch
            elif alias is None and prop in args:
                unrequested[prop] = args[prop]  # 如果 alias 为 None 并且 prop 在 args 中，则将其加入未请求的参数字典
            elif alias in args:
                res[prop] = args[alias]  # 如果 alias 在 args 中，则将其对应的值加入结果 Bunch
        if unrequested:
            if self.method in COMPOSITE_METHODS:
                callee_methods = COMPOSITE_METHODS[self.method]  # 获取复合方法中的方法列表
            else:
                callee_methods = [self.method]  # 否则，使用当前方法构成列表
            set_requests_on = "".join(
                [
                    f".set_{method}_request({{metadata}}=True/False)"
                    for method in callee_methods
                ]
            )  # 构建设置请求的字符串模板
            message = (
                f"[{', '.join([key for key in unrequested])}] are passed but are not"
                " explicitly set as requested or not requested for"
                f" {self.owner}.{self.method}, which is used within"
                f" {parent}.{caller}. Call `{self.owner}"
                + set_requests_on
                + "` for each metadata you want to request/ignore."
            )  # 构建错误消息字符串
            raise UnsetMetadataPassedError(
                message=message,
                unrequested_params=unrequested,
                routed_params=res,
            )  # 抛出未设置元数据传递错误，包括错误消息、未请求的参数和路由的参数
        return res  # 返回结果 Bunch

    def _consumes(self, params):
        """检查给定的参数是否被此方法消耗。

        Parameters
        ----------
        params : iterable of str
            要检查的参数迭代器。

        Returns
        -------
        consumed : set of str
            被此方法消耗的参数集合。
        """
        params = set(params)  # 将参数转换为集合
        res = set()  # 初始化一个空集合，用于存储被消耗的参数
        for prop, alias in self._requests.items():
            if alias is True and prop in params:
                res.add(prop)  # 如果 alias 为 True 并且 prop 在 params 中，则将其加入结果集合
            elif isinstance(alias, str) and alias in params:
                res.add(alias)  # 如果 alias 是字符串并且在 params 中，则将其加入结果集合
        return res  # 返回被消耗的参数集合
    # 定义一个方法 `_serialize`，用于序列化对象
    def _serialize(self):
        """Serialize the object.

        Returns
        -------
        obj : dict
            A serialized version of the instance in the form of a dictionary.
        """
        # 返回对象的请求信息 `_requests`，以字典形式序列化的结果
        return self._requests

    # 定义一个特殊方法 `__repr__`，返回对象的字符串表示形式
    def __repr__(self):
        # 调用 `_serialize` 方法，并将其结果转换为字符串返回
        return str(self._serialize())

    # 定义一个特殊方法 `__str__`，返回对象的字符串表示形式
    def __str__(self):
        # 调用 `__repr__` 方法获取对象的字符串表示形式，并返回其字符串形式
        return str(repr(self))
class MetadataRequest:
    """Contains the metadata request info of a consumer.

    Instances of `MethodMetadataRequest` are used in this class for each
    available method under `metadatarequest.{method}`.

    Consumer-only classes such as simple estimators return a serialized
    version of this class as the output of `get_metadata_routing()`.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : str
        The name of the object to which these requests belong.
    """

    # this is here for us to use this attribute's value instead of doing
    # `isinstance` in our checks, so that we avoid issues when people vendor
    # this file instead of using it directly from scikit-learn.
    _type = "metadata_request"  # 定义了一个类属性 `_type`，用于在检查时代替 `isinstance` 的使用，以避免在使用此文件时出现问题。

    def __init__(self, owner):
        self.owner = owner  # 初始化实例属性 `owner`，表示这些请求所属的对象的名称
        for method in SIMPLE_METHODS:  # 遍历 `SIMPLE_METHODS` 列表中的每个方法名
            setattr(
                self,
                method,
                MethodMetadataRequest(owner=owner, method=method),  # 使用 `MethodMetadataRequest` 初始化每个方法对应的属性
            )

    def consumes(self, method, params):
        """Check whether the given parameters are consumed by the given method.

        .. versionadded:: 1.4

        Parameters
        ----------
        method : str
            The name of the method to check.

        params : iterable of str
            An iterable of parameters to check.

        Returns
        -------
        consumed : set of str
            A set of parameters which are consumed by the given method.
        """
        return getattr(self, method)._consumes(params=params)  # 调用特定方法对应属性的 `_consumes` 方法，并传入参数进行检查
    # 当默认属性访问失败并引发 AttributeError 时调用此方法
    # （即 __getattribute__() 因 name 不是实例属性或者 self 的类树中的属性，
    # 或者因为 name 属性是一个属性且 __get__() 引发 AttributeError）。
    # 此方法应返回计算得到的属性值或引发 AttributeError 异常。
    # https://docs.python.org/3/reference/datamodel.html#object.__getattr__
    def __getattr__(self, name):
        # 如果 name 不在 COMPOSITE_METHODS 中，则引发 AttributeError 异常
        if name not in COMPOSITE_METHODS:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # 初始化一个空字典 requests
        requests = {}

        # 遍历 COMPOSITE_METHODS[name] 中的每个方法
        for method in COMPOSITE_METHODS[name]:
            # 获取 self 对象中名为 method 的属性值
            mmr = getattr(self, method)
            # 获取当前 requests 字典中已有的键集合
            existing = set(requests.keys())
            # 获取 mmr 对象中 requests 字典的键集合
            upcoming = set(mmr.requests.keys())
            # 计算 existing 和 upcoming 的交集
            common = existing & upcoming
            # 找出存在冲突的键列表，即两者具有相同键但值不同的情况
            conflicts = [key for key in common if requests[key] != mmr._requests[key]]
            # 如果存在冲突，则引发 ValueError 异常
            if conflicts:
                raise ValueError(
                    f"Conflicting metadata requests for {', '.join(conflicts)} while"
                    f" composing the requests for {name}. Metadata with the same name"
                    f" for methods {', '.join(COMPOSITE_METHODS[name])} should have the"
                    " same request value."
                )
            # 更新 requests 字典，将 mmr 对象中的 _requests 字典合并进去
            requests.update(mmr._requests)

        # 返回一个 MethodMetadataRequest 对象，包含所有请求的元数据
        return MethodMetadataRequest(owner=self.owner, method=name, requests=requests)

    def _get_param_names(self, method, return_alias, ignore_self_request=None):
        """Get names of all metadata that can be consumed or routed by specified \
            method.

        This method returns the names of all metadata, even the ``False``
        ones.

        Parameters
        ----------
        method : str
            The name of the method for which metadata names are requested.

        return_alias : bool
            Controls whether original or aliased names should be returned. If
            ``False``, aliases are ignored and original names are returned.

        ignore_self_request : bool
            Ignored. Present for API compatibility.

        Returns
        -------
        names : set of str
            A set of strings with the names of all parameters.
        """
        # 调用 self 对象中名为 method 的方法的 _get_param_names 方法，并返回其结果
        return getattr(self, method)._get_param_names(return_alias=return_alias)
    def _route_params(self, *, params, method, parent, caller):
        """准备给定的参数以便传递给方法。

        此方法的输出可以直接用作额外的关键字参数，传递元数据给相应的方法。

        Parameters
        ----------
        params : dict
            提供的元数据字典。

        method : str
            请求和路由参数的方法的名称。

        parent : object
            父类对象，路由元数据。

        caller : str
            来自父类对象的方法，元数据从中路由。

        Returns
        -------
        params : Bunch
            一个 :class:`~sklearn.utils.Bunch`，其中包含{prop: value}，可以提供给相应的方法。
        """
        return getattr(self, method)._route_params(
            params=params, parent=parent, caller=caller
        )

    def _check_warnings(self, *, method, params):
        """检查是否传递了标记为WARN的元数据。

        如果传递了任何标记为WARN的元数据，则会引发警告。

        Parameters
        ----------
        method : str
            应检查警告的方法的名称。

        params : dict
            传递给方法的元数据。
        """
        getattr(self, method)._check_warnings(params=params)

    def _serialize(self):
        """序列化对象。

        Returns
        -------
        obj : dict
            实例的序列化版本，以字典形式返回。
        """
        output = dict()
        for method in SIMPLE_METHODS:
            mmr = getattr(self, method)
            if len(mmr.requests):
                output[method] = mmr._serialize()
        return output

    def __repr__(self):
        return str(self._serialize())

    def __str__(self):
        return str(repr(self))
# Metadata Request for Routers
# ============================
# This section includes all objects required for MetadataRouter which is used
# in routers, returned by their ``get_metadata_routing``.

# This namedtuple is used to store a (mapping, routing) pair. Mapping is a
# MethodMapping object, and routing is the output of `get_metadata_routing`.
# MetadataRouter stores a collection of these namedtuples.
RouterMappingPair = namedtuple("RouterMappingPair", ["mapping", "router"])

# A namedtuple storing a single method route. A collection of these namedtuples
# is stored in a MetadataRouter.
MethodPair = namedtuple("MethodPair", ["caller", "callee"])


class MethodMapping:
    """Stores the mapping between caller and callee methods for a router.

    This class is primarily used in a ``get_metadata_routing()`` of a router
    object when defining the mapping between a sub-object (a sub-estimator or a
    scorer) to the router's methods. It stores a collection of namedtuples.

    Iterating through an instance of this class will yield named
    ``MethodPair(caller, callee)`` tuples.

    .. versionadded:: 1.3
    """

    def __init__(self):
        # Initialize an empty list to store method route mappings
        self._routes = []

    def __iter__(self):
        # Returns an iterator over the method route mappings
        return iter(self._routes)

    def add(self, *, caller, callee):
        """Add a method mapping.

        Parameters
        ----------

        caller : str
            Parent estimator's method name in which the ``callee`` is called.

        callee : str
            Child object's method name. This method is called in ``caller``.

        Returns
        -------
        self : MethodMapping
            Returns self.
        """
        # Check if caller and callee methods are valid
        if caller not in METHODS:
            raise ValueError(
                f"Given caller:{caller} is not a valid method. Valid methods are:"
                f" {METHODS}"
            )
        if callee not in METHODS:
            raise ValueError(
                f"Given callee:{callee} is not a valid method. Valid methods are:"
                f" {METHODS}"
            )
        # Add the method pair (caller, callee) to the list of routes
        self._routes.append(MethodPair(caller=caller, callee=callee))
        return self

    def _serialize(self):
        """Serialize the object.

        Returns
        -------
        obj : list
            A serialized version of the instance in the form of a list.
        """
        # Serialize method route mappings into a list of dictionaries
        result = list()
        for route in self._routes:
            result.append({"caller": route.caller, "callee": route.callee})
        return result

    def __repr__(self):
        # Return a string representation of the serialized method route mappings
        return str(self._serialize())

    def __str__(self):
        # Return a string representation of the object
        return str(repr(self))


class MetadataRouter:
    """Stores and handles metadata routing for a router object.

    This class is used by router objects to store and handle metadata routing.
    Routing information is stored as a dictionary of the form ``{"object_name":
    RouteMappingPair(method_mapping, routing_info)}``, where ``method_mapping``
    """
    # 这是用来在检查中使用这个属性的值，而不是在我们的检查中使用 `isinstance`，
    # 这样可以避免在人们从 scikit-learn 直接使用时，将这个文件捆绑进去而产生的问题。
    _type = "metadata_router"

    def __init__(self, owner):
        # 用于存储路由映射的字典
        self._route_mappings = dict()
        # `_self_request` 在路由器本身也是消费者时使用。
        # `_self_request` 是通过 `add_self_request()` 方法添加的，
        # 与存储在 `_route_mappings` 中的其他对象不同对待。
        self._self_request = None
        # 初始化所有者的名称
        self.owner = owner

    def add_self_request(self, obj):
        """Add `self` (as a consumer) to the routing.

        This method is used if the router is also a consumer, and hence the
        router itself needs to be included in the routing. The passed object
        can be an estimator or a
        :class:`~sklearn.utils.metadata_routing.MetadataRequest`.

        A router should add itself using this method instead of `add` since it
        should be treated differently than the other objects to which metadata
        is routed by the router.

        Parameters
        ----------
        obj : object
            This is typically the router instance, i.e. `self` in a
            ``get_metadata_routing()`` implementation. It can also be a
            ``MetadataRequest`` instance.

        Returns
        -------
        self : MetadataRouter
            Returns `self`.
        """
        # 如果传入的 `obj` 是 `metadata_request` 类型，则深度复制给 `_self_request`
        if getattr(obj, "_type", None) == "metadata_request":
            self._self_request = deepcopy(obj)
        # 如果 `obj` 具有 `_get_metadata_request` 属性，则调用其获取方法，并深度复制给 `_self_request`
        elif hasattr(obj, "_get_metadata_request"):
            self._self_request = deepcopy(obj._get_metadata_request())
        # 如果 `obj` 既不是 `MetadataRequest` 类型，也没有实现所需的 API，则引发 ValueError 异常
        else:
            raise ValueError(
                "Given `obj` is neither a `MetadataRequest` nor does it implement the"
                " required API. Inheriting from `BaseEstimator` implements the required"
                " API."
            )
        # 返回当前对象 `self`，即 MetadataRouter 实例
        return self
    def add(self, *, method_mapping, **objs):
        """
        Add named objects with their corresponding method mapping.

        Parameters
        ----------
        method_mapping : MethodMapping
            The mapping between the child and the parent's methods.

        **objs : dict
            A dictionary of objects from which metadata is extracted by calling
            :func:`~sklearn.utils.metadata_routing.get_routing_for_object` on them.

        Returns
        -------
        self : MetadataRouter
            Returns `self`.
        """
        # 创建 method_mapping 的深层副本，确保不会影响原始对象
        method_mapping = deepcopy(method_mapping)

        # 遍历传入的对象字典，将每个对象与其方法映射加入到路由映射表中
        for name, obj in objs.items():
            self._route_mappings[name] = RouterMappingPair(
                mapping=method_mapping, router=get_routing_for_object(obj)
            )

        # 返回当前对象的引用
        return self

    def consumes(self, method, params):
        """
        Check whether the given parameters are consumed by the given method.

        .. versionadded:: 1.4

        Parameters
        ----------
        method : str
            The name of the method to check.

        params : iterable of str
            An iterable of parameters to check.

        Returns
        -------
        consumed : set of str
            A set of parameters which are consumed by the given method.
        """
        # 初始化结果集
        res = set()

        # 如果存在 self._self_request 对象，则调用其 consumes 方法检查参数
        if self._self_request:
            res = res | self._self_request.consumes(method=method, params=params)

        # 遍历路由映射表中的每一对映射
        for _, route_mapping in self._route_mappings.items():
            for caller, callee in route_mapping.mapping:
                # 如果找到与传入 method 匹配的 caller 方法
                if caller == method:
                    # 调用对应的路由器对象的 consumes 方法，检查参数
                    res = res | route_mapping.router.consumes(
                        method=callee, params=params
                    )

        # 返回参数被消耗的集合
        return res
    # 定义一个方法来获取指定方法可消耗或路由的所有元数据名称集合。
    # 返回的集合包括所有元数据名称，即使是`False`的名称也包括在内。
    def _get_param_names(self, *, method, return_alias, ignore_self_request):
        """Get names of all metadata that can be consumed or routed by specified \
            method.

        This method returns the names of all metadata, even the ``False``
        ones.

        Parameters
        ----------
        method : str
            The name of the method for which metadata names are requested.

        return_alias : bool
            Controls whether original or aliased names should be returned,
            which only applies to the stored `self`. If no `self` routing
            object is stored, this parameter has no effect.

        ignore_self_request : bool
            If `self._self_request` should be ignored. This is used in `_route_params`.
            If ``True``, ``return_alias`` has no effect.

        Returns
        -------
        names : set of str
            A set of strings with the names of all parameters.
        """
        # 初始化一个空集合来存储结果
        res = set()
        
        # 如果存在self._self_request并且不应该忽略self._self_request
        if self._self_request and not ignore_self_request:
            # 将self._self_request._get_param_names方法返回的结果并入res集合中
            res = res.union(
                self._self_request._get_param_names(
                    method=method, return_alias=return_alias
                )
            )

        # 遍历self._route_mappings.items()，其中每个元素是(name, route_mapping)
        for name, route_mapping in self._route_mappings.items():
            # 遍历route_mapping.mapping里的每个(caller, callee)对
            for caller, callee in route_mapping.mapping:
                # 如果caller等于指定的method
                if caller == method:
                    # 将route_mapping.router._get_param_names方法返回的结果并入res集合中
                    res = res.union(
                        route_mapping.router._get_param_names(
                            method=callee, return_alias=True, ignore_self_request=False
                        )
                    )
        
        # 返回最终的结果集合res，其中包含所有参数名称
        return res
    def _route_params(self, *, params, method, parent, caller):
        """Prepare the given parameters to be passed to the method.

        This is used when a router is used as a child object of another router.
        The parent router then passes all parameters understood by the child
        object to it and delegates their validation to the child.

        The output of this method can be used directly as the input to the
        corresponding method as extra props.

        Parameters
        ----------
        params : dict
            A dictionary of provided metadata.

        method : str
            The name of the method for which the parameters are requested and
            routed.

        parent : object
            Parent class object, that routes the metadata.

        caller : str
            Method from the parent class object, where the metadata is routed from.

        Returns
        -------
        params : Bunch
            A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to the
            corresponding method.
        """
        # Initialize an empty Bunch object to store the result
        res = Bunch()
        
        # If self._self_request is True, recursively call _route_params on self._self_request
        if self._self_request:
            res.update(
                self._self_request._route_params(
                    params=params,
                    method=method,
                    parent=parent,
                    caller=caller,
                )
            )

        # Obtain parameter names for the current method using _get_param_names
        param_names = self._get_param_names(
            method=method, return_alias=True, ignore_self_request=True
        )
        
        # Filter out parameters from `params` that are not in `param_names`
        child_params = {
            key: value for key, value in params.items() if key in param_names
        }
        
        # Check for conflicts between `res` and `child_params`
        for key in set(res.keys()).intersection(child_params.keys()):
            # Raise ValueError if conflicting objects are different
            if child_params[key] is not res[key]:
                raise ValueError(
                    f"In {self.owner}, there is a conflict on {key} between what is"
                    " requested for this estimator and what is requested by its"
                    " children. You can resolve this conflict by using an alias for"
                    " the child estimator(s) requested metadata."
                )

        # Update `res` with `child_params`
        res.update(child_params)
        
        # Return the resulting Bunch object `res`
        return res
    # 返回由子对象请求的输入参数

    # 该方法的输出是一个:class:`~sklearn.utils.Bunch`对象，其中包含了路由器的`caller`方法所使用的每个子对象方法的元数据。
    
    # 如果路由器本身也是消费者，还会检查`self`/消费者请求的元数据警告。

    # Parameters 参数:
    # caller : str
    #     请求参数并路由的方法名称。如果在路由器的:term:`fit`方法中调用，则为`"fit"`。
    # params : dict
    #     提供的元数据字典。

    # Returns 返回:
    # Bunch
    #     一个:class:`~sklearn.utils.Bunch`对象，形式为
    #     ``{"object_name": {"method_name": {params: value}}}``，可用于将所需的元数据传递给相应的方法或子对象。
    
    if self._self_request:
        # 如果存在self._self_request对象，则检查其警告信息
        self._self_request._check_warnings(params=params, method=caller)
    
    # 初始化一个空的Bunch对象，用于存储路由结果
    res = Bunch()

    # 遍历路由映射字典中的每个条目
    for name, route_mapping in self._route_mappings.items():
        # 解包路由器和映射
        router, mapping = route_mapping.router, route_mapping.mapping

        # 为当前路由映射名称初始化一个空的Bunch对象
        res[name] = Bunch()

        # 遍历映射中的每个调用者-被调用者对
        for _caller, _callee in mapping:
            # 如果调用者与给定的caller匹配
            if _caller == caller:
                # 调用router对象的_route_params方法，获取方法的路由参数
                res[name][_callee] = router._route_params(
                    params=params,
                    method=_callee,
                    parent=self.owner,
                    caller=caller,
                )
    # 返回填充完毕的Bunch对象作为路由参数的结果
    return res


# 验证给定方法的元数据是否有效

# 如果一些传递的元数据未被子对象理解，则会引发``TypeError``异常。

# Parameters 参数:
# method : str
#     要验证其参数的方法名称。如果在路由器的:term:`fit`方法中调用，则为`"fit"`。
# params : dict
#     提供的元数据字典。
def validate_metadata(self, *, method, params):
    # 获取方法的参数名称列表，排除别名，并考虑是否忽略self请求
    param_names = self._get_param_names(
        method=method, return_alias=False, ignore_self_request=False
    )

    # 如果存在self._self_request对象，则获取该对象的参数名称列表
    if self._self_request:
        self_params = self._self_request._get_param_names(
            method=method, return_alias=False
        )
    else:
        self_params = set()

    # 查找在params中存在但未在param_names和self_params中的额外键
    extra_keys = set(params.keys()) - param_names - self_params

    # 如果存在额外的键，则抛出TypeError异常
    if extra_keys:
        raise TypeError(
            f"{self.owner}.{method} got unexpected argument(s) {extra_keys}, which"
            " are not routed to any object."
        )
    # 序列化对象的方法，将对象转换为字典形式
    
    def _serialize(self):
        """Serialize the object.
    
        Returns
        -------
        obj : dict
            A serialized version of the instance in the form of a dictionary.
        """
        # 创建一个空字典作为序列化结果
        res = dict()
        
        # 如果存在 self._self_request 属性，则将其序列化并添加到结果字典中
        if self._self_request:
            res["$self_request"] = self._self_request._serialize()
        
        # 遍历 self._route_mappings 字典的键值对
        for name, route_mapping in self._route_mappings.items():
            # 为每个键创建一个字典
            res[name] = dict()
            # 将 route_mapping.mapping 属性序列化并存储在字典中
            res[name]["mapping"] = route_mapping.mapping._serialize()
            # 将 route_mapping.router 属性序列化并存储在字典中
            res[name]["router"] = route_mapping.router._serialize()
    
        # 返回包含序列化后内容的字典
        return res
    
    
    # 迭代器方法，用于迭代对象的元素
    def __iter__(self):
        # 如果存在 self._self_request 属性，则创建一个 MethodMapping 实例并添加到迭代结果中
        if self._self_request:
            method_mapping = MethodMapping()
            for method in METHODS:
                method_mapping.add(caller=method, callee=method)
            # 返回包含 $self_request 的 RouterMappingPair 对象
            yield "$self_request", RouterMappingPair(
                mapping=method_mapping, router=self._self_request
            )
        
        # 遍历 self._route_mappings 字典的键值对并迭代返回
        for name, route_mapping in self._route_mappings.items():
            yield (name, route_mapping)
    
    
    # 字符串表示方法，返回对象的序列化字符串表示
    def __repr__(self):
        return str(self._serialize())
    
    
    # 打印字符串方法，返回对象的序列化字符串表示
    def __str__(self):
        return str(repr(self))
# 从给定对象获取 ``Metadata{Router, Request}`` 实例的方法。
#
# 此函数根据输入返回一个 :class:`~sklearn.utils.metadata_routing.MetadataRouter` 
# 或 :class:`~sklearn.utils.metadata_routing.MetadataRequest` 的实例。
#
# 此函数始终返回一个从输入构造的副本或实例，因此更改此函数的输出不会改变原始对象。
#
# .. versionadded:: 1.3
#
# Parameters
# ----------
# obj : object
#     - 如果对象提供 `get_metadata_routing` 方法，则返回该方法输出的副本。
#     - 如果对象已经是 :class:`~sklearn.utils.metadata_routing.MetadataRequest` 或
#       :class:`~sklearn.utils.metadata_routing.MetadataRouter`，则返回该对象的副本。
#     - 否则返回一个空的 :class:`~sklearn.utils.metadata_routing.MetadataRequest`。
#
# Returns
# -------
# obj : MetadataRequest or MetadataRouting
#     从给定对象获取或创建的 ``MetadataRequest`` 或 ``MetadataRouting`` 实例。
def get_routing_for_object(obj=None):
    # 使用 hasattr 而不是 try/except，因为 AttributeError 可能由其他原因引发。
    if hasattr(obj, "get_metadata_routing"):
        return deepcopy(obj.get_metadata_routing())  # 返回 `get_metadata_routing` 方法的深拷贝

    elif getattr(obj, "_type", None) in ["metadata_request", "metadata_router"]:
        return deepcopy(obj)  # 返回对象的深拷贝

    return MetadataRequest(owner=None)  # 返回一个空的 MetadataRequest 实例


# Request method
# ==============
# 此部分包括请求方法描述符所需的内容，以及它们在元类中的动态生成。
#
# 这些字符串用于动态生成 set_{method}_request 方法的文档字符串。
REQUESTER_DOC = """        请求传递给 ``{method}`` 方法的元数据。
        
        请注意，仅当 ``enable_metadata_routing=True`` 时（参见 :func:`sklearn.set_config`），
        此方法才相关。请参阅 :ref:`User Guide <metadata_routing>` 了解路由机制的工作方式。
        
        每个参数的选项为：
        
        - ``True``: 请求元数据，并在提供时传递给 ``{method}``。如果未提供元数据，则请求将被忽略。
        
        - ``False``: 不请求元数据，元估计器将不会将其传递给 ``{method}``。
        
        - ``None``: 不请求元数据，如果用户提供了元数据，则元估计器将引发错误。
        
        - ``str``: 应将元数据与 ``{method}`` 一起传递。
"""
Class for handling metadata routing for estimator methods.

.. versionadded:: 1.3

Parameters
----------
name : str
    The name of the method for which the request function should be
    created, e.g. ``"fit"`` would create a ``set_fit_request`` function.
keys : list of str
    A list of strings which are accepted parameters by the created
    function, e.g. ``["sample_weight"]`` if the corresponding method
    accepts it as a metadata.
validate_keys : bool, default=True
    Whether to check if the requested parameters fit the actual parameters
    of the method.

Notes
-----
This class is a descriptor [1]_ and uses PEP-362 to set the signature of
the returned function [2]_.

References
----------
.. [1] https://docs.python.org/3/howto/descriptor.html
.. [2] https://www.python.org/dev/peps/pep-0362/
"""
class RequestMethod:
    """
    A descriptor for request methods.

    .. versionadded:: 1.3

    Parameters
    ----------
    name : str
        The name of the method for which the request function should be
        created, e.g. ``"fit"`` would create a ``set_fit_request`` function.
    keys : list of str
        A list of strings which are accepted parameters by the created
        function, e.g. ``["sample_weight"]`` if the corresponding method
        accepts it as a metadata.
    validate_keys : bool, default=True
        Whether to check if the requested parameters fit the actual parameters
        of the method.

    Notes
    -----
    This class is a descriptor [1]_ and uses PEP-362 to set the signature of
    the returned function [2]_.

    References
    ----------
    .. [1] https://docs.python.org/3/howto/descriptor.html
    .. [2] https://www.python.org/dev/peps/pep-0362/
    """
    def __init__(self, name, keys, validate_keys=True):
        self.name = name
        self.keys = keys
        self.validate_keys = validate_keys

"""
Mixin class for adding metadata request functionality to BaseEstimator.

``BaseEstimator`` inherits from this Mixin.

.. versionadded:: 1.3
"""
class _MetadataRequester:
    """
    Mixin class for adding metadata request functionality.

    ``BaseEstimator`` inherits from this Mixin.

    .. versionadded:: 1.3
    """
    if TYPE_CHECKING:  # pragma: no cover
        # 在运行时永远不会执行这段代码，而是用于类型检查。
        # 类型检查器无法理解动态生成的 `set_{method}_request` 方法，会抱怨其未定义。
        # 我们在这里定义它们，以使类型检查器满意。
        # 在类型检查期间，假定这是真实的。
        # 下面定义的方法列表与 SIMPLE_METHODS 中的方法列表一致。
        # fmt: off
        def set_fit_request(self, **kwargs): pass
        def set_partial_fit_request(self, **kwargs): pass
        def set_predict_request(self, **kwargs): pass
        def set_predict_proba_request(self, **kwargs): pass
        def set_predict_log_proba_request(self, **kwargs): pass
        def set_decision_function_request(self, **kwargs): pass
        def set_score_request(self, **kwargs): pass
        def set_split_request(self, **kwargs): pass
        def set_transform_request(self, **kwargs): pass
        def set_inverse_transform_request(self, **kwargs): pass
        # fmt: on

    def __init_subclass__(cls, **kwargs):
        """设置 ``set_{method}_request`` 方法。

        这里使用 PEP-487 [1]_ 来设置 ``set_{method}_request`` 方法。
        它查找通过 ``__metadata_request__*`` 类属性设置的默认值，
        或从方法签名推断出的信息。

        当一个方法没有显式接受元数据作为参数时，或者开发者希望为那些与默认值 ``None`` 不同的元数据
        指定请求值时，会使用 ``__metadata_request__*`` 类属性。

        参考
        ----------
        .. [1] https://www.python.org/dev/peps/pep-0487
        """
        try:
            requests = cls._get_default_requests()
        except Exception:
            # 如果默认值出现问题，当调用 ``get_metadata_routing`` 时会引发异常。
            # 在这里我们忽略所有的问题，比如坏的默认值等。
            super().__init_subclass__(**kwargs)
            return

        for method in SIMPLE_METHODS:
            mmr = getattr(requests, method)
            # 设置 ``set_{method}_request`` 方法
            if not len(mmr.requests):
                continue
            setattr(
                cls,
                f"set_{method}_request",
                RequestMethod(method, sorted(mmr.requests.keys())),
            )
        super().__init_subclass__(**kwargs)

    @classmethod
    def _build_request_for_signature(cls, router, method):
        """Build the `MethodMetadataRequest` for a method using its signature.

        This method takes all arguments from the method signature and uses
        ``None`` as their default request value, except ``X``, ``y``, ``Y``,
        ``Xt``, ``yt``, ``*args``, and ``**kwargs``.

        Parameters
        ----------
        router : MetadataRequest
            The parent object for the created `MethodMetadataRequest`.
        method : str
            The name of the method.

        Returns
        -------
        method_request : MethodMetadataRequest
            The prepared request using the method's signature.
        """
        # Create a MethodMetadataRequest object with the owner class name and method name
        mmr = MethodMetadataRequest(owner=cls.__name__, method=method)

        # Check if the class `cls` does not have the method `method` or if it's not a function
        # If either condition is true, return the MethodMetadataRequest as is
        if not hasattr(cls, method) or not inspect.isfunction(getattr(cls, method)):
            return mmr

        # Get the parameters of the method's signature, excluding the first parameter (usually "self")
        params = list(inspect.signature(getattr(cls, method)).parameters.items())[1:]

        # Iterate over each parameter and add a request to MethodMetadataRequest `mmr`
        for pname, param in params:
            # Skip parameters named "X", "y", "Y", "Xt", "yt"
            if pname in {"X", "y", "Y", "Xt", "yt"}:
                continue
            # Skip *args and **kwargs
            if param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
                continue
            
            # Add a request to `mmr` for the parameter `pname` with alias as `None`
            mmr.add_request(
                param=pname,
                alias=None,
            )

        # Return the prepared MethodMetadataRequest `mmr`
        return mmr

    @classmethod
    def _get_default_requests(cls):
        """Collect default request values.

        This method combines the information present in ``__metadata_request__*``
        class attributes, as well as determining request keys from method
        signatures.
        """
        # 创建一个 MetadataRequest 对象，以类名作为所有者信息
        requests = MetadataRequest(owner=cls.__name__)

        # 遍历预定义的 SIMPLE_METHODS 列表
        for method in SIMPLE_METHODS:
            # 使用 _build_request_for_signature 方法为每个方法生成请求对象，并设置为 requests 的属性
            setattr(
                requests,
                method,
                cls._build_request_for_signature(router=requests, method=method),
            )

        # 然后使用类属性中的 __metadata_request__* 属性覆盖默认值。
        # 在属性中设置的默认值优先于方法签名推断。
        
        # 需要通过 MRO（方法解析顺序）来获取父类的属性，因为 vars 不会显示父类的属性。
        # 反向遍历 MRO，以确保子类的属性覆盖父类的属性。
        substr = "__metadata_request__"
        for base_class in reversed(inspect.getmro(cls)):
            for attr, value in vars(base_class).items():
                if substr not in attr:
                    continue
                # 获取属性名称后缀，用于确定请求的方法
                method = attr[attr.index(substr) + len(substr) :]
                for prop, alias in value.items():
                    # 将通过类属性指定的请求值添加到 MetadataRequest 对象中。
                    # 添加已经存在的请求会覆盖先前的请求。由于反向遍历 MRO，因此继承树最底层的类的属性生效。
                    getattr(requests, method).add_request(param=prop, alias=alias)

        # 返回收集到的所有请求信息
        return requests

    def _get_metadata_request(self):
        """Get requested data properties.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        request : MetadataRequest
            A :class:`~sklearn.utils.metadata_routing.MetadataRequest` instance.
        """
        # 如果对象具有 _metadata_request 属性，则获取该属性的路由信息
        if hasattr(self, "_metadata_request"):
            requests = get_routing_for_object(self._metadata_request)
        else:
            # 否则调用 _get_default_requests 方法获取默认请求信息
            requests = self._get_default_requests()

        # 返回请求信息对象
        return requests

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRequest
            A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
            routing information.
        """
        # 返回对象的元数据路由信息
        return self._get_metadata_request()
# Process Routing in Routers
# ==========================
# This is almost always the only method used in routers to process and route
# given metadata. This is to minimize the boilerplate required in routers.


# Here the first two arguments are positional only which makes everything
# passed as keyword argument a metadata. The first two args also have an `_`
# prefix to reduce the chances of name collisions with the passed metadata, and
# since they're positional only, users will never type those underscores.
def process_routing(_obj, _method, /, **kwargs):
    """Validate and route input parameters.

    This function is used inside a router's method, e.g. :term:`fit`,
    to validate the metadata and handle the routing.

    Assuming this signature of a router's fit method:
    ``fit(self, X, y, sample_weight=None, **fit_params)``,
    a call to this function would be:
    ``process_routing(self, "fit", sample_weight=sample_weight, **fit_params)``.

    Note that if routing is not enabled and ``kwargs`` is empty, then it
    returns an empty routing where ``process_routing(...).ANYTHING.ANY_METHOD``
    is always an empty dictionary.

    .. versionadded:: 1.3

    Parameters
    ----------
    _obj : object
        An object implementing ``get_metadata_routing``. Typically a
        meta-estimator.

    _method : str
        The name of the router's method in which this function is called.

    **kwargs : dict
        Metadata to be routed.

    Returns
    -------
    routed_params : Bunch
        A :class:`~utils.Bunch` of the form ``{"object_name": {"method_name":
        {params: value}}}`` which can be used to pass the required metadata to
        corresponding methods or corresponding child objects. The object names
        are those defined in `obj.get_metadata_routing()`.
    """
    if not kwargs:
        # If routing is not enabled and kwargs are empty, then we don't have to
        # try doing any routing, we can simply return a structure which returns
        # an empty dict on routed_params.ANYTHING.ANY_METHOD.
        class EmptyRequest:
            def get(self, name, default=None):
                # Returns an empty Bunch object for each method in METHODS
                return Bunch(**{method: dict() for method in METHODS})

            def __getitem__(self, name):
                # Returns an empty Bunch object for each method in METHODS
                return Bunch(**{method: dict() for method in METHODS})

            def __getattr__(self, name):
                # Returns an empty Bunch object for each method in METHODS
                return Bunch(**{method: dict() for method in METHODS})

        # Return an instance of EmptyRequest when no metadata is provided
        return EmptyRequest()

    # Check if _obj has the required method or is an instance of MetadataRouter
    if not (hasattr(_obj, "get_metadata_routing") or isinstance(_obj, MetadataRouter)):
        # Raise an AttributeError if _obj cannot handle metadata routing
        raise AttributeError(
            f"The given object ({repr(_obj.__class__.__name__)}) needs to either"
            " implement the routing method `get_metadata_routing` or be a"
            " `MetadataRouter` instance."
        )
    # 检查 _method 是否存在于预定义的 METHODS 列表中，如果不存在则引发 TypeError 异常
    if _method not in METHODS:
        raise TypeError(
            f"Can only route and process input on these methods: {METHODS}, "
            f"while the passed method is: {_method}."
        )
    
    # 获取针对 _obj 的路由信息
    request_routing = get_routing_for_object(_obj)
    
    # 使用 request_routing 对象验证参数的元数据，传入方法 _method
    request_routing.validate_metadata(params=kwargs, method=_method)
    
    # 使用 request_routing 对象将参数路由，传入调用者 _method，并返回路由后的参数
    routed_params = request_routing.route_params(params=kwargs, caller=_method)
    
    # 返回经过路由处理后的参数
    return routed_params
```