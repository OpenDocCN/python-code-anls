# `.\pytorch\torch\fx\experimental\meta_tracer.py`

```
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 导入 torch.fx 库
import torch.fx
# 导入警告模块
import warnings
# 导入 functools 模块
import functools
# 导入内建函数模块
import builtins
# 导入类型注解相关库
from typing import Any, Callable, Dict, Optional, Union

# 定义函数 embedding_override，用于替换 torch.nn.Embedding 的行为
def embedding_override(self, input):
    return torch.empty(*input.shape, self.weight.shape[-1], device='meta')

# 定义函数 nn_layernorm_override，用于替换 torch.nn.LayerNorm 的行为
def nn_layernorm_override(self, input):
    return input

# 定义函数 torch_relu_override，用于替换 torch.relu 的行为
def torch_relu_override(x):
    return x

# 定义函数 torch_nn_relu_override，用于替换 torch.nn.ReLU 的行为
def torch_nn_relu_override(self, x):
    return x

# 定义函数 functional_relu_override，用于替换 torch.nn.functional.relu 的行为
def functional_relu_override(x, inplace=False):
    assert not inplace, 'dont support inplace functional.relu for metatensor analysis'
    return x

# 定义函数 torch_where_override，用于替换 torch.where 的行为
def torch_where_override(condition, x, y):
    # torch.where 返回广播后的张量，这里使用加法进行 hack
    return condition.to(device='meta') + x.to(device='meta') + y.to(device='meta')

# 定义函数 torch_abs_override，用于替换 torch.abs 的行为
def torch_abs_override(input, *, out=None):
    assert out is None, 'Dont support in-place abs for MetaTensor analysis'
    return input

# 创建字典 manual_meta_overrides，将需要替换的函数映射到其对应的替换函数上
manual_meta_overrides: Dict[Callable, Callable] = {
    torch.nn.Embedding: embedding_override,
    torch.nn.LayerNorm: nn_layernorm_override,
    torch.relu: torch_relu_override,
    torch.nn.functional.relu: functional_relu_override,
    torch.nn.ReLU: torch_nn_relu_override,
    torch.where: torch_where_override,
    torch.abs: torch_abs_override,
}

# 定义函数 gen_constructor_wrapper，用于生成构造函数的包装器
def gen_constructor_wrapper(target):
    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = None

        # 内部函数 check_has_proxy，用于检查参数是否为 torch.fx.Proxy 类型
        def check_has_proxy(v):
            if isinstance(v, torch.fx.Proxy):
                nonlocal proxy
                proxy = v
        # 对 args 中的每个元素调用 check_has_proxy 函数
        torch.fx.node.map_aggregate(args, check_has_proxy)
        # 对 kwargs 中的每个元素调用 check_has_proxy 函数
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        # 如果 proxy 不为 None，则返回使用代理对象创建的代理
        if proxy is not None:
            return proxy.tracer.create_proxy('call_function', target, args, kwargs)
        else:
            # 否则直接调用目标函数 target
            return target(*args, **kwargs)
    return wrapper, target

# 定义类 MetaProxy，继承自 torch.fx.Proxy
class MetaProxy(torch.fx.Proxy):
    # 方法 install_tensor_meta，用于安装张量元信息
    def install_tensor_meta(self, tensor_meta):
        self._tensor_meta = tensor_meta

    # 方法 size，用于返回张量的尺寸大小
    def size(self, dim=None):
        if hasattr(self, '_tensor_meta') and self._tensor_meta is not None:
            return self._tensor_meta.size(*[dim] if dim else [])
        return self.tracer.create_proxy('call_method', 'size', (self, dim) if dim else (self,), {})

    # 方法 dim，用于返回张量的维度数
    def dim(self):
        if hasattr(self, '_tensor_meta') and self._tensor_meta is not None:
            return self._tensor_meta.dim()
        return self.tracer.create_proxy('call_method', 'dim', (self,), {})

    # 属性 shape，用于返回张量的形状
    @property
    def shape(self):
        if hasattr(self, '_tensor_meta') and self._tensor_meta is not None:
            return self._tensor_meta.shape
        return self.tracer.create_proxy('call_function', builtins.getattr, (self, 'shape'), {})

    # 属性 dtype，用于返回张量的数据类型
    @property
    def dtype(self):
        if hasattr(self, '_tensor_meta') and self._tensor_meta is not None:
            return self._tensor_meta.dtype
        return self.tracer.create_proxy('call_function', builtins.getattr, (self, 'dtype'), {})
    # 将函数装饰为属性访问器，用于获取设备信息
    @property
    def device(self):
        # 通过 MetaDeviceAttribute 封装设备属性，用于元张量传播期间替换为常量 'meta'
        return MetaDeviceAttribute(self, 'device')

    # 自定义类中的特殊方法，用于动态获取属性
    def __getattr__(self, k):
        # 如果请求的属性为 '_tensor_meta'，直接返回该属性
        if k == '_tensor_meta':
            return self.__getattribute__(k)
        # 如果属性尚未添加到图中，且请求为方法调用，则优化为方法调用的 MetaAttribute
        # 对象返回
        return MetaAttribute(self, k)
class MetaAttribute(MetaProxy):
    # MetaAttribute 类继承自 MetaProxy

    def __init__(self, root, attr: str):
        # 初始化方法，接受 root 和 attr 参数

        self.root = root
        # 将 root 参数赋值给实例变量 self.root

        self.attr = attr
        # 将 attr 参数赋值给实例变量 self.attr

        self.tracer = root.tracer
        # 将 root 对象的 tracer 属性赋值给实例变量 self.tracer

        self._node = None
        # 初始化实例变量 self._node 为 None，用于延迟加载

    @property
    def node(self):
        # 属性访问器方法，用于获取节点对象

        # 如果节点对象尚未初始化
        if self._node is None:
            # 使用 tracer 创建一个代理节点，类型为 'call_function'，调用 getattr 方法
            # 参数为 (self.root, self.attr)，额外的配置为空字典 {}
            self._node = self.tracer.create_proxy('call_function', getattr, (self.root, self.attr), {}).node

        # 返回已初始化或新创建的节点对象
        return self._node

    def __call__(self, *args, **kwargs):
        # 实现对象的可调用行为

        # 使用 tracer 创建一个代理，类型为 'call_method'，方法名为 self.attr
        # 参数包括 self.root 和可变参数 args，关键字参数 kwargs
        return self.tracer.create_proxy('call_method', self.attr, (self.root,) + args, kwargs)


class MetaDeviceAttribute(MetaAttribute):
    # MetaDeviceAttribute 类继承自 MetaAttribute
    pass


def proxys_to_metas(v):
    # 定义函数 proxys_to_metas，接受参数 v

    if isinstance(v, MetaDeviceAttribute):
        # 如果 v 是 MetaDeviceAttribute 类型的实例
        return 'meta'
        # 返回字符串 'meta'

    if isinstance(v, torch.fx.Proxy):
        # 如果 v 是 torch.fx.Proxy 类型的实例
        assert isinstance(v, MetaProxy), f'Expected MetaProxy but got {type(v)}'
        # 断言 v 是 MetaProxy 的实例，如果不是则抛出异常
        assert hasattr(v, '_tensor_meta'), 'MetaProxy does not have an associated meta'
        # 断言 v 具有 _tensor_meta 属性，如果没有则抛出异常
        return v._tensor_meta
        # 返回 v 的 _tensor_meta 属性

    return v
    # 其他情况直接返回 v


class MetaTracer(torch.fx.Tracer):
    # MetaTracer 类继承自 torch.fx.Tracer

    allow_insert_stateless_mods : bool = True
    # 设置类属性 allow_insert_stateless_mods 为 True

    _TORCH_METHODS_TO_PATCH = ['arange', 'zeros', 'ones', 'full_like', 'eye']
    # 设置类属性 _TORCH_METHODS_TO_PATCH，包含需要修补的 torch 方法名称列表
    # 创建代理对象的方法，继承自父类的方法，并对代理对象进行特定处理
    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        # 调用父类方法创建代理对象
        rv = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

        # 如果代理对象的类型是 'placeholder' 并且目标在元数据参数中
        if kind == 'placeholder' and target in self.meta_args:
            # 为代理对象安装张量元数据
            rv.install_tensor_meta(self.meta_args[target])
            return rv

        # 如果目标在原始函数中
        if target in self.orig_fns:
            # 如果 kwargs 中包含 'device' 参数
            if 'device' in kwargs:
                # 将 'device' 参数设置为 'meta'
                kwargs['device'] = 'meta'

        try:
            # 将 args 中的代理对象映射为元数据
            args_metas = torch.fx.node.map_aggregate(args, proxys_to_metas)
            # 将 kwargs 中的代理对象映射为元数据
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, proxys_to_metas)

            # 根据 kind 类型进行不同的处理
            if kind == 'call_function':
                # 获取手动设置的元数据目标，如果没有则使用 target 自身
                meta_target = manual_meta_overrides.get(target, target)
                # 使用元数据参数调用目标函数
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == 'call_method':
                # 调用 args 的第一个元素的目标方法，并传入元数据参数
                meta_out = getattr(args_metas[0], target)(*args_metas[1:], **kwargs_metas)
            elif kind == 'call_module':
                # 确保存在 self.orig_forward 方法
                assert hasattr(self, 'orig_forward')
                # 禁用模块的 getattr 方法
                self._disable_module_getattr = True
                try:
                    # 获取根模块中的子模块
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    # 如果模块类型在手动元数据覆盖中
                    if mod_type in manual_meta_overrides:
                        # 使用手动设置的元数据函数计算元数据
                        meta_out = manual_meta_overrides[mod_type](mod, *args_metas, **kwargs_metas)
                    else:
                        # 使用原始 forward 方法计算元数据
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                finally:
                    # 恢复模块的 getattr 方法
                    self._disable_module_getattr = False
            elif kind == 'get_attr':
                # 禁用模块的 getattr 方法
                self._disable_module_getattr = True
                try:
                    # 从根模块开始获取属性迭代器
                    attr_itr = self.root
                    atoms = target.split('.')
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    # 确保属性迭代器是 torch.Tensor 类型
                    assert isinstance(attr_itr, torch.Tensor)
                    # 将属性迭代器转换为 'meta' 设备上的张量
                    meta_out = attr_itr.to(device='meta')
                finally:
                    # 恢复模块的 getattr 方法
                    self._disable_module_getattr = False
            else:
                return rv

            # 断言 rv 是 torch.fx.Proxy 类型的对象，用于安装张量元数据
            assert isinstance(rv, torch.fx.Proxy), 'Dont support composite output yet'
            rv.install_tensor_meta(meta_out)
        except Exception as e:
            # 异常处理：警告无法计算目标 {kind} 的元数据
            warnings.warn(f'Could not compute metadata for {kind} target {target}: {e}')

        # 返回创建的代理对象 rv
        return rv

    # 获取属性的方法，如果禁用模块的 getattr 方法，则直接返回属性值，否则调用父类的方法处理
    def getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, '_disable_module_getattr', False):
            return attr_val
        else:
            return super().getattr(attr, attr_val, parameter_proxy_cache)
    # 设置当前对象的原始前向函数为指定的前向函数，并返回超类的相应调用结果
    def call_module(self, m, forward, args, kwargs):
        self.orig_forward = forward
        return super().call_module(m, forward, args, kwargs)

    # 尝试将未声明为子模块的模块插入到根模块中的辅助方法
    def _insert_module_as_submodule(self, mod: torch.nn.Module) -> str:
        idx = 0
        # 获取模块类名的小写形式作为基础路径名
        mod_name = mod.__class__.__name__.lower()
        path = f"{mod_name}_{idx}"
        # 如果根模块已经存在该路径名，则自增索引直到找到可用路径名
        while hasattr(self.root, path):
            path = f"{mod_name}_{idx}"
            idx += 1

        # 将模块添加为子模块到根模块中，并返回路径名
        self.root.add_module(path, mod)
        return path

    # 获取给定模块在树中的路径
    def path_of_module(self, mod: torch.nn.Module) -> str:
        try:
            # 调用超类方法获取模块路径
            return super().path_of_module(mod)
        except NameError as e:
            # 如果允许插入无状态模块且模块无参数和缓冲区，则插入并返回路径
            if self.allow_insert_stateless_mods and len(list(mod.parameters())) == 0 and len(list(mod.buffers())) == 0:
                path = self._insert_module_as_submodule(mod)
                self.prev_module = path
                return path
            # 否则，抛出异常
            raise

    # 返回给定节点的元数据代理
    def proxy(self, node):
        return MetaProxy(node, self)

    # 对给定的根模块进行追踪，使用元数据参数进行配置，可以选择具体参数
    def trace(self, root, meta_args: Dict[str, torch.Tensor], concrete_args=None):
        assert isinstance(meta_args, dict)
        # 存储元数据参数
        self.meta_args = meta_args

        # 生成torch方法的构造函数包装器，并存储原始函数
        self.patched_torch_methods = {
            target: gen_constructor_wrapper(getattr(torch, target)) for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        # 将包装后的torch方法设置为其对应的包装器
        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        try:
            # 调用超类的trace方法追踪计算图
            graph = super().trace(root, concrete_args)
            # 添加额外的跟踪器信息到计算图中
            graph._tracer_extras = {'meta_args': meta_args}
            return graph
        finally:
            # 在finally中恢复每个torch方法的原始函数
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)
# 定义一个函数，用于生成符号跟踪的图模块
def symbolic_trace(root: Union[torch.nn.Module, Callable[..., Any]],
                   meta_args: Optional[Dict[str, torch.Tensor]] = None,
                   concrete_args: Optional[Dict[str, Any]] = None) -> torch.fx.GraphModule:
    # 创建一个元数据跟踪器对象
    tracer = MetaTracer()
    # 使用跟踪器对象对指定的根对象进行符号跟踪，生成一个计算图
    graph = tracer.trace(root, meta_args, concrete_args)
    # 根据根对象的类型名称，创建一个图模块对象
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    gm = torch.fx.GraphModule(tracer.root, graph, name)
    # 返回生成的图模块对象
    return gm
```