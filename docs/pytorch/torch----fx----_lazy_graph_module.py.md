# `.\pytorch\torch\fx\_lazy_graph_module.py`

```
# 添加类型检查允许未声明的函数
mypy: allow-untyped-defs
# 导入上下文管理器
from contextlib import contextmanager

# 导入 torch 的 GraphModule 类和一些辅助函数
from torch.fx import GraphModule
from torch.fx.graph_module import (
    _format_import_block,
    reduce_graph_module,
    reduce_package_graph_module,
)
# 导入用于打包导出的 PackageExporter 和 sys_importer
from torch.package import PackageExporter, sys_importer
# 导入兼容性函数 compatibility
from ._compatibility import compatibility

# 是否启用 lazy graph module 的标志位，默认为 False
_use_lazy_graph_module_flag = False
# 强制跳过 lazy graph module 的标志位，默认为 False
_force_skip_lazy_graph_module_flag = False


# 定义一个装饰器，用于标记不向后兼容的函数，并作为上下文管理器
@compatibility(is_backward_compatible=False)
@contextmanager
def _force_skip_lazy_graph_module():
    """
    Skip using lazy graph module disregarding the setting of _use_lazy_graph_module.
    Use to skip _LazyGraphModule when testing inductor torchscript related backend.

    torch.jit.script a _LazyGraphModule results in following error:
        https://gist.github.com/shunting314/5143654c8084aed84ecd19b818258a69
    """
    try:
        # 获取全局的 _force_skip_lazy_graph_module_flag，并保存当前值
        global _force_skip_lazy_graph_module_flag
        prior = _force_skip_lazy_graph_module_flag
        # 设置 _force_skip_lazy_graph_module_flag 为 True
        _force_skip_lazy_graph_module_flag = True
        yield
    finally:
        # 恢复 _force_skip_lazy_graph_module_flag 的先前值
        _force_skip_lazy_graph_module_flag = prior


# 定义一个装饰器，用于设置是否使用 lazy graph module 的标志位
@compatibility(is_backward_compatible=False)
@contextmanager
def _use_lazy_graph_module(should_use: bool):
    try:
        # 获取全局的 _use_lazy_graph_module_flag，并保存当前值
        global _use_lazy_graph_module_flag
        prior = _use_lazy_graph_module_flag
        # 设置 _use_lazy_graph_module_flag，根据 should_use 和 _force_skip_lazy_graph_module_flag 的值来决定
        _use_lazy_graph_module_flag = (
            should_use and not _force_skip_lazy_graph_module_flag
        )
        yield
    finally:
        # 恢复 _use_lazy_graph_module_flag 的先前值
        _use_lazy_graph_module_flag = prior


# 返回要使用的 graph module 类，根据 _use_lazy_graph_module_flag 的值来决定
@compatibility(is_backward_compatible=False)
def _get_graph_module_cls():
    return _LazyGraphModule if _use_lazy_graph_module_flag else GraphModule


# 创建 graph module 实例的工厂函数，可以根据传入的参数来选择使用哪种 graph module 类
def _make_graph_module(*args, graph_module_cls=None, **kwargs):
    if graph_module_cls is None:
        # 如果 graph_module_cls 没有指定，则根据 _get_graph_module_cls 函数选择合适的类
        graph_module_cls = _get_graph_module_cls()

    return graph_module_cls(*args, **kwargs)


# 一个 _LazyGraphModule 类的子类，用于延迟编译图模块
@compatibility(is_backward_compatible=False)
class _LazyGraphModule(GraphModule):
    """
    The main difference between _LazyGraphModule and GraphModule is how recompile happens.
    GraphModule will do a 'recompile' call to generate python code and the forward method when it's
    constructed. Later on if the graph get updated, recompile method can be called again to refresh
    the saved python code and forward method.

    However in some cases especially in inductor, the recompilation can be a waste since we never
    check the python code for the graph module or call its forward method. A few more concreate
    examples regarding pattern matching fx passes in inductor:
    1. some passes will update the graph to be compiled and then call recompile on the GraphModule.
    2. some passes will trace small pattern function to search it in the graph being compiled and
       replace the match with the traced graph of a replacement function. The pattern graph and
       replacement graph are quite small but there are large amount of them. Doing GraphModule.recompile
       for them in GraphModule.__init__ is also a waste of time.
    """
    """
    However simply skip calling GraphModule.recompile in these scenarios is also dangeruous.
    People may want to check the python code or call the GraphModule's forward method for debugging purposes.

    The way _LazyGraphModule solves it is, we override the recompile method to just mark the
    need for recompilation but does not do the actual recompilation. Later on if people really
    access the compiled python code or call the GraphModule's forward method, we do the real
    recompilation.
    """



    @classmethod
    def from_graphmodule(cls, gm: GraphModule):
        if isinstance(gm, _LazyGraphModule):
            return gm
        else:
            return _LazyGraphModule(gm, gm.graph)



    @staticmethod
    def force_recompile(gm):
        """
        Sometimes we need force a recompile as a workaround
        - we want to do the real recompilation before symbolic_trace to avoid error:
            https://gist.github.com/shunting314/75549c2e82ae07ac1139c94a3583d259
        """
        if isinstance(gm, _LazyGraphModule):
            gm.real_recompile()



    def real_recompile(self):
        if self._needs_recompile():
            self._real_recompile()



    @classmethod
    def _needs_recompile(cls):
        return cls.forward is cls._lazy_forward



    def _lazy_forward(self, *args, **kwargs):
        # Call self.real_recompile() rather than self._real_recompile() here.
        # The _lazy_forward method may be saved and call repeatedly.
        # Calling self.real_recompile can make sure we skip recompilation if
        # we have already done so.
        self.real_recompile()
        assert not self._needs_recompile()

        # call `__call__` rather than 'forward' since recompilation may
        # install a wrapper for `__call__` to provide a customized error
        # message.
        return self(*args, **kwargs)



    forward = _lazy_forward



    # TODO: we shold handle __reduce_deploy__ the same way as __reduce_package__,
    # or __reduce__ by calling _real_recompile. But I don't find a good way
    # to test __reduce_deploy__ out. Also it's very unlikely that LazyGraphModule
    # will be used in torch::deploy. So it's skipped for now.
    def __reduce_package__(self, exporter: PackageExporter):
        """
        Follow GraphModule.__reduce__ but call 'self._real_recompile' rather
        than 'self.recompile' since for a _LazyGraphModule, self.recompile just
        mark the need of recompilation and does not return the PythonCode object.
        """
        # 调用实际的重新编译方法，生成 PythonCode 对象
        python_code = self._real_recompile()
        # 复制当前对象的字典，排除 "_graph" 属性
        dict_without_graph = self.__dict__.copy()
        dict_without_graph["_graphmodule_cls_name"] = self.__class__.__name__
        del dict_without_graph["_graph"]

        # 生成一个唯一的模块名
        generated_module_name = f"fx-generated._{exporter.get_unique_id()}"
        # 格式化导入代码块，基于 python_code 的全局变量和 exporter 的导入器
        import_block = _format_import_block(python_code.globals, exporter.importer)
        # 拼接导入代码块和 self.code，生成模块代码
        module_code = import_block + self.code
        # 将生成的模块代码保存到 exporter 中，使用生成的模块名
        exporter.save_source_string(generated_module_name, module_code)
        # 返回一个元组，表示如何重构此对象
        return (
            reduce_package_graph_module,
            (dict_without_graph, generated_module_name),
        )

    def __reduce__(self):
        """
        Follow GraphModule.__reduce__ but call 'self._real_recompile' rather
        than 'self.recompile' since for a _LazyGraphModule, self.recompile just
        mark the need of recompilation and does not return the PythonCode object.
        """
        # 调用实际的重新编译方法，生成 PythonCode 对象
        python_code = self._real_recompile()
        # 复制当前对象的字典，排除 "_graph" 属性
        dict_without_graph = self.__dict__.copy()
        # 格式化导入代码块，基于 python_code 的全局变量和系统导入器
        import_block = _format_import_block(python_code.globals, sys_importer)
        del dict_without_graph["_graph"]
        # 返回一个元组，表示如何重构此对象
        return (reduce_graph_module, (dict_without_graph, import_block))

    def _real_recompile(self):
        # 调用超类方法执行实际的重新编译
        return super().recompile()

    @classmethod
    def recompile(cls):
        # 将类方法 recompile 重新定义为使用 _lazy_forward 方法
        cls.forward = cls._lazy_forward

    @property
    def code(self) -> str:
        # 调用实例方法 real_recompile，确保 _code 属性可用，然后返回其超类的 code 属性
        self.real_recompile()
        return super().code

    def __str__(self) -> str:
        """
        str(GraphModule) will access the _code attribute. Make sure recompile
        happens so _code attribute is available.
        """
        # 调用实例方法 real_recompile，确保 _code 属性可用，然后返回其超类的字符串表示
        self.real_recompile()
        return super().__str__()
```