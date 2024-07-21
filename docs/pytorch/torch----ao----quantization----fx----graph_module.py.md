# `.\pytorch\torch\ao\quantization\fx\graph_module.py`

```py
# mypy: allow-untyped-defs
# 引入PyTorch库
import torch
# 复制库
import copy
# 从torch.fx中引入GraphModule
from torch.fx import GraphModule
# 从torch.fx.graph中引入Graph
from torch.fx.graph import Graph
# 引入类型提示Union, Dict, Any, Set
from typing import Union, Dict, Any, Set

# 定义公开的类名列表
__all__ = [
    "FusedGraphModule",
    "ObservedGraphModule",
    "ObservedStandaloneGraphModule",
    "QuantizedGraphModule",
]

# 继承GraphModule的FusedGraphModule类
class FusedGraphModule(GraphModule):
    # 初始化方法
    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, preserved_attr_names: Set[str]):
        # 保留的属性名称集合
        self.preserved_attr_names = preserved_attr_names
        # 根据root对象获取保留的属性字典
        preserved_attrs = {attr: getattr(root, attr) for attr in self.preserved_attr_names if hasattr(root, attr)}
        # 调用父类GraphModule的初始化方法
        super().__init__(root, graph)
        # 将保留的属性设置到当前对象中
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # 覆盖__deepcopy__方法以正确复制量化特定属性
    def __deepcopy__(self, memo):
        # 创建一个假的Module对象
        fake_mod = torch.nn.Module()
        # 深度复制当前对象的字典内容到假的Module对象中
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        # 返回一个新的FusedGraphModule对象，复制了假的Module对象、图对象和保留属性名称集合
        return FusedGraphModule(fake_mod, copy.deepcopy(self.graph), copy.deepcopy(self.preserved_attr_names))

# 继承GraphModule的ObservedGraphModule类
class ObservedGraphModule(GraphModule):

    # 初始化方法
    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, preserved_attr_names: Set[str]):
        # 定义扩展的保留属性名称集合
        self.preserved_attr_names = {
            '_activation_post_process_map',
            '_activation_post_process_indexes',
            '_patterns',
            '_node_name_to_qconfig',
            '_prepare_custom_config',
            '_equalization_node_name_to_qconfig',
            '_node_name_to_scope',
            '_qconfig_mapping',
            '_is_qat',
            '_observed_node_names'}.union(preserved_attr_names)
        # 根据root对象获取扩展保留的属性字典
        preserved_attrs = {attr: getattr(root, attr) for attr in self.preserved_attr_names if hasattr(root, attr)}
        # 调用父类GraphModule的初始化方法
        super().__init__(root, graph)
        # 将扩展保留的属性设置到当前对象中
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # 覆盖__deepcopy__方法以正确复制量化特定属性
    def __deepcopy__(self, memo):
        # 创建一个假的Module对象
        fake_mod = torch.nn.Module()
        # 深度复制当前对象的字典内容到假的Module对象中
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        # 返回一个新的ObservedGraphModule对象，复制了假的Module对象、图对象和扩展保留属性名称集合
        return ObservedGraphModule(fake_mod, copy.deepcopy(self.graph), copy.deepcopy(self.preserved_attr_names))

# 判断是否为观察模块的函数
def _is_observed_module(module: Any) -> bool:
    # 检查module是否具有"meta"属性并且" _observed_graph_module_attrs"在其meta属性中
    return hasattr(module, "meta") and "_observed_graph_module_attrs" in module.meta

# 获取观察图模块属性的函数
def _get_observed_graph_module_attr(model: Union[torch.nn.Module, GraphModule], attr_name: str) -> Any:
    # 如果model具有"meta"属性并且" _observed_graph_module_attrs"在其meta属性中
    if hasattr(model, "meta") and "_observed_graph_module_attrs" in model.meta:  # type: ignore[operator, index]
        # 返回model.meta["_observed_graph_module_attrs"]中attr_name对应的属性
        return getattr(model.meta["_observed_graph_module_attrs"], attr_name)  # type: ignore[index]
    # 否则返回None
    return None

# 继承ObservedGraphModule的ObservedStandaloneGraphModule类
class ObservedStandaloneGraphModule(ObservedGraphModule):
    # 初始化方法，接受参数包括根节点（torch.nn.Module 或字典）、图结构（Graph）、保留的属性名称集合
    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, preserved_attr_names: Set[str]):
        # 更新保留的属性名称集合，添加两个特定的保留属性名称
        preserved_attr_names = preserved_attr_names.union({
            "_standalone_module_input_quantized_idxs",
            "_standalone_module_output_quantized_idxs"})
        # 调用父类的初始化方法，传入根节点、图结构和更新后的保留属性名称集合
        super().__init__(root, graph, preserved_attr_names)

    # 深拷贝方法，用于创建当前对象的副本
    def __deepcopy__(self, memo):
        # 创建一个空的 torch.nn.Module 实例作为假的模块
        fake_mod = torch.nn.Module()
        # 深拷贝当前对象的字典属性到假的模块实例中
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        # 返回一个以假模块、深拷贝的图结构和深拷贝的保留属性名称集合创建的 ObservedStandaloneGraphModule 实例作为副本
        return ObservedStandaloneGraphModule(fake_mod, copy.deepcopy(self.graph), copy.deepcopy(self.preserved_attr_names))
`
def _is_observed_standalone_module(module: Any) -> bool:
    # 检查模块是否既是观察到的模块，又符合独立观察模块的条件
    return _is_observed_module(module) and module.meta["_observed_graph_module_attrs"].is_observed_standalone_module

def _save_packed_weight(self, destination, prefix, keep_vars):
    # 遍历当前对象的所有属性
    for attr_name in dir(self):
        # 检查属性名是否包含 "_packed_weight" 并且该属性是 torch._C.ScriptObject 类型
        if "_packed_weight" in attr_name and \
           isinstance(getattr(self, attr_name), torch._C.ScriptObject):  # type: ignore[attr-defined]
            # 获取属性值
            packed_weight = getattr(self, attr_name)
            # 将属性值保存到目标字典，键名为 prefix + 属性名
            destination[prefix + attr_name] = packed_weight

class QuantizedGraphModule(GraphModule):
    """ 
    此类的目的是确保 PackedParams（例如 LinearPackedParams, Conv2dPackedParams）出现在 state_dict 中，
    这样我们可以使用 torch.save(m.state_dict()) 和 m.load_state_dict(state_dict) 对量化图形模块进行序列化和反序列化
    """
    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, preserved_attr_names: Set[str]):
        # 初始化保留的属性名集合
        self.preserved_attr_names = preserved_attr_names
        # 根据保留的属性名，从 root 中获取属性
        preserved_attrs = {attr: getattr(root, attr) for attr in self.preserved_attr_names if hasattr(root, attr)}
        # 调用父类初始化方法
        super().__init__(root, graph)
        # 设置保留的属性到当前对象
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])
        # 注册 state_dict 钩子函数
        self._register_state_dict_hook(_save_packed_weight)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 初始化待删除属性列表
        attrs_to_pop = []
        # 遍历 state_dict 中的每个属性名
        for attr_name in state_dict:
            # 检查属性名是否以 "_packed_weight" 开头，并且属性值是 torch._C.ScriptObject 类型
            if attr_name.startswith("_packed_weight") and isinstance(state_dict[attr_name], torch._C.ScriptObject):  # type: ignore[attr-defined] # noqa: B950
                # 设置属性到当前对象
                setattr(self, attr_name, state_dict[attr_name])
                # 将属性名添加到待删除列表
                attrs_to_pop.append(attr_name)

        # 删除 packed param 属性
        for attr_name in attrs_to_pop:
            state_dict.pop(attr_name)

        # 调用父类的 _load_from_state_dict 方法进行加载
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def __deepcopy__(self, memo):
        # 创建一个新的 torch.nn.Module 对象
        fake_mod = torch.nn.Module()
        # 深拷贝当前对象的字典属性
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        # 返回一个新的 QuantizedGraphModule 对象，使用拷贝后的数据
        return QuantizedGraphModule(fake_mod, copy.deepcopy(self.graph), copy.deepcopy(self.preserved_attr_names))
```