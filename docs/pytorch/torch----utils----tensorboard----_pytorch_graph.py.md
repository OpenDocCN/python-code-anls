# `.\pytorch\torch\utils\tensorboard\_pytorch_graph.py`

```py
# 使用 'mypy: allow-untyped-defs' 指令允许未类型化的定义
from collections import OrderedDict  # 导入有序字典模块
import contextlib  # 导入上下文管理模块
from typing import Dict, Any  # 导入类型提示模块，指定 Dict 和 Any 类型

from tensorboard.compat.proto.config_pb2 import RunMetadata  # 导入 RunMetadata 协议缓冲区定义
from tensorboard.compat.proto.graph_pb2 import GraphDef  # 导入 GraphDef 协议缓冲区定义
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats  # 导入 StepStats 和 DeviceStepStats 协议缓冲区定义
from tensorboard.compat.proto.versions_pb2 import VersionDef  # 导入 VersionDef 协议缓冲区定义

import torch  # 导入 PyTorch 库
from ._proto_graph import node_proto  # 从 _proto_graph 模块中导入 node_proto 对象

methods_OP = [  # 定义操作节点相关的方法列表
    "attributeNames",  # 属性名
    "hasMultipleOutputs",  # 是否有多个输出
    "hasUses",  # 是否有用途
    "inputs",  # 输入
    "kind",  # 类型
    "outputs",  # 输出
    "outputsSize",  # 输出大小
    "scopeName",  # 作用域名称
]

# Some additional methods to explure for methods_IO are
#
#   'unique' (type int)
#   'type' (type <Tensor<class 'torch._C.Type'>>)
#
# But the below are sufficient for now.

# 定义输入输出节点相关的方法列表
methods_IO = ["node", "offset", "debugName"]

GETATTR_KIND = "prim::GetAttr"  # 获取属性的类型常量
CLASSTYPE_KIND = "ClassType"  # 类型类别的常量


class NodeBase:
    def __init__(
        self,
        debugName=None,
        inputs=None,
        scope=None,
        tensor_size=None,
        op_type="UnSpecified",
        attributes="",
    ):
        # 初始化节点基类，设置各属性
        self.debugName = debugName  # 调试名称
        self.inputs = inputs  # 输入
        self.tensor_size = tensor_size  # 张量大小
        self.kind = op_type  # 操作类型
        self.attributes = attributes  # 属性
        self.scope = scope  # 作用域

    def __repr__(self):
        repr = []  # 初始化 repr 列表
        repr.append(str(type(self)))  # 添加当前对象类型的字符串表示
        for m in dir(self):
            if "__" not in m:  # 排除私有属性
                repr.append(
                    m + ": " + str(getattr(self, m)) + str(type(getattr(self, m)))
                )  # 将属性名、属性值及其类型添加到 repr 列表中
        return "\n".join(repr) + "\n\n"  # 返回所有属性的字符串表示形式


class NodePy(NodeBase):
    def __init__(self, node_cpp, valid_methods):
        super().__init__(node_cpp)  # 调用父类的初始化方法，传入 node_cpp 对象
        valid_methods = valid_methods[:]  # 复制有效方法列表
        self.inputs = []  # 初始化输入列表为空

        for m in valid_methods:
            if m == "inputs" or m == "outputs":  # 如果是输入或输出方法
                list_of_node = list(getattr(node_cpp, m)())  # 获取节点对象列表
                io_unique_names = []  # 初始化唯一名称列表
                io_tensor_sizes = []  # 初始化张量大小列表
                for n in list_of_node:
                    io_unique_names.append(n.debugName())  # 将节点的调试名称添加到唯一名称列表
                    if n.isCompleteTensor():
                        io_tensor_sizes.append(n.type().sizes())  # 如果节点是完整张量，则获取其大小
                    else:
                        io_tensor_sizes.append(None)  # 否则，添加 None 到张量大小列表

                setattr(self, m, io_unique_names)  # 设置对象的输入/输出属性为唯一名称列表
                setattr(self, m + "tensor_size", io_tensor_sizes)  # 设置对象的输入/输出张量大小属性为张量大小列表

            else:
                setattr(self, m, getattr(node_cpp, m)())  # 设置对象的其他属性为节点对象的对应属性值


class NodePyIO(NodePy):
    pass  # NodePyIO 类继承自 NodePy 类，无需添加额外属性或方法
    # 初始化方法，用于创建一个新的对象实例
    def __init__(self, node_cpp, input_or_output=None):
        # 调用父类的初始化方法，传入参数 node_cpp 和 methods_IO
        super().__init__(node_cpp, methods_IO)
        
        # 尝试获取节点的张量大小信息
        try:
            tensor_size = node_cpp.type().sizes()
        except RuntimeError:
            # 如果运行时出现异常，设置默认的张量大小为 [1]
            tensor_size = [
                1,
            ]  # 在使用常量模型时可能会失败。

        # 将获取到的张量大小保存在对象实例的 tensor_size 属性中
        self.tensor_size = tensor_size

        # 设置对象实例的 kind 属性为 "Parameter"，这是一个描述性字符串，
        # 将在 TensorBoard 的图插件中显示在节点的详细信息中。
        # NodePyOP 节点可以从其 kind() 方法获取这个值。
        self.kind = "Parameter"

        # 如果输入了 input_or_output 参数
        if input_or_output:
            # 将 input_or_output 参数保存在对象实例的 input_or_output 属性中
            self.input_or_output = input_or_output
            # 将对象实例的 kind 属性设置为 "IO Node"
            self.kind = "IO Node"
class NodePyOP(NodePy):
    def __init__(self, node_cpp):
        # 调用父类构造函数，初始化 NodePyOP 对象
        super().__init__(node_cpp, methods_OP)
        # 替换单引号，因为在 TensorBoard 中会导致奇怪的行为
        # TODO: 看看以后是否可以移除这个替换
        # 将节点的属性转换为字符串，并替换掉单引号为空格
        self.attributes = str(
            {k: _node_get(node_cpp, k) for k in node_cpp.attributeNames()}
        ).replace("'", " ")
        # 获取节点的类型
        self.kind = node_cpp.kind()


class GraphPy:
    """Helper class to convert torch.nn.Module to GraphDef proto and visualization with TensorBoard.

    GraphDef generation operates in two passes:

    In the first pass, all nodes are read and saved to two lists.
    One list is for input/output nodes (nodes_io), which only have inbound
    or outbound connections, but not both. Another list is for internal
    operator nodes (nodes_op). The first pass also saves all scope name
    appeared in the nodes in scope_name_appeared list for later processing.

    In the second pass, scope names are fully applied to all nodes.
    debugNameToScopedName is a mapping from a node's ID to its fully qualified
    scope name. e.g. Net1/Linear[0]/1. Unfortunately torch.jit doesn't have
    totally correct scope output, so this is nontrivial. The function
    populate_namespace_from_OP_to_IO and find_common_root are used to
    assign scope name to a node based on the connection between nodes
    in a heuristic kind of way. Bookkeeping is done with shallowest_scope_name
    and scope_name_appeared.
    """

    def __init__(self):
        # 初始化 GraphPy 对象，设置初始节点列表、IO 节点字典、唯一名称到完全作用域名称的映射、最浅的作用域名称和出现过的作用域名称列表
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = "default"
        self.scope_name_appeared = []

    def append(self, x):
        # 将节点 x 添加到合适的节点列表中
        if isinstance(x, NodePyIO):
            self.nodes_io[x.debugName] = x
        if isinstance(x, NodePyOP):
            self.nodes_op.append(x)

    def printall(self):
        # 打印所有节点信息
        print("all nodes")
        for node in self.nodes_op:
            print(node)
        for key in self.nodes_io:
            print(self.nodes_io[key])

    def find_common_root(self):
        # 找到所有出现过的作用域名称中最浅的那个作用域
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split("/")[0]
    def populate_namespace_from_OP_to_IO(self):
        # 遍历 self.nodes_op 中的每个节点
        for node in self.nodes_op:
            # 遍历当前节点的输出及其大小
            for node_output, outputSize in zip(node.outputs, node.outputstensor_size):
                # 将当前节点的作用域名添加到 scope_name_appeared 列表中
                self.scope_name_appeared.append(node.scopeName)
                # 将节点输出及其相关信息存储到 nodes_io 字典中
                self.nodes_io[node_output] = NodeBase(
                    node_output,
                    node.inputs,
                    node.scopeName,
                    outputSize,
                    op_type=node.kind,
                    attributes=node.attributes,
                )

        # 调用 find_common_root 方法寻找公共根节点
        self.find_common_root()

        # 再次遍历 self.nodes_op 中的每个节点
        for node in self.nodes_op:
            # 遍历当前节点的输入节点列表
            for input_node_id in node.inputs:
                # 将输入节点的唯一名称映射为完整作用域路径
                self.unique_name_to_scoped_name[input_node_id] = (
                    node.scopeName + "/" + input_node_id
                )

        # 遍历 nodes_io 字典中的每个节点
        for key, node in self.nodes_io.items():
            # 如果节点类型为 NodeBase 类型
            if type(node) == NodeBase:
                # 将节点的唯一名称映射为完整作用域路径和调试名称
                self.unique_name_to_scoped_name[key] = node.scope + "/" + node.debugName
            # 如果节点具有属性 "input_or_output"
            if hasattr(node, "input_or_output"):
                # 将节点的唯一名称映射为输入/输出类型和调试名称
                self.unique_name_to_scoped_name[key] = (
                    node.input_or_output + "/" + node.debugName
                )

            # 如果节点具有作用域属性且不为 None
            if hasattr(node, "scope") and node.scope is not None:
                # 将节点的唯一名称映射为作用域路径和调试名称
                self.unique_name_to_scoped_name[key] = node.scope + "/" + node.debugName
                # 如果节点的作用域为空字符串且存在浅层作用域名称
                if node.scope == "" and self.shallowest_scope_name:
                    # 将节点调试名称映射为浅层作用域路径和调试名称
                    self.unique_name_to_scoped_name[node.debugName] = (
                        self.shallowest_scope_name + "/" + node.debugName
                    )

        # 替换名称
        # 再次遍历 nodes_io 字典中的每个节点
        for key, node in self.nodes_io.items():
            # 更新节点的输入列表，使用唯一名称映射
            self.nodes_io[key].inputs = [
                self.unique_name_to_scoped_name[node_input_id]
                for node_input_id in node.inputs
            ]
            # 如果节点的调试名称存在于唯一名称映射中
            if node.debugName in self.unique_name_to_scoped_name:
                # 更新节点的调试名称为映射后的名称
                self.nodes_io[key].debugName = self.unique_name_to_scoped_name[
                    node.debugName
                ]

    def to_proto(self):
        """Convert graph representation of GraphPy object to TensorBoard required format."""
        # TODO: compute correct memory usage and CPU time once
        # PyTorch supports it
        # 构建节点列表，将 nodes_io 中的节点信息转换为 protobuf 格式
        nodes = []
        for v in self.nodes_io.values():
            nodes.append(
                node_proto(
                    v.debugName,
                    input=v.inputs,
                    outputsize=v.tensor_size,
                    op=v.kind,
                    attributes=v.attributes,
                )
            )
        return nodes
# 解析优化过的 PyTorch 模型图，并生成节点和节点统计信息的列表
def parse(graph, trace, args=None, omit_useless_nodes=True):
    """Parse an optimized PyTorch model graph and produces a list of nodes and node stats.

    Useful for eventual conversion to TensorBoard protobuf format.

    Args:
      graph (PyTorch module): The model graph to be parsed.
      trace (PyTorch JIT TracedModule): The model trace to be parsed.
      args (tuple): input tensor[s] for the model.
      omit_useless_nodes (boolean): Whether to remove nodes from the graph.
    """
    n_inputs = len(args)  # 计算输入张量的数量

    scope = {}  # 初始化作用域字典，用于存储节点作用域信息
    nodes_py = GraphPy()  # 创建 GraphPy 对象，用于存储节点的 Python 表示

    # 遍历模型图的输入节点
    for node in graph.inputs():
        if omit_useless_nodes:
            # 如果要移除无用节点并且节点没有使用者（输出数为零），则跳过该节点
            if len(node.uses()) == 0:
                continue

        # 如果节点类型不是 CLASSTYPE_KIND，则将其添加到 nodes_py 中作为输入节点
        if node.type().kind() != CLASSTYPE_KIND:
            nodes_py.append(NodePyIO(node, "input"))

    attr_to_scope: Dict[Any, str] = {}  # 用于存储属性到作用域的映射字典

    # 遍历模型图的所有节点
    for node in graph.nodes():
        if node.kind() == GETATTR_KIND:
            # 如果节点类型是 GETATTR_KIND，则处理属性名称和作用域信息
            attr_name = node.s("name")
            attr_key = node.output().debugName()
            parent = node.input().node()

            if parent.kind() == GETATTR_KIND:
                # 如果父节点不是顶层的 "self" 节点，则处理父节点的作用域信息
                parent_attr_name = parent.s("name")
                parent_attr_key = parent.output().debugName()
                parent_scope = attr_to_scope[parent_attr_key]
                attr_scope = parent_scope.split("/")[-1]
                attr_to_scope[attr_key] = f"{parent_scope}/{attr_scope}.{attr_name}"
            else:
                attr_to_scope[attr_key] = f"__module.{attr_name}"

            # 如果节点的输出类型不是 CLASSTYPE_KIND，则将节点添加到 nodes_py 中作为操作节点
            if node.output().type().kind() != CLASSTYPE_KIND:
                node_py = NodePyOP(node)
                node_py.scopeName = attr_to_scope[attr_key]  # 设置节点的作用域名称
                nodes_py.append(node_py)
        else:
            # 如果节点类型不是 GETATTR_KIND，则将节点添加到 nodes_py 中作为操作节点
            nodes_py.append(NodePyOP(node))

    # 遍历模型图的输出节点，为输出操作创建 sink 节点
    for i, node in enumerate(graph.outputs()):
        node_pyio = NodePyIO(node, "output")
        node_pyio.debugName = f"output.{i + 1}"
        node_pyio.inputs = [node.debugName()]
        nodes_py.append(node_pyio)

    # 定义一个函数，用于解析追踪模块的名称
    def parse_traced_name(module):
        if isinstance(module, torch.jit.TracedModule):
            module_name = module._name
        else:
            module_name = getattr(module, "original_name", "Module")
        return module_name

    alias_to_name = {}  # 创建别名到名称的映射字典
    base_name = parse_traced_name(trace)  # 解析追踪模块的基础名称

    # 遍历追踪模块的命名模块，生成模块别名到名称的映射
    for name, module in trace.named_modules(prefix="__module"):
        mod_name = parse_traced_name(module)
        attr_name = name.split(".")[-1]
        alias_to_name[name] = f"{mod_name}[{attr_name}]"
    # 遍历 nodes_op 列表中的每个节点
    for node in nodes_py.nodes_op:
        # 将节点的作用域名称按斜杠分割成模块别名列表
        module_aliases = node.scopeName.split("/")
        # 对每个模块别名进行替换，如果别名在 alias_to_name 中存在，则替换为对应的名称，否则使用别名中最后一个点号后的部分
        replacements = [
            alias_to_name[alias] if alias in alias_to_name else alias.split(".")[-1]
            for alias in module_aliases
        ]
        # 将节点的作用域名称设置为基础名称 base_name
        node.scopeName = base_name
        # 如果存在任何替换项，则将替换项按斜杠连接到节点的作用域名称后面
        if any(replacements):
            node.scopeName += "/" + "/".join(replacements)
    
    # 从 OP 到 IO 中填充 nodes_py 的命名空间
    nodes_py.populate_namespace_from_OP_to_IO()
    # 将 nodes_py 对象转换为 Protocol Buffer 格式并返回
    return nodes_py.to_proto()
# 处理 PyTorch 模型并生成可记录到 TensorBoard 的 `GraphDef` 协议数据
def graph(model, args, verbose=False, use_strict_trace=True):
    """
    Process a PyTorch model and produces a `GraphDef` proto that can be logged to TensorBoard.

    Args:
      model (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      verbose (bool): Whether to print out verbose information while
        processing.
      use_strict_trace (bool): Whether to pass keyword argument `strict` to
        `torch.jit.trace`. Pass False when you want the tracer to
        record your mutable container types (list, dict)
    """
    # 将模型设置为评估模式，并临时处理
    with _set_model_to_eval(model):
        try:
            # 使用 Torch 的追踪功能来追踪模型执行路径
            trace = torch.jit.trace(model, args, strict=use_strict_trace)
            # 获取追踪生成的图形对象
            graph = trace.graph
            # 在图形上执行内联优化
            torch._C._jit_pass_inline(graph)
        except RuntimeError as e:
            # 捕获运行时异常并打印错误信息
            print(e)
            print("Error occurs, No graph saved")
            # 重新引发异常
            raise e

    if verbose:
        # 如果指定了 verbose 参数，打印生成的图形对象
        print(graph)
    # 解析图形对象，并返回节点列表
    list_of_nodes = parse(graph, trace, args)
    
    # 硬编码指示此过程在 CPU 上运行，即使实际上可能在 GPU 上运行
    # 注意：此信息用于 TensorBoard 显示，与实际执行无关
    # TODO: 查看是否可以从 PyTorch 模型中提取 GPU vs CPU 信息，并正确传递给 TensorBoard。
    #
    # StepStats 和 DeviceStepStats 的定义可以在以下链接找到：
    # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/graph/tf_graph_common/test/graph-test.ts
    # 和
    # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/step_stats.proto
    stepstats = RunMetadata(
        step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")])
    )
    # 返回图的定义和版本信息
    return GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)), stepstats
    # 生产者版本是从标准 TensorBoard 记录的数据中反向工程得出的。


@contextlib.contextmanager
def _set_model_to_eval(model):
    """Context manager to temporarily set the training mode of ``model`` to eval."""
    # 如果模型不是 torch.jit.ScriptFunction 类型
    if not isinstance(model, torch.jit.ScriptFunction):
        # 保存原始的训练模式，并将模型设置为评估模式
        originally_training = model.training
        model.train(False)
        try:
            # 在上下文中执行操作
            yield
        finally:
            # 恢复原始的训练模式
            model.train(originally_training)
    else:
        # 对于 ScriptFunction 类型的模型，不执行任何操作
        try:
            yield
        finally:
            pass


def _node_get(node: torch._C.Node, key: str):
    """Get attributes of a node which is polymorphic over return type."""
    # 获取节点的指定属性，并支持多态的返回类型
    sel = node.kindOf(key)
    return getattr(node, sel)(key)
```