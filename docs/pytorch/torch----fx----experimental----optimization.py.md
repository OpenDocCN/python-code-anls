# `.\pytorch\torch\fx\experimental\optimization.py`

```
# 添加类型检查的标志，允许未定义类型的函数
# 模块导入：导入 torch.fx 库作为 fx 别名
# 导入 torch.fx.node 模块中的 Argument, Target 类
# 导入 torch.nn.utils.fusion 模块中的 fuse_conv_bn_eval 函数
# 导入 Type, Dict, Any, Tuple, Iterable, Optional, List, cast 类型及函数
# 导入 torch 库
# 导入 torch.nn 库
# 导入 torch.nn.functional 库
# 导入 torch.fx.passes.shape_prop 模块中的 ShapeProp 类
# 导入 copy 库
# 导入 collections 库中的 defaultdict 函数
# 导入 torch.utils.mkldnn 库作为 th_mkldnn 别名
# 导入 operator 库
# 导入 time 库
# 导入 logging 库
# 导入 enum 库中的 Enum 类

# 定义函数 _parent_name，接收一个字符串类型参数 target，返回一个元组 (父路径, 最后一个元素)
def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    # 将目标字符串按最后一个点号分隔，得到父路径和最后一个元素
    *parent, name = target.rsplit('.', 1)
    # 返回父路径的字符串形式，如果没有父路径则返回空字符串，以及最后一个元素的字符串形式
    return parent[0] if parent else '', name

# 定义函数 matches_module_pattern，接收三个参数：模式 pattern，节点 node，模块字典 modules
def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    # 如果节点的参数个数为 0，则返回 False
    if len(node.args) == 0:
        return False
    # 将节点的第一个参数和节点本身组成一个元组
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    # 遍历模式和节点的元组
    for expected_type, current_node in zip(pattern, nodes):
        # 如果当前节点不是 fx.Node 类型，则返回 False
        if not isinstance(current_node, fx.Node):
            return False
        # 如果当前节点的操作不是 'call_module'，则返回 False
        if current_node.op != 'call_module':
            return False
        # 如果当前节点的目标不是字符串类型，则返回 False
        if not isinstance(current_node.target, str):
            return False
        # 如果当前节点的目标不在模块字典中，则返回 False
        if current_node.target not in modules:
            return False
        # 如果当前节点对应的模块类型不是预期类型 expected_type，则返回 False
        if type(modules[current_node.target]) is not expected_type:
            return False
    # 如果所有条件都满足，则返回 True
    return True

# 定义函数 replace_node_module，接收三个参数：节点 node，模块字典 modules，新模块 new_module
def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    # 断言节点的目标是字符串类型
    assert isinstance(node.target, str)
    # 调用 _parent_name 函数，获取父路径和模块名
    parent_name, name = _parent_name(node.target)
    # 更新模块字典，将节点的目标指向新模块
    modules[node.target] = new_module
    # 设置父模块中的对应属性为新模块
    setattr(modules[parent_name], name, new_module)

# 定义函数 fuse，接收三个参数：torch.nn.Module 类型的 model，布尔类型的 inplace，默认为 False，布尔类型的 no_trace，默认为 False
# 返回一个 torch.nn.Module 类型的对象
def fuse(model: torch.nn.Module, inplace=False, no_trace=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    # 定义融合模式列表，包括 (nn.Conv1d, nn.BatchNorm1d)，(nn.Conv2d, nn.BatchNorm2d)，(nn.Conv3d, nn.BatchNorm3d)
    patterns = [(nn.Conv1d, nn.BatchNorm1d),
                (nn.Conv2d, nn.BatchNorm2d),
                (nn.Conv3d, nn.BatchNorm3d)]
    # 如果 inplace 参数为 False，则深度复制模型
    if not inplace:
        model = copy.deepcopy(model)
    # 如果不允许无迹或者模型不是 torch.fx.GraphModule 类型，则进行符号跟踪
    if not no_trace or not isinstance(model, torch.fx.GraphModule):
        fx_model = fx.symbolic_trace(model)
    else:
        fx_model = model
    # 创建模块字典，包含 fx_model 中所有命名模块的映射
    modules = dict(fx_model.named_modules())
    # 深度复制 fx_model 的图形结构
    new_graph = copy.deepcopy(fx_model.graph)
    # 遍历给定的模式列表
    for pattern in patterns:
        # 遍历新图中的每个节点
        for node in new_graph.nodes:
            # 检查当前节点是否匹配指定模式并且属于给定的模块集合
            if matches_module_pattern(pattern, node, modules):
                # 如果当前节点的第一个参数被多个节点使用，则跳过
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                # 获取当前节点的第一个参数对应的卷积模块
                conv = modules[node.args[0].target]
                # 获取当前节点对应的批归一化模块
                bn = modules[node.target]
                # 如果批归一化模块不跟踪运行时统计信息，则跳过
                if not bn.track_running_stats:
                    continue
                # 将卷积和批归一化模块融合，并得到融合后的卷积模块
                fused_conv = fuse_conv_bn_eval(conv, bn)
                # 替换当前节点的第一个参数为融合后的卷积模块
                replace_node_module(node.args[0], modules, fused_conv)
                # 将当前节点的所有使用替换为其第一个参数
                node.replace_all_uses_with(node.args[0])
                # 在新图中删除当前节点
                new_graph.erase_node(node)
    # 返回经过修改后的 GraphModule 对象
    return fx.GraphModule(fx_model, new_graph)
# 从模块中移除所有的 dropout 层，返回修改后的模块
def remove_dropout(model: nn.Module) -> nn.Module:
    # 使用 FX 符号化追踪模型，以获取模型的符号化表示
    fx_model = fx.symbolic_trace(model)

    # 定义一个特殊的转换器类，用于移除 dropout 层
    class DropoutRemover(torch.fx.Transformer):
        # 重写 call_module 方法，在模块调用时进行修改
        def call_module(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
            # 如果目标模块是 nn.Dropout 类型，则将其调用参数直接返回
            if isinstance(self.submodules[target], nn.Dropout):
                assert len(args) == 1
                return args[0]
            else:
                # 否则，继续调用父类的 call_module 方法
                return super().call_module(target, args, kwargs)
    
    # 使用 DropoutRemover 实例对符号化模型进行转换并返回
    return DropoutRemover(fx_model).transform()

# 提取原始模块中指定子图节点的子模块
def extract_subgraph(orig_module: nn.Module, nodes: List[fx.Node], inputs: List[fx.Node], outputs: List[fx.Node]):
    # 创建一个新的 FX 图形对象
    new_graph = fx.Graph()
    # 用于映射输入节点和新节点的字典环境
    env: Dict[fx.Node, fx.Node] = {}
    
    # 复制输入节点到新图中，并建立环境映射
    for input in inputs:
        new_node = new_graph.placeholder(input.name)
        env[input] = new_node
    
    # 复制子图中的每个节点到新图中，并更新环境映射
    for node in nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x])
        env[node] = new_node
    
    # 指定新图的输出节点
    new_graph.output([env[output] for output in outputs])
    # 检查新图的一致性
    new_graph.lint()
    
    # 创建并返回一个新的图形模块，将原始模块和新图结合起来
    return fx.GraphModule(orig_module, new_graph)

# 可以预先转换为 MKLDNN 格式的模块列表
mkldnn_supported = [
    nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
    torch.relu, torch.transpose, torch.sigmoid,
    F.relu, F.avg_pool2d, F.adaptive_avg_pool2d
]

# 这些运算符可能无法转换为 MKLDNN 操作（例如，参数是标量值），因此仅当其参数已经是 MKLDNN 时，才将它们包含在子图中
# TODO: 在类型推断后确定是否可以移除这些
mkldnn_supported_unknown = [operator.add, operator.mul]

# 定义模块到 MKLDNN 模块的映射关系
mkldnn_map = {
    nn.Conv2d: th_mkldnn.MkldnnConv2d,
    nn.Linear: th_mkldnn.MkldnnLinear,
    nn.BatchNorm2d: lambda a, _: th_mkldnn.MkldnnBatchNorm(a)
}

# 将节点列表中的模块转换为 MKLDNN 格式，并返回转换前的模块列表
def modules_to_mkldnn(nodes: List[fx.Node], modules: Dict[str, nn.Module]):
    # 用于存储原始模块到新 MKLDNN 模块的映射关系
    old_modules: Dict[nn.Module, nn.Module] = {}
    
    # 遍历节点列表
    for node in nodes:
        # 如果节点是模块调用节点
        if node.op == 'call_module':
            assert isinstance(node.target, str)
            cur_module = modules[node.target]
            # 如果当前模块类型在 mkldnn_map 中
            if type(cur_module) in mkldnn_map:
                # 创建新的 MKLDNN 格式模块
                new_module = mkldnn_map[type(cur_module)](cur_module, torch.float)
                assert isinstance(new_module, nn.Module)
                # 备份并替换节点的模块
                old_modules[new_module] = copy.deepcopy(cur_module)
                replace_node_module(node, modules, new_module)
    
    # 返回原始模块到新 MKLDNN 模块的映射关系
    return old_modules

# 将已经改变的模块（通过 `modules_to_mkldnn` 方法）映射回其原始模块
def reset_modules(nodes: List[fx.Node], modules: Dict[str, nn.Module], old_modules: Dict[nn.Module, nn.Module]):
    # TODO: 完成映射回原始模块的逻辑
    pass
    # 遍历节点列表 `nodes`
    for node in nodes:
        # 检查节点操作是否为调用模块
        if node.op == 'call_module':
            # 断言节点的目标（模块名称）为字符串类型
            assert (isinstance(node.target, str))
            # 获取当前节点目标对应的模块
            cur_module = modules[node.target]
            # 如果当前模块在旧模块字典中
            if cur_module in old_modules:
                # 替换当前节点中的模块为旧模块对应的新模块
                replace_node_module(node, modules, old_modules[cur_module])
class MklSubgraph:
    def __init__(self, fx_graph: fx.Graph):
        # 初始化函数，将传入的 fx.Graph 对象存储在实例变量中
        self.fx_graph = fx_graph
        # 初始化空列表，用于存储图中的节点
        self.nodes: List[fx.Node] = []
        # 初始化空列表，用于存储起始节点
        self.start_nodes: List[fx.Node] = []
        # 初始化空列表，用于存储结束节点
        self.end_nodes: List[fx.Node] = []

def gen_mkl_autotuner(example_inputs, iters=10, warmup=1):
    """
    This generates a heuristic that can be passed into `optimize_for_inference` that
    determines whether a subgraph should be run in MKL by running it with the example_inputs.

    Example usage:
        heuristic = gen_mkl_autotuner(example_inputs, iters=10)
        fast_model = optimization.optimize_for_inference(model, heuristic)
    """
    fx_model = None
    old_modules = None

    def use_mkl_heuristic(graph: MklSubgraph) -> bool:
        nonlocal fx_model, old_modules
        # 获取图的起始节点
        input_nodes = graph.start_nodes
        if fx_model is None:
            # 如果 fx_model 尚未初始化，则将其设置为图的拥有模块
            fx_model = graph.fx_graph.owning_module
            # 获取图的旧模块
            old_modules = graph.fx_graph.old_modules  # type: ignore[attr-defined]
            # 使用示例输入传播形状属性到模型
            ShapeProp(fx_model).propagate(example_inputs)
        # 生成样本输入，每个节点的形状为随机生成的张量
        sample_inputs = [torch.randn(node.shape) for node in input_nodes]  # type: ignore[attr-defined]
        # 获取结束节点的参数列表
        output_args = cast(List[fx.Node], [node.args[0] for node in graph.end_nodes])
        # 提取子图
        submodule = extract_subgraph(fx_model, graph.nodes, input_nodes, output_args)

        def benchmark(f):
            # 对函数进行预热
            for _ in range(warmup):
                f()
            begin = time.time()
            # 进行性能测试
            for _ in range(iters):
                out = f()
            return time.time() - begin

        # 测试使用 MKL 的时间
        mkl_time = benchmark(lambda: [i.to_dense() for i in submodule(*[i.to_mkldnn() for i in sample_inputs])])

        # 恢复模块状态
        reset_modules(submodule.graph.nodes, dict(submodule.named_modules()), old_modules)
        # 测试不使用 MKL 的时间
        no_mkl_time = benchmark(lambda: submodule(*sample_inputs))
        # 返回是否使用 MKL 的启发式判断结果
        return mkl_time < no_mkl_time
    return use_mkl_heuristic

def use_mkl_length(graph: MklSubgraph) -> bool:
    """
    This is a heuristic that can be passed into `optimize_for_inference` that
    determines whether a subgraph should be run in MKL by checking if there
    are more than 2 nodes in it
    """
    # 判断图中节点数量是否大于2，返回结果作为是否使用 MKL 的依据
    return len(graph.nodes) > 2

class UnionFind:
    def __init__(self, n):
        # 初始化函数，创建并初始化并查集
        self.parent: List[Optional[int]] = [None] * n
        self.size: List[int] = [0] * n

    def make_set(self, v: int):
        # 创建新的集合，将 v 加入到集合中
        self.parent[v] = v
        self.size[v] = 1

    def find(self, v: int) -> int:
        # 查找 v 所属的集合，并进行路径压缩优化
        par = self.parent[v]
        if v == par:
            return v
        assert par is not None
        self.parent[v] = self.find(par)
        return cast(int, self.parent[v])

    def join(self, a: int, b: int):
        # 将两个元素所在的集合进行合并
        a, b = self.find(a), self.find(b)
        if a == b:
            return a
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]

def optimize_for_inference(
    model: torch.nn.Module,
    pass_config: Optional[Dict[str, Any]] = None,
    tracer: Type[fx.Tracer] = fx.Tracer
    """
    为了进行推理目的的优化，执行一系列优化步骤以优化模型。具体运行的步骤包括：
    1. 合并 Conv/BN（Batch Normalization）
    2. 移除 Dropout 层
    3. MKL 布局优化

    第三个优化步骤需要一个名为 `use_mkl_heuristic` 的函数，用于确定是否应该显式地在 MKL 布局下运行子图。

    注意：由于 FX 目前不处理别名，此优化假设不存在别名。如果这不成立，请自行承担风险。
    """
    default_pass_config = {
        "conv_bn_fuse": True,               # 是否进行 Conv/BN 合并优化
        "remove_dropout": True,             # 是否移除 Dropout 层
        "mkldnn_layout_optimize": {'heuristic': use_mkl_length},  # MKL 布局优化配置，使用指定的启发式方法
    }

    if pass_config is None:
        pass_config = {}

    default_pass_config.update(pass_config)   # 更新默认配置

    if default_pass_config["conv_bn_fuse"]:
        model = fuse(model)   # 执行 Conv/BN 合并优化

    if default_pass_config["remove_dropout"]:
        model = remove_dropout(model)   # 移除 Dropout 层

    if default_pass_config["mkldnn_layout_optimize"] is False:
        return model   # 如果禁用了 MKL 布局优化，则直接返回模型

    if not isinstance(default_pass_config["mkldnn_layout_optimize"], dict):
        raise RuntimeError("mkldnn_layout_optimize config is not a dict")   # 检查配置是否为字典类型

    if "heuristic" not in default_pass_config["mkldnn_layout_optimize"]:
        raise RuntimeError("Heuristic not found in mkldnn_layout_optimize config")   # 检查是否在配置中找到了启发式方法

    use_mkl_heuristic = default_pass_config["mkldnn_layout_optimize"]["heuristic"]   # 获取启发式方法

    cur_tracer = tracer()   # 创建追踪器对象
    fx_graph = cur_tracer.trace(copy.deepcopy(model))   # 对模型进行深度复制并追踪计算图
    fx_model = fx.GraphModule(cur_tracer.root, fx_graph)   # 构建 FX 的图模块
    modules: Dict[str, nn.Module] = dict(model.named_modules())   # 获取模型的所有模块

    class MklSupport(Enum):
        NO = 1
        YES = 2
        UNKNOWN = 3

    # 在我们希望成为 MKLDNN 节点的每个节点周围插入 to_mkldnn 和 to_dense 操作。
    # 如果操作在 `mkldnn_supported` 中，则始终将其视为 MKLDNN 节点。
    # 但是，如果操作在 `mkldnn_supported_unknown` 中，那么仅当其输入为 MKLDNN 节点时才将其视为 MKLDNN 节点。
    # 遍历图中的每个节点
    for node in list(fx_graph.nodes):
        # 默认不支持 MKLDNN
        supports_mkldnn = MklSupport.NO
        
        # 如果节点操作是调用模块
        if node.op == 'call_module':
            # 获取当前模块
            cur_module = modules[node.target]
            
            # 如果当前模块的类型在支持的 MKLDNN 模块列表中
            if type(cur_module) in mkldnn_supported:
                supports_mkldnn = MklSupport.YES
                
                # 获取当前模块的一个样例参数
                sample_parameter = next(cur_module.parameters(), None)
                
                # 如果样例参数不为空
                if sample_parameter is not None:
                    # 确保参数类型为 torch.float，此转换仅适用于 torch.float 类型的模块
                    assert sample_parameter.dtype == torch.float, "this pass is only for torch.float modules"
                    # 确保参数在 CPU 上，此转换仅适用于 CPU 上的模块
                    assert sample_parameter.device == torch.device('cpu'), "this pass is only for CPU modules"
        
        # 如果节点操作是调用函数
        elif node.op == 'call_function':
            # 如果函数在支持的 MKLDNN 函数列表中
            if node.target in mkldnn_supported:
                supports_mkldnn = MklSupport.YES
            # 如果函数在未知的 MKLDNN 函数列表中
            elif node.target in mkldnn_supported_unknown:
                supports_mkldnn = MklSupport.UNKNOWN
        
        # 如果支持 MKLDNN 不是 NO
        if supports_mkldnn != MklSupport.NO:
            # 如果支持 MKLDNN 是 UNKNOWN，并且节点参数中没有 'to_dense' 的调用，则跳过当前节点
            if supports_mkldnn == MklSupport.UNKNOWN:
                if not any(arg.target == 'to_dense' for arg in node.args):
                    continue
            
            # 在当前节点之前插入新节点
            with fx_graph.inserting_before(node):
                # 将当前节点的参数转换为 MKLDNN 张量
                mkldnn_args = fx.map_arg(node.args, lambda n: fx_graph.call_method('to_mkldnn', (n, )))
            
            # 更新当前节点的参数为转换后的 MKLDNN 参数
            node.args = cast(Tuple[fx.node.Argument], mkldnn_args)
            
            # 在当前节点之后插入新节点
            with fx_graph.inserting_after(node):
                # 创建一个调用 'to_dense' 方法的新节点
                dense_x = fx_graph.create_node('call_method', 'to_dense', (node,))
                # 将当前节点的所有使用替换为新创建的 'to_dense' 节点
                node.replace_all_uses_with(dense_x)
                dense_x.args = (node,)
    
    # 将所有模块预转换为 MKLDNN 格式（如果可能）
    old_modules = modules_to_mkldnn(list(fx_graph.nodes), modules)
    # 将预转换后的模块列表附加到 fx_graph 上
    fx_graph.old_modules = old_modules  # type: ignore[attr-defined]
    
    # 优化所有 a -> to_dense -> to_mkldnn -> b 的模式为 a -> b
    for node in fx_graph.nodes:
        # 如果节点操作是调用方法且目标是 'to_dense'
        if node.op == 'call_method' and node.target == 'to_dense':
            # 获取 'to_dense' 节点的前一个节点
            prv_node = node.args[0]
            # 获取所有使用 'to_dense' 节点的用户节点列表
            users = list(node.users)
            # 遍历每个使用 'to_dense' 节点的用户
            for user in users:
                # 如果用户节点操作是调用方法且目标是 'to_mkldnn'
                if user.op == 'call_method' and user.target == 'to_mkldnn':
                    # 将用户节点的所有使用替换为 'to_dense' 节点的前一个节点
                    user.replace_all_uses_with(prv_node)
                    # 从图中删除用户节点
                    fx_graph.erase_node(user)
            # 如果 'to_dense' 节点没有使用者，则从图中删除该节点
            if len(node.users) == 0:
                fx_graph.erase_node(node)
    
    # 获取图中节点的数量
    num_nodes = len(fx_graph.nodes)
    # 创建一个 UnionFind 数据结构，用于查找节点的连通分量
    uf = UnionFind(num_nodes)
    
    # 定义函数，用于获取节点的颜色（用于表示 MKL 子图）
    def get_color(n):
        # 如果节点有 'color' 属性，则当前节点是 MKL 子图的一部分
        if hasattr(n, 'color'):
            return uf.find(n.color)
        # 如果节点有 'start_color' 属性，则当前节点是 MKL 子图的输入节点
        if hasattr(n, 'start_color'):
            return uf.find(n.start_color)
        return None
    
    # 以下代码用于查找每个 MKLDNN 子图。每个 MKLDNN 子图由输入节点（仅为 'to_mkldnn' 调用）、
    # 输出节点（'to_dense' 调用）和中间节点组成，这些节点完全在 MKLDNN 布局张量上运行。
    #
    # 具体来说，此代码在有向无环图上进行洪泛填充（flood fill）。
    # 对于每个节点，从每个可能的“起始节点”（即`to_mkldnn`节点）开始构建有向无环图(DAG)。
    # 如果每个节点只有一个输入，那么这已经足够了。然而，如果一个节点有多个输入来自不同的起始节点（即颜色），
    # 我们需要将这些不同颜色的节点合并为一个。这是通过不相交集合（Disjoint Set Union）来完成的。
    for cur_idx, node in enumerate(fx_graph.nodes):
        if node.op == 'call_method' and node.target == 'to_mkldnn':
            # 给每个 `to_mkldnn` 节点设置起始颜色，并在并查集中创建对应的集合
            node.start_color = cur_idx
            uf.make_set(cur_idx)
        elif node.op == 'call_method' and node.target == 'to_dense':
            # 对于 `to_dense` 节点，确保其输入节点的颜色不为None，并设置其结束颜色
            assert get_color(node.args[0]) is not None
            node.end_color = get_color(node.args[0])
        else:
            # 获取当前节点所有输入节点的颜色，并确保颜色不为None
            cur_colors = [get_color(i) for i in node.all_input_nodes if isinstance(i, fx.Node) if get_color(i) is not None]

            if len(cur_colors) == 0:
                continue
            # 确保当前节点的所有输入节点的颜色都不为None
            assert not any(i is None for i in cur_colors)
            # 对当前节点的颜色进行排序
            cur_colors = sorted(cur_colors)
            # 将当前节点设置为具有最小颜色的颜色
            node.color = cur_colors[0]
            # 将其他颜色与具有最小颜色的颜色进行合并
            for other_color in cur_colors[1:]:
                uf.join(cur_colors[0], other_color)

    # 创建以节点颜色为键的MKLDNN子图字典，默认使用给定的fx_graph作为子图的初始值
    mkldnn_graphs: Dict[int, MklSubgraph] = defaultdict(lambda: MklSubgraph(fx_graph))
    for node in fx_graph.nodes:
        # 将具有颜色属性的节点添加到相应颜色的MKLDNN子图中
        if hasattr(node, 'color'):
            mkldnn_graphs[uf.find(node.color)].nodes.append(node)
        # 将具有start_color属性的节点添加到相应起始颜色的MKLDNN子图中
        if hasattr(node, 'start_color'):
            mkldnn_graphs[uf.find(node.start_color)].start_nodes.append(node)
        # 将具有end_color属性的节点添加到相应结束颜色的MKLDNN子图中
        if hasattr(node, 'end_color'):
            mkldnn_graphs[uf.find(node.end_color)].end_nodes.append(node)

    # 现在我们有了所有的子图，需要决定哪些MKLDNN子图实际上需要保留在MKLDNN中。
    for graph in mkldnn_graphs.values():
        # 如果不使用MKL启发式方法决定不保留当前子图，则将其起始节点和结束节点的所有用法替换为其先前的用法，并在fx_graph中删除这些节点。
        if not use_mkl_heuristic(graph):
            for node in graph.start_nodes + graph.end_nodes:
                prv = node.args[0]
                node.replace_all_uses_with(prv)
                fx_graph.erase_node(node)
            # 重置图中的模块到其旧模块状态
            reset_modules(graph.nodes, modules, old_modules)

    # 统计在fx_graph中执行的to_mkldnn和to_dense转换的次数
    mkldnn_conversions = 0
    for node in fx_graph.nodes:
        if node.target == 'to_mkldnn' or node.target == 'to_dense':
            mkldnn_conversions += 1

    # 记录MKLDNN转换的次数到日志
    logging.getLogger(__name__).info("mkldnn conversions: %s", mkldnn_conversions)
    # 对fx_graph进行语法检查
    fx_graph.lint()
    # 使用模型和更新后的图形创建并返回一个GraphModule对象
    result = fx.GraphModule(model, fx_graph)
    return result
```