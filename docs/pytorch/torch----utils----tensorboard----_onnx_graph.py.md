# `.\pytorch\torch\utils\tensorboard\_onnx_graph.py`

```
# mypy: allow-untyped-defs
# 引入必要的类和函数
from tensorboard.compat.proto.graph_pb2 import GraphDef  # 导入 GraphDef 类
from tensorboard.compat.proto.node_def_pb2 import NodeDef  # 导入 NodeDef 类
from tensorboard.compat.proto.versions_pb2 import VersionDef  # 导入 VersionDef 类
from tensorboard.compat.proto.attr_value_pb2 import AttrValue  # 导入 AttrValue 类
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto  # 导入 TensorShapeProto 类

# 加载 ONNX 图形并解析
def load_onnx_graph(fname):
    import onnx  # 导入 onnx 模块

    m = onnx.load(fname)  # type: ignore[attr-defined]  # 加载 ONNX 文件并忽略类型定义错误
    g = m.graph  # 获取 ONNX 模型的图形对象
    return parse(g)  # 调用 parse 函数解析图形


# 解析图形结构
def parse(graph):
    nodes = []  # 初始化空列表存放 NodeDef 对象
    import itertools  # 导入 itertools 模块

    # 将图形中的输入和输出节点组合为一个列表
    nodes_proto = list(itertools.chain(graph.input, graph.output))

    # 遍历图形中的输入和输出节点
    for node in nodes_proto:
        print(node.name)  # 打印节点名称
        # 构建 TensorShapeProto 对象，指定节点的形状
        shapeproto = TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=d.dim_value)
                for d in node.type.tensor_type.shape.dim
            ]
        )
        # 创建 NodeDef 对象并添加到 nodes 列表中
        nodes.append(
            NodeDef(
                name=node.name.encode(encoding="utf_8"),  # 将节点名称编码为 UTF-8
                op="Variable",  # 设置操作类型为 Variable
                input=[],  # 设置输入为空列表
                attr={
                    "dtype": AttrValue(type=node.type.tensor_type.elem_type),  # 设置数据类型属性
                    "shape": AttrValue(shape=shapeproto),  # 设置形状属性
                },
            )
        )

    # 遍历图形中的普通节点
    for node in graph.node:
        _attr = []
        # 遍历节点的属性列表
        for s in node.attribute:
            _attr.append(" = ".join([str(f[1]) for f in s.ListFields()]))  # 拼接属性键值对字符串
        attr = ", ".join(_attr).encode(encoding="utf_8")  # 将属性字符串编码为 UTF-8
        print(node.output[0])  # 打印节点的输出名称
        # 创建 NodeDef 对象并添加到 nodes 列表中
        nodes.append(
            NodeDef(
                name=node.output[0].encode(encoding="utf_8"),  # 将节点输出名称编码为 UTF-8
                op=node.op_type,  # 设置操作类型
                input=node.input,  # 设置输入列表
                attr={"parameters": AttrValue(s=attr)},  # 设置参数属性
            )
        )

    # 创建节点名称映射字典，将节点名称与操作类型和节点名称结合
    mapping = {}
    for node in nodes:
        mapping[node.name] = node.op + "_" + node.name

    # 返回包含节点列表和版本信息的 GraphDef 对象
    return GraphDef(node=nodes, versions=VersionDef(producer=22))
```