# `.\pytorch\torch\utils\tensorboard\_proto_graph.py`

```
# mypy: allow-untyped-defs
# 导入必要的类型定义
from typing import Optional
# 导入相关的 Protobuf 消息定义
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto


def attr_value_proto(dtype, shape, s):
    """Create a dict of objects matching a NodeDef's attr field.

    根据 NodeDef 的 attr 字段创建一个对象字典。
    根据 https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/attr_value.proto
    这个链接指定的格式，特别适用于 NodeDef。这些值是从标准 TensorBoard 记录的数据中反向工程出来的。
    """
    attr = {}
    if s is not None:
        attr["attr"] = AttrValue(s=s.encode(encoding="utf_8"))
    if shape is not None:
        shapeproto = tensor_shape_proto(shape)
        attr["_output_shapes"] = AttrValue(list=AttrValue.ListValue(shape=[shapeproto]))
    return attr


def tensor_shape_proto(outputsize):
    """Create an object matching a tensor_shape field.

    根据 tensor_shape 字段创建一个匹配的对象。
    遵循 https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor_shape.proto 。
    """
    return TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in outputsize])


def node_proto(
    name,
    op="UnSpecified",
    input=None,
    dtype=None,
    shape: Optional[tuple] = None,
    outputsize=None,
    attributes="",
):
    """Create an object matching a NodeDef.

    根据 NodeDef 创建一个匹配的对象。
    遵循 https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/node_def.proto 。
    """
    if input is None:
        input = []
    if not isinstance(input, list):
        input = [input]
    return NodeDef(
        name=name.encode(encoding="utf_8"),
        op=op,
        input=input,
        attr=attr_value_proto(dtype, outputsize, attributes),
    )
```