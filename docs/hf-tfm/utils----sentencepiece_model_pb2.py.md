# `.\transformers\utils\sentencepiece_model_pb2.py`

```
# 由协议缓冲区编译器生成。请勿编辑！
# 来源：sentencepiece_model.proto

# 版权所有 2022 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database


# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


# 创建文件描述符
DESCRIPTOR = _descriptor.FileDescriptor(
    name="sentencepiece_model.proto",
    package="sentencepiece",
    syntax="proto2",
    serialized_options=b"H\003",
    create_key=_descriptor._internal_create_key,
    ),
)


# 创建枚举描述符
_TRAINERSPEC_MODELTYPE = _descriptor.EnumDescriptor(
    name="ModelType",
    full_name="sentencepiece.TrainerSpec.ModelType",
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        # 枚举值描述符
        _descriptor.EnumValueDescriptor(
            name="UNIGRAM",
            index=0,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="BPE",
            index=1,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="WORD",
            index=2,
            number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="CHAR",
            index=3,
            number=4,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=1294,
    serialized_end=1347,
)
_sym_db.RegisterEnumDescriptor(_TRAINERSPEC_MODELTYPE)

# 创建枚举描述符
_MODELPROTO_SENTENCEPIECE_TYPE = _descriptor.EnumDescriptor(
    name="Type",
    full_name="sentencepiece.ModelProto.SentencePiece.Type",
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    # 定义一个包含多个枚举值的列表
    values=[
        # 创建一个枚举值描述符对象，表示NORMAL
        _descriptor.EnumValueDescriptor(
            name="NORMAL",  # 枚举值名称为NORMAL
            index=0,  # 枚举值在列表中的索引为0
            number=1,  # 枚举值的编号为1
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        # 创建一个枚举值描述符对象，表示UNKNOWN
        _descriptor.EnumValueDescriptor(
            name="UNKNOWN",  # 枚举值名称为UNKNOWN
            index=1,  # 枚举值在列表中的索引为1
            number=2,  # 枚举值的编号为2
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        # 创建一个枚举值描述符对象，表示CONTROL
        _descriptor.EnumValueDescriptor(
            name="CONTROL",  # 枚举值名称为CONTROL
            index=2,  # 枚举值在列表中的索引为2
            number=3,  # 枚举值的编号为3
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        # 创建一个枚举值描述符对象，表示USER_DEFINED
        _descriptor.EnumValueDescriptor(
            name="USER_DEFINED",  # 枚举值名称为USER_DEFINED
            index=3,  # 枚举值在列表中的索引为3
            number=4,  # 枚举值的编号为4
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        # 创建一个枚举值描述符对象，表示BYTE
        _descriptor.EnumValueDescriptor(
            name="BYTE",  # 枚举值名称为BYTE
            index=4,  # 枚举值在列表中的索引为4
            number=6,  # 枚举值的编号为6
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        # 创建一个枚举值描述符对象，表示UNUSED
        _descriptor.EnumValueDescriptor(
            name="UNUSED",  # 枚举值名称为UNUSED
            index=5,  # 枚举值在列表中的索引为5
            number=5,  # 枚举值的编号为5
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,  # 枚举类型的包含类型为None
    serialized_options=None,
    serialized_start=2100,  # 序列化起始位置为2100
    serialized_end=2184,  # 序列化结束位置为2184
# 注册枚举描述符_MODELPROTO_SENTENCEPIECE_TYPE到符号数据库
_sym_db.RegisterEnumDescriptor(_MODELPROTO_SENTENCEPIECE_TYPE)

# 定义TrainerSpec消息类型
_TRAINERSPEC = _descriptor.Descriptor(
    name="TrainerSpec",
    full_name="sentencepiece.TrainerSpec",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    ],
    extensions=[],
    nested_types=[],
    enum_types=[
        _TRAINERSPEC_MODELTYPE,
    ],
    serialized_options=None,
    is_extendable=True,
    syntax="proto2",
    extension_ranges=[
        (200, 536870912),
    ],
    oneofs=[],
    serialized_start=45,
    serialized_end=1358,
)

# 定义NormalizerSpec消息类型
_NORMALIZERSPEC = _descriptor.Descriptor(
    name="NormalizerSpec",
    full_name="sentencepiece.NormalizerSpec",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=True,
    syntax="proto2",
    extension_ranges=[
        (200, 536870912),
    ],
    oneofs=[],
    serialized_start=1361,
    serialized_end=1570,
)

# 定义SelfTestData消息类型中的Sample字段
_SELFTESTDATA_SAMPLE = _descriptor.Descriptor(
    name="Sample",
    full_name="sentencepiece.SelfTestData.Sample",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="input",
            full_name="sentencepiece.SelfTestData.Sample.input",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="expected",
            full_name="sentencepiece.SelfTestData.Sample.expected",
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto2",
    extension_ranges=[],
    oneofs=[],
    serialized_start=1641,
    serialized_end=1682,
)

# 定义SelfTestData消息类型
_SELFTESTDATA = _descriptor.Descriptor(
    name="SelfTestData",
    full_name="sentencepiece.SelfTestData",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    # 创建一个内部的创建键
    create_key=_descriptor._internal_create_key,
    # 定义字段描述符列表
    fields=[
        # 定义字段描述符对象
        _descriptor.FieldDescriptor(
            name="samples",  # 字段名
            full_name="sentencepiece.SelfTestData.samples",  # 完整字段名
            index=0,  # 索引
            number=1,  # 编号
            type=11,  # 类型
            cpp_type=10,  # C++ 类型
            label=3,  # 标签
            has_default_value=False,  # 是否有默认值
            default_value=[],  # 默认值
            message_type=None,  # 消息类型
            enum_type=None,  # 枚举类型
            containing_type=None,  # 包含类型
            is_extension=False,  # 是否是扩展
            extension_scope=None,  # 扩展范围
            serialized_options=None,  # 序列化选项
            file=DESCRIPTOR,  # 文件
            create_key=_descriptor._internal_create_key,  # 创建键
        ),
    ],
    # 扩展列表
    extensions=[],
    # 嵌套类型列表
    nested_types=[
        _SELFTESTDATA_SAMPLE,
    ],
    # 枚举类型列表
    enum_types=[],
    # 序列化选项
    serialized_options=None,
    # 是否可扩展
    is_extendable=True,
    # 语法版本
    syntax="proto2",
    # 扩展范围
    extension_ranges=[
        (200, 536870912),
    ],
    # 一对一关系列表
    oneofs=[],
    # 序列化起始位置
    serialized_start=1572,
    # 序列化结束位置
    serialized_end=1693,
# 定义了一个名为 SentencePiece 的消息类型，包含了 piece、score 和 type 三个字段
_MODELPROTO_SENTENCEPIECE = _descriptor.Descriptor(
    name="SentencePiece",
    full_name="sentencepiece.ModelProto.SentencePiece",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        # 定义了一个名为 piece 的字段
        _descriptor.FieldDescriptor(
            name="piece",
            full_name="sentencepiece.ModelProto.SentencePiece.piece",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        # 定义了一个名为 score 的字段
        _descriptor.FieldDescriptor(
            name="score",
            full_name="sentencepiece.ModelProto.SentencePiece.score",
            index=1,
            number=2,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        # 定义了一个名为 type 的字段
        _descriptor.FieldDescriptor(
            name="type",
            full_name="sentencepiece.ModelProto.SentencePiece.type",
            index=2,
            number=3,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=True,
            default_value=1,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[
        _MODELPROTO_SENTENCEPIECE_TYPE,
    ],
    serialized_options=None,
    is_extendable=True,
    syntax="proto2",
    extension_ranges=[
        (200, 536870912),
    ],
    oneofs=[],
    serialized_start=1985,
    serialized_end=2195,
)

# 定义了一个名为 ModelProto 的消息���型，包含了 SentencePiece 类型的字段
_MODELPROTO = _descriptor.Descriptor(
    name="ModelProto",
    full_name="sentencepiece.ModelProto",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    ],
    extensions=[],
    nested_types=[
        _MODELPROTO_SENTENCEPIECE,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=True,
    syntax="proto2",
    extension_ranges=[
        (200, 536870912),
    ],
    oneofs=[],
    serialized_start=1696,
    serialized_end=2206,
)
# 设置_TRAINERSPEC.fields_by_name["model_type"]的枚举类型为_TRAINERSPEC_MODELTYPE
_TRAINERSPEC.fields_by_name["model_type"].enum_type = _TRAINERSPEC_MODELTYPE
# 设置_TRAINERSPEC_MODELTYPE的包含类型为_TRAINERSPEC
_TRAINERSPEC_MODELTYPE.containing_type = _TRAINERSPEC
# 设置_SELFTESTDATA_SAMPLE的包含类型为_SELFTESTDATA
_SELFTESTDATA_SAMPLE.containing_type = _SELFTESTDATA
# 设置_SELFTESTDATA.fields_by_name["samples"]的消息类型为_SELFTESTDATA_SAMPLE
_SELFTESTDATA.fields_by_name["samples"].message_type = _SELFTESTDATA_SAMPLE
# 设置_MODELPROTO_SENTENCEPIECE.fields_by_name["type"]的枚举类型为_MODELPROTO_SENTENCEPIECE_TYPE
_MODELPROTO_SENTENCEPIECE.fields_by_name["type"].enum_type = _MODELPROTO_SENTENCEPIECE_TYPE
# 设置_MODELPROTO_SENTENCEPIECE的包含类型为_MODELPROTO
_MODELPROTO_SENTENCEPIECE.containing_type = _MODELPROTO
# 设置_MODELPROTO_SENTENCEPIECE_TYPE的包含类型为_MODELPROTO_SENTENCEPIECE
_MODELPROTO_SENTENCEPIECE_TYPE.containing_type = _MODELPROTO_SENTENCEPIECE
# 设置_MODELPROTO.fields_by_name["pieces"]的消息类型为_MODELPROTO_SENTENCEPIECE
_MODELPROTO.fields_by_name["pieces"].message_type = _MODELPROTO_SENTENCEPIECE
# 设置_MODELPROTO.fields_by_name["trainer_spec"]的消息类型为_TRAINERSPEC
_MODELPROTO.fields_by_name["trainer_spec"].message_type = _TRAINERSPEC
# 设置_MODELPROTO.fields_by_name["normalizer_spec"]的消息类型为_NORMALIZERSPEC
_MODELPROTO.fields_by_name["normalizer_spec"].message_type = _NORMALIZERSPEC
# 设置_MODELPROTO.fields_by_name["self_test_data"]的消息类型为_SELFTESTDATA
_MODELPROTO.fields_by_name["self_test_data"].message_type = _SELFTESTDATA
# 设置_MODELPROTO.fields_by_name["denormalizer_spec"]的消息类型为_NORMALIZERSPEC
_MODELPROTO.fields_by_name["denormalizer_spec"].message_type = _NORMALIZERSPEC
# 将"TrainerSpec"注册到DESCRIPTOR的消息类型中
DESCRIPTOR.message_types_by_name["TrainerSpec"] = _TRAINERSPEC
# 将"NormalizerSpec"注册到DESCRIPTOR的消息类型中
DESCRIPTOR.message_types_by_name["NormalizerSpec"] = _NORMALIZERSPEC
# 将"SelfTestData"注册到DESCRIPTOR的消息类型中
DESCRIPTOR.message_types_by_name["SelfTestData"] = _SELFTESTDATA
# 将"ModelProto"注册到DESCRIPTOR的消息类型中
DESCRIPTOR.message_types_by_name["ModelProto"] = _MODELPROTO
# 注册文件描述符到_sym_db
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

# 创建并注册TrainerSpec消息类型
TrainerSpec = _reflection.GeneratedProtocolMessageType(
    "TrainerSpec",
    (_message.Message,),
    {
        "DESCRIPTOR": _TRAINERSPEC,
        "__module__": "sentencepiece_model_pb2",
        # @@protoc_insertion_point(class_scope:sentencepiece.TrainerSpec)
    },
)
_sym_db.RegisterMessage(TrainerSpec)

# 创建并注册NormalizerSpec消息类型
NormalizerSpec = _reflection.GeneratedProtocolMessageType(
    "NormalizerSpec",
    (_message.Message,),
    {
        "DESCRIPTOR": _NORMALIZERSPEC,
        "__module__": "sentencepiece_model_pb2",
        # @@protoc_insertion_point(class_scope:sentencepiece.NormalizerSpec)
    },
)
_sym_db.RegisterMessage(NormalizerSpec)

# 创建并注册SelfTestData消息类型
SelfTestData = _reflection.GeneratedProtocolMessageType(
    "SelfTestData",
    (_message.Message,),
    {
        "Sample": _reflection.GeneratedProtocolMessageType(
            "Sample",
            (_message.Message,),
            {
                "DESCRIPTOR": _SELFTESTDATA_SAMPLE,
                "__module__": "sentencepiece_model_pb2",
                # @@protoc_insertion_point(class_scope:sentencepiece.SelfTestData.Sample)
            },
        ),
        "DESCRIPTOR": _SELFTESTDATA,
        "__module__": "sentencepiece_model_pb2",
        # @@protoc_insertion_point(class_scope:sentencepiece.SelfTestData)
    },
)
_sym_db.RegisterMessage(SelfTestData)
_sym_db.RegisterMessage(SelfTestData.Sample)

# 创建并注册ModelProto消息类型
ModelProto = _reflection.GeneratedProtocolMessageType(
    "ModelProto",
    (_message.Message,),
    {
        # 定义 SentencePiece 类，继承自 Message 类
        "SentencePiece": _reflection.GeneratedProtocolMessageType(
            "SentencePiece",
            (_message.Message,),
            {
                # 设置描述符为 _MODELPROTO_SENTENCEPIECE
                "DESCRIPTOR": _MODELPROTO_SENTENCEPIECE,
                "__module__": "sentencepiece_model_pb2",
                # @@protoc_insertion_point(class_scope:sentencepiece.ModelProto.SentencePiece)
            },
        ),
        # 设置描述符为 _MODELPROTO
        "DESCRIPTOR": _MODELPROTO,
        "__module__": "sentencepiece_model_pb2",
        # @@protoc_insertion_point(class_scope:sentencepiece.ModelProto)
    },
# 注册 ModelProto 消息到符号数据库
_sym_db.RegisterMessage(ModelProto)
# 注册 ModelProto.SentencePiece 消息到符号数据库
_sym_db.RegisterMessage(ModelProto.SentencePiece)

# 设置 DESCRIPTOR 的 _options 为 None
DESCRIPTOR._options = None
# 设置 TRAINERSPEC 中字段 "mining_sentence_size" 的 _options 为 None
_TRAINERSPEC.fields_by_name["mining_sentence_size"]._options = None
# 设置 TRAINERSPEC 中字段 "training_sentence_size" 的 _options 为 None
_TRAINERSPEC.fields_by_name["training_sentence_size"]._options = None
# 插入协议缩进点
# @@protoc_insertion_point(module_scope)
```