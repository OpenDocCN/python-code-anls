# `.\utils\sentencepiece_model_pb2.py`

```py
# 由协议缓冲区编译器生成。请勿编辑！
# 源文件：sentencepiece_model.proto

# 版权 2022 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件基于“原样”分发，
# 不提供任何明示或暗示的担保或条件。
# 有关许可证的详细信息，请参阅许可证。

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

# 获取默认的符号数据库实例
_sym_db = _symbol_database.Default()

# 定义描述文件 DESCRIPTOR
DESCRIPTOR = _descriptor.FileDescriptor(
    name="sentencepiece_model.proto",
    package="sentencepiece",
    syntax="proto2",
    serialized_options=b"H\003",
    create_key=_descriptor._internal_create_key,
)

# 定义枚举类型 _TRAINERSPEC_MODELTYPE
_TRAINERSPEC_MODELTYPE = _descriptor.EnumDescriptor(
    name="ModelType",
    full_name="sentencepiece.TrainerSpec.ModelType",
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
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

# 在符号数据库中注册枚举描述符 _TRAINERSPEC_MODELTYPE
_sym_db.RegisterEnumDescriptor(_TRAINERSPEC_MODELTYPE)

# 定义枚举类型 _MODELPROTO_SENTENCEPIECE_TYPE
_MODELPROTO_SENTENCEPIECE_TYPE = _descriptor.EnumDescriptor(
    name="Type",
    full_name="sentencepiece.ModelProto.SentencePiece.Type",
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
)
    values=[
        # 第一个枚举值描述符：NORMAL
        _descriptor.EnumValueDescriptor(
            name="NORMAL",  # 枚举值名称
            index=0,        # 枚举值索引
            number=1,       # 枚举值数值
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        # 第二个枚举值描述符：UNKNOWN
        _descriptor.EnumValueDescriptor(
            name="UNKNOWN",  # 枚举值名称
            index=1,        # 枚举值索引
            number=2,       # 枚举值数值
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        # 第三个枚举值描述符：CONTROL
        _descriptor.EnumValueDescriptor(
            name="CONTROL",  # 枚举值名称
            index=2,        # 枚举值索引
            number=3,       # 枚举值数值
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        # 第四个枚举值描述符：USER_DEFINED
        _descriptor.EnumValueDescriptor(
            name="USER_DEFINED",  # 枚举值名称
            index=3,             # 枚举值索引
            number=4,            # 枚举值数值
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        # 第五个枚举值描述符：BYTE
        _descriptor.EnumValueDescriptor(
            name="BYTE",    # 枚举值名称
            index=4,        # 枚举值索引
            number=6,       # 枚举值数值
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        # 第六个枚举值描述符：UNUSED
        _descriptor.EnumValueDescriptor(
            name="UNUSED",  # 枚举值名称
            index=5,        # 枚举值索引
            number=5,       # 枚举值数值
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,    # 枚举类型的容器类型，这里为None
    serialized_options=None, # 序列化选项，这里为None
    serialized_start=2100,   # 序列化起始位置
    serialized_end=2184,     # 序列化结束位置
# 注册枚举描述符到符号数据库中
_sym_db.RegisterEnumDescriptor(_MODELPROTO_SENTENCEPIECE_TYPE)

# TrainerSpec 类型的描述符对象定义
_TRAINERSPEC = _descriptor.Descriptor(
    name="TrainerSpec",  # 类型名称为 TrainerSpec
    full_name="sentencepiece.TrainerSpec",  # 完整名称指定为 sentencepiece.TrainerSpec
    filename=None,  # 文件名为 None
    file=DESCRIPTOR,  # 使用全局变量 DESCRIPTOR 指定文件
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    # 下面是字段定义的列表
    fields=[
        # 这里省略了字段的具体定义，字段定义包括名称、序号、类型等详细信息
    ],
    extensions=[],  # 扩展字段为空列表
    nested_types=[],  # 嵌套类型为空列表
    enum_types=[
        _TRAINERSPEC_MODELTYPE,  # 包含一个枚举类型 _TRAINERSPEC_MODELTYPE
    ],
    serialized_options=None,  # 序列化选项为 None
    is_extendable=True,  # 可扩展属性设置为 True
    syntax="proto2",  # 使用的协议语法版本为 proto2
    extension_ranges=[  # 扩展范围定义
        (200, 536870912),  # 具体范围的起始值和结束值
    ],
    oneofs=[],  # 不包含任何 oneof 定义
    serialized_start=45,  # 序列化起始位置
    serialized_end=1358,  # 序列化结束位置
)

# NormalizerSpec 类型的描述符对象定义
_NORMALIZERSPEC = _descriptor.Descriptor(
    name="NormalizerSpec",  # 类型名称为 NormalizerSpec
    full_name="sentencepiece.NormalizerSpec",  # 完整名称指定为 sentencepiece.NormalizerSpec
    filename=None,  # 文件名为 None
    file=DESCRIPTOR,  # 使用全局变量 DESCRIPTOR 指定文件
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[],  # 字段定义为空列表
    extensions=[],  # 扩展字段为空列表
    nested_types=[],  # 嵌套类型为空列表
    enum_types=[],  # 枚举类型为空列表
    serialized_options=None,  # 序列化选项为 None
    is_extendable=True,  # 可扩展属性设置为 True
    syntax="proto2",  # 使用的协议语法版本为 proto2
    extension_ranges=[  # 扩展范围定义
        (200, 536870912),  # 具体范围的起始值和结束值
    ],
    oneofs=[],  # 不包含任何 oneof 定义
    serialized_start=1361,  # 序列化起始位置
    serialized_end=1570,  # 序列化结束位置
)

# Sample 类型的描述符对象定义
_SELFTESTDATA_SAMPLE = _descriptor.Descriptor(
    name="Sample",  # 类型名称为 Sample
    full_name="sentencepiece.SelfTestData.Sample",  # 完整名称指定为 sentencepiece.SelfTestData.Sample
    filename=None,  # 文件名为 None
    file=DESCRIPTOR,  # 使用全局变量 DESCRIPTOR 指定文件
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        # 下面是字段列表的具体定义，包括 input 和 expected 两个字段
    ],
    extensions=[],  # 扩展字段为空列表
    nested_types=[],  # 嵌套类型为空列表
    enum_types=[],  # 枚举类型为空列表
    serialized_options=None,  # 序列化选项为 None
    is_extendable=False,  # 不可扩展属性设置为 False
    syntax="proto2",  # 使用的协议语法版本为 proto2
    extension_ranges=[],  # 扩展范围为空列表
    oneofs=[],  # 不包含任何 oneof 定义
    serialized_start=1641,  # 序列化起始位置
    serialized_end=1682,  # 序列化结束位置
)

# SelfTestData 类型的描述符对象定义
_SELFTESTDATA = _descriptor.Descriptor(
    name="SelfTestData",  # 类型名称为 SelfTestData
    full_name="sentencepiece.SelfTestData",  # 完整名称指定为 sentencepiece.SelfTestData
    filename=None,  # 文件名为 None
    file=DESCRIPTOR,  # 使用全局变量 DESCRIPTOR 指定文件
    containing_type=None,
    # 下面省略了其他字段的定义，包括扩展字段、嵌套类型等
)
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="samples",  # 字段名称为 "samples"
            full_name="sentencepiece.SelfTestData.samples",  # 字段的完整名称
            index=0,  # 字段在列表中的索引位置
            number=1,  # 字段编号
            type=11,  # 字段类型（这里是一个特定的整数值）
            cpp_type=10,  # 字段的 C++ 类型（这里是一个特定的整数值）
            label=3,  # 字段的标签类型（这里是一个特定的整数值）
            has_default_value=False,  # 是否有默认值
            default_value=[],  # 默认值为空列表
            message_type=None,  # 消息类型（这里为空）
            enum_type=None,  # 枚举类型（这里为空）
            containing_type=None,  # 包含该字段的类型（这里为空）
            is_extension=False,  # 是否是扩展字段
            extension_scope=None,  # 扩展字段的作用域（这里为空）
            serialized_options=None,  # 序列化选项（这里为空）
            file=DESCRIPTOR,  # 字段所属的文件描述符
            create_key=_descriptor._internal_create_key,  # 创建键的函数
        ),
    ],
    extensions=[],  # 扩展列表为空
    nested_types=[  # 嵌套类型列表
        _SELFTESTDATA_SAMPLE,  # 嵌套类型的引用
    ],
    enum_types=[],  # 枚举类型列表为空
    serialized_options=None,  # 序列化选项为空
    is_extendable=True,  # 可扩展性为 True
    syntax="proto2",  # 协议语法版本
    extension_ranges=[  # 扩展范围列表
        (200, 536870912),  # 扩展范围的起始和结束值
    ],
    oneofs=[],  # OneOf 列表为空
    serialized_start=1572,  # 序列化起始位置
    serialized_end=1693,  # 序列化结束位置
# 创建一个名为 SentencePiece 的消息类型描述符对象
_MODELPROTO_SENTENCEPIECE = _descriptor.Descriptor(
    # 指定消息类型名称为 "SentencePiece"
    name="SentencePiece",
    # 指定消息类型的完整名称
    full_name="sentencepiece.ModelProto.SentencePiece",
    filename=None,
    # 指定该消息类型所属的文件描述符
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        # 定义字段描述符列表开始
        _descriptor.FieldDescriptor(
            # 字段名为 "piece"
            name="piece",
            # 字段的完整名称
            full_name="sentencepiece.ModelProto.SentencePiece.piece",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            # 默认值为空字符串
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
            name="score",
            full_name="sentencepiece.ModelProto.SentencePiece.score",
            index=1,
            number=2,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            # 默认值为浮点数 0.0
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
        _descriptor.FieldDescriptor(
            name="type",
            full_name="sentencepiece.ModelProto.SentencePiece.type",
            index=2,
            number=3,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=True,
            # 默认值为整数 1
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
    # 声明了一个枚举类型 _MODELPROTO_SENTENCEPIECE_TYPE，但未给出具体定义
    enum_types=[
        _MODELPROTO_SENTENCEPIECE_TYPE,
    ],
    serialized_options=None,
    is_extendable=True,
    syntax="proto2",
    # 扩展范围的序列化索引
    extension_ranges=[
        (200, 536870912),
    ],
    oneofs=[],
    # 该消息类型的序列化开始和结束索引
    serialized_start=1985,
    serialized_end=2195,
)

# 创建一个名为 ModelProto 的消息类型描述符对象
_MODELPROTO = _descriptor.Descriptor(
    # 指定消息类型名称为 "ModelProto"
    name="ModelProto",
    # 指定消息类型的完整名称
    full_name="sentencepiece.ModelProto",
    filename=None,
    # 指定该消息类型所属的文件描述符
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    ],
    extensions=[],
    nested_types=[
        # 包含一个名为 SentencePiece 的嵌套消息类型
        _MODELPROTO_SENTENCEPIECE,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=True,
    syntax="proto2",
    # 扩展范围的序列化索引
    extension_ranges=[
        (200, 536870912),
    ],
    oneofs=[],
    # 该消息类型的序列化开始和结束索引
    serialized_start=1696,
    serialized_end=2206,
)
# 将 "model_type" 字段的枚举类型设置为 _TRAINERSPEC_MODELTYPE
_TRAINERSPEC.fields_by_name["model_type"].enum_type = _TRAINERSPEC_MODELTYPE
# 将 _TRAINERSPEC_MODELTYPE 的包含类型设置为 _TRAINERSPEC
_TRAINERSPEC_MODELTYPE.containing_type = _TRAINERSPEC
# 将 "_SELFTESTDATA_SAMPLE" 的包含类型设置为 _SELFTESTDATA
_SELFTESTDATA_SAMPLE.containing_type = _SELFTESTDATA
# 将 "samples" 字段的消息类型设置为 _SELFTESTDATA_SAMPLE
_SELFTESTDATA.fields_by_name["samples"].message_type = _SELFTESTDATA_SAMPLE
# 将 "type" 字段的枚举类型设置为 _MODELPROTO_SENTENCEPIECE_TYPE
_MODELPROTO_SENTENCEPIECE.fields_by_name["type"].enum_type = _MODELPROTO_SENTENCEPIECE_TYPE
# 将 _MODELPROTO_SENTENCEPIECE 的包含类型设置为 _MODELPROTO
_MODELPROTO_SENTENCEPIECE.containing_type = _MODELPROTO
# 将 _MODELPROTO_SENTENCEPIECE_TYPE 的包含类型设置为 _MODELPROTO_SENTENCEPIECE
_MODELPROTO_SENTENCEPIECE_TYPE.containing_type = _MODELPROTO_SENTENCEPIECE
# 将 "pieces" 字段的消息类型设置为 _MODELPROTO_SENTENCEPIECE
_MODELPROTO.fields_by_name["pieces"].message_type = _MODELPROTO_SENTENCEPIECE
# 将 "trainer_spec" 字段的消息类型设置为 _TRAINERSPEC
_MODELPROTO.fields_by_name["trainer_spec"].message_type = _TRAINERSPEC
# 将 "normalizer_spec" 字段的消息类型设置为 _NORMALIZERSPEC
_MODELPROTO.fields_by_name["normalizer_spec"].message_type = _NORMALIZERSPEC
# 将 "self_test_data" 字段的消息类型设置为 _SELFTESTDATA
_MODELPROTO.fields_by_name["self_test_data"].message_type = _SELFTESTDATA
# 将 "denormalizer_spec" 字段的消息类型设置为 _NORMALIZERSPEC
_MODELPROTO.fields_by_name["denormalizer_spec"].message_type = _NORMALIZERSPEC
# 将 DESCRIPTOR 中名为 "TrainerSpec" 的消息类型设置为 _TRAINERSPEC
DESCRIPTOR.message_types_by_name["TrainerSpec"] = _TRAINERSPEC
# 将 DESCRIPTOR 中名为 "NormalizerSpec" 的消息类型设置为 _NORMALIZERSPEC
DESCRIPTOR.message_types_by_name["NormalizerSpec"] = _NORMALIZERSPEC
# 将 DESCRIPTOR 中名为 "SelfTestData" 的消息类型设置为 _SELFTESTDATA
DESCRIPTOR.message_types_by_name["SelfTestData"] = _SELFTESTDATA
# 将 DESCRIPTOR 中名为 "ModelProto" 的消息类型设置为 _MODELPROTO
DESCRIPTOR.message_types_by_name["ModelProto"] = _MODELPROTO
# 将描述符注册到 _sym_db
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

# 创建名为 TrainerSpec 的生成协议消息类型，继承自 _message.Message
TrainerSpec = _reflection.GeneratedProtocolMessageType(
    "TrainerSpec",
    (_message.Message,),
    {
        "DESCRIPTOR": _TRAINERSPEC,
        "__module__": "sentencepiece_model_pb2",
        # @@protoc_insertion_point(class_scope:sentencepiece.TrainerSpec)
    },
)
# 将 TrainerSpec 注册到 _sym_db
_sym_db.RegisterMessage(TrainerSpec)

# 创建名为 NormalizerSpec 的生成协议消息类型，继承自 _message.Message
NormalizerSpec = _reflection.GeneratedProtocolMessageType(
    "NormalizerSpec",
    (_message.Message,),
    {
        "DESCRIPTOR": _NORMALIZERSPEC,
        "__module__": "sentencepiece_model_pb2",
        # @@protoc_insertion_point(class_scope:sentencepiece.NormalizerSpec)
    },
)
# 将 NormalizerSpec 注册到 _sym_db
_sym_db.RegisterMessage(NormalizerSpec)

# 创建名为 SelfTestData 的生成协议消息类型，继承自 _message.Message
SelfTestData = _reflection.GeneratedProtocolMessageType(
    "SelfTestData",
    (_message.Message,),
    {
        # 创建名为 Sample 的生成协议消息类型，继承自 _message.Message
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
# 将 SelfTestData 和 SelfTestData.Sample 注册到 _sym_db
_sym_db.RegisterMessage(SelfTestData)
_sym_db.RegisterMessage(SelfTestData.Sample)

# 创建名为 ModelProto 的生成协议消息类型，继承自 _message.Message
ModelProto = _reflection.GeneratedProtocolMessageType(
    "ModelProto",
    (_message.Message,),
    {
        ...
    {
        # 定义名为 "SentencePiece" 的 GeneratedProtocolMessageType 对象
        "SentencePiece": _reflection.GeneratedProtocolMessageType(
            "SentencePiece",  # 类名为 "SentencePiece"
            (_message.Message,),  # 继承自 _message.Message
            {
                "DESCRIPTOR": _MODELPROTO_SENTENCEPIECE,  # 使用 _MODELPROTO_SENTENCEPIECE 描述符
                "__module__": "sentencepiece_model_pb2",  # 模块名称为 "sentencepiece_model_pb2"
                # @@protoc_insertion_point(class_scope:sentencepiece.ModelProto.SentencePiece)
                # 插入点标记，指明该类的定义位置
            },
        ),
        # 定义名为 "DESCRIPTOR" 的变量，其值为 _MODELPROTO
        "DESCRIPTOR": _MODELPROTO,
        "__module__": "sentencepiece_model_pb2",  # 模块名称为 "sentencepiece_model_pb2"
        # @@protoc_insertion_point(class_scope:sentencepiece.ModelProto)
        # 插入点标记，指明该变量的定义位置
    },
# 注册 ModelProto 消息到 _sym_db
_sym_db.RegisterMessage(ModelProto)

# 注册 ModelProto.SentencePiece 消息到 _sym_db
_sym_db.RegisterMessage(ModelProto.SentencePiece)

# 清除 DESCRIPTOR 对象的 _options 属性，使其为 None
DESCRIPTOR._options = None

# 清除 _TRAINERSPEC 字段中 "mining_sentence_size" 字段的 _options 属性，使其为 None
_TRAINERSPEC.fields_by_name["mining_sentence_size"]._options = None

# 清除 _TRAINERSPEC 字段中 "training_sentence_size" 字段的 _options 属性，使其为 None
_TRAINERSPEC.fields_by_name["training_sentence_size"]._options = None

# 插入点提示，指示这是代码生成工具插入代码的位置
# @@protoc_insertion_point(module_scope)
```