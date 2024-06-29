# `.\numpy\numpy\_core\code_generators\numpy_api.py`

```py
"""Here we define the exported functions, types, etc... which need to be
exported through a global C pointer.

Each dictionary contains name -> index pair.

Whenever you change one index, you break the ABI (and the ABI version number
should be incremented). Whenever you add an item to one of the dict, the API
needs to be updated in both numpy/core/meson.build and by adding an appropriate
entry to cversion.txt (generate the hash via "python cversions.py").

When adding a function, make sure to use the next integer not used as an index
(in case you use an existing index or jump, the build will stop and raise an
exception, so it should hopefully not get unnoticed).

"""

import os  # 导入操作系统接口模块
import importlib.util  # 导入模块导入工具

def get_annotations():
    # Convoluted because we can't import from numpy.distutils
    # (numpy is not yet built)
    # 根据文件路径生成 genapi.py 的完整路径
    genapi_py = os.path.join(os.path.dirname(__file__), 'genapi.py')
    # 根据 genapi.py 文件创建模块规范
    spec = importlib.util.spec_from_file_location('conv_template', genapi_py)
    # 使用模块规范加载模块
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StealRef, mod.MinVersion  # 返回模块中的 StealRef 和 MinVersion 对象

StealRef, MinVersion = get_annotations()  # 调用获取注释函数并分配结果给全局变量

#from code_generators.genapi import StealRef

# index, type
multiarray_global_vars = {
    'NPY_NUMUSERTYPES':             (7, 'int'),           # NPY_NUMUSERTYPES 对应的索引和类型
    'NPY_DEFAULT_ASSIGN_CASTING':   (292, 'NPY_CASTING'),  # NPY_DEFAULT_ASSIGN_CASTING 对应的索引和类型
    'PyDataMem_DefaultHandler':     (306, 'PyObject*'),    # PyDataMem_DefaultHandler 对应的索引和类型
}

multiarray_scalar_bool_values = {
    '_PyArrayScalar_BoolValues':    (9,)  # _PyArrayScalar_BoolValues 对应的索引
}

# index, annotations
# please mark functions that have been checked to not need any annotations
multiarray_types_api = {
    # Slot 1 was never meaningfully used by NumPy
    'PyArray_Type':                     (2,),  # PyArray_Type 对应的索引
    # Internally, PyArrayDescr_Type is a PyArray_DTypeMeta,
    # the following also defines PyArrayDescr_TypeFull (Full appended)
    'PyArrayDescr_Type':                (3, "PyArray_DTypeMeta"),  # PyArrayDescr_Type 对应的索引和注释
    # Unused slot 4, was `PyArrayFlags_Type`
    'PyArrayIter_Type':                 (5,),  # PyArrayIter_Type 对应的索引
    'PyArrayMultiIter_Type':            (6,),  # PyArrayMultiIter_Type 对应的索引
    'PyBoolArrType_Type':               (8,),  # PyBoolArrType_Type 对应的索引
    'PyGenericArrType_Type':            (10,),  # PyGenericArrType_Type 对应的索引
    'PyNumberArrType_Type':             (11,),  # PyNumberArrType_Type 对应的索引
    'PyIntegerArrType_Type':            (12,),  # PyIntegerArrType_Type 对应的索引
    'PySignedIntegerArrType_Type':      (13,),  # PySignedIntegerArrType_Type 对应的索引
    'PyUnsignedIntegerArrType_Type':    (14,),  # PyUnsignedIntegerArrType_Type 对应的索引
    'PyInexactArrType_Type':            (15,),  # PyInexactArrType_Type 对应的索引
    'PyFloatingArrType_Type':           (16,),  # PyFloatingArrType_Type 对应的索引
    'PyComplexFloatingArrType_Type':    (17,),  # PyComplexFloatingArrType_Type 对应的索引
    'PyFlexibleArrType_Type':           (18,),  # PyFlexibleArrType_Type 对应的索引
    'PyCharacterArrType_Type':          (19,),  # PyCharacterArrType_Type 对应的索引
    'PyByteArrType_Type':               (20,),  # PyByteArrType_Type 对应的索引
    'PyShortArrType_Type':              (21,),  # PyShortArrType_Type 对应的索引
    'PyIntArrType_Type':                (22,),  # PyIntArrType_Type 对应的索引
    'PyLongArrType_Type':               (23,),  # PyLongArrType_Type 对应的索引
    'PyLongLongArrType_Type':           (24,),  # PyLongLongArrType_Type 对应的索引
    'PyUByteArrType_Type':              (25,),  # PyUByteArrType_Type 对应的索引
    'PyUShortArrType_Type':             (26,),  # PyUShortArrType_Type 对应的索引
    'PyUIntArrType_Type':               (27,),  # PyUIntArrType_Type 对应的索引
    'PyULongArrType_Type':              (28,),  # PyULongArrType_Type 对应的索引
    'PyULongLongArrType_Type':          (29,),
    'PyFloatArrType_Type':              (30,),
    'PyDoubleArrType_Type':             (31,),
    'PyLongDoubleArrType_Type':         (32,),
    'PyCFloatArrType_Type':             (33,),
    'PyCDoubleArrType_Type':            (34,),
    'PyCLongDoubleArrType_Type':        (35,),
    'PyObjectArrType_Type':             (36,),
    'PyStringArrType_Type':             (37,),
    'PyUnicodeArrType_Type':            (38,),
    'PyVoidArrType_Type':               (39,),


    # 定义一系列的Python数组类型，每个类型对应的唯一标识符用元组表示
    # 这些类型标识符是在Python C API中使用的

    # End 1.5 API
    'PyTimeIntegerArrType_Type':        (214,),
    'PyDatetimeArrType_Type':           (215,),
    'PyTimedeltaArrType_Type':          (216,),
    'PyHalfArrType_Type':               (217,),
    'NpyIter_Type':                     (218,),


    # 定义更多的Python数组类型和它们的唯一标识符
    # 这些标识符属于1.6版本的API

    # End 1.6 API
    # 注意：插槽 320-360 在 `_experimental_dtype_api.h` 中定义，
    #      并在代码生成器之外显式填充，因为元类使得它们难以公开。
    #      （这可能会被重构。）
    # 插槽 366、367、368 是抽象的数据类型
    # End 2.0 API
}
# 在这里有一个单独的右括号，似乎是代码片段的一部分，但没有上下文可以解释其含义
# 因此，这行可能是无意的代码或者漏掉了与之相关的内容

# 定义宏，用于访问 PyArray_API 数组中的索引 6 处的整数值
# 这里假设 PyArray_API 是一个数组或者结构体，存储了一些与 NumPy 多维数组相关的 API 函数或数据

# 定义宏，用于访问 PyArray_API 数组中索引 7 处的 PyTypeObject 结构
# 该结构可能与 NumPy 的布尔数组类型相关

# 定义宏，用于访问 PyArray_API 数组中索引 8 处的 PyBoolScalarObject 结构
# 这可能是一个特定的 NumPy 标量对象，用于处理布尔值

multiarray_funcs_api = {
    '__unused_indices__': (
        [1, 4, 40, 41, 66, 67, 68, 81, 82, 83,
         103, 115, 117, 122, 163, 164, 171, 173, 197,
         201, 202, 208, 219, 220, 221, 222, 223, 278,
         291, 293, 294, 295, 301]
        # range/slots reserved DType classes (see _public_dtype_api_table.h):
        + list(range(320, 361)) + [366, 367, 368]
        ),
    'PyArray_GetNDArrayCVersion':           (0,),
    # 未使用的插槽 40，之前是 `PyArray_SetNumericOps`
    # 未使用的插槽 41，之前是 `PyArray_GetNumericOps`
    'PyArray_INCREF':                       (42,),
    'PyArray_XDECREF':                      (43,),
    # `PyArray_SetStringFunction` 被存根化，未来应该移除
    'PyArray_SetStringFunction':            (44,),
    'PyArray_DescrFromType':                (45,),
    'PyArray_TypeObjectFromType':           (46,),
    'PyArray_Zero':                         (47,),
    'PyArray_One':                          (48,),
    'PyArray_CastToType':                   (49, StealRef(2)),
    'PyArray_CopyInto':                     (50,),
    'PyArray_CopyAnyInto':                  (51,),
    'PyArray_CanCastSafely':                (52,),
    'PyArray_CanCastTo':                    (53,),
    'PyArray_ObjectType':                   (54,),
    'PyArray_DescrFromObject':              (55,),
    'PyArray_ConvertToCommonType':          (56,),
    'PyArray_DescrFromScalar':              (57,),
    'PyArray_DescrFromTypeObject':          (58,),
    'PyArray_Size':                         (59,),
    'PyArray_Scalar':                       (60,),
    'PyArray_FromScalar':                   (61, StealRef(2)),
    'PyArray_ScalarAsCtype':                (62,),
    'PyArray_CastScalarToCtype':            (63,),
    'PyArray_CastScalarDirect':             (64,),
    'PyArray_Pack':                         (65, MinVersion("2.0")),
    # 未使用的插槽 66，之前是 `PyArray_GetCastFunc`
    # 未使用的插槽 67，之前是 `PyArray_FromDims`
    # 未使用的插槽 68，之前是 `PyArray_FromDimsAndDataAndDescr`
    'PyArray_FromAny':                      (69, StealRef(2)),
    'PyArray_EnsureArray':                  (70, StealRef(1)),
    'PyArray_EnsureAnyArray':               (71, StealRef(1)),
    'PyArray_FromFile':                     (72,),
    'PyArray_FromString':                   (73,),
    'PyArray_FromBuffer':                   (74,),
    'PyArray_FromIter':                     (75, StealRef(2)),
    'PyArray_Return':                       (76, StealRef(1)),
    'PyArray_GetField':                     (77, StealRef(2)),
    'PyArray_SetField':                     (78, StealRef(2)),
    'PyArray_Byteswap':                     (79,),
    'PyArray_Resize':                       (80,),
    # 未使用的插槽 81，之前是 `PyArray_MoveInto`
}
# 以上是一个 Python 字典，包含了多个 NumPy 多维数组库中的函数名和它们在 PyArray_API 中的索引
# 每个键值对表示一个函数名及其对应的 API 索引或版本信息，用于在 Python 中操作 NumPy 数组
   `
    # Unused slot 82 was `PyArray_CopyInto` (which replaces `..._CastTo`)
    # 未使用的槽位 82 是 `PyArray_CopyInto`（替代了 `..._CastTo`）

    # Unused slot 82 was `PyArray_CopyAnyInto` (which replaces `..._CastAnyTo`)
    # 未使用的槽位 82 是 `PyArray_CopyAnyInto`（替代了 `..._CastAnyTo`）

    'PyArray_CopyObject':                   (84,),
    # `PyArray_CopyObject` 对应的槽位号为 84

    'PyArray_NewCopy':                      (85,),
    # `PyArray_NewCopy` 对应的槽位号为 85

    'PyArray_ToList':                       (86,),
    # `PyArray_ToList` 对应的槽位号为 86

    'PyArray_ToString':                     (87,),
    # `PyArray_ToString` 对应的槽位号为 87

    'PyArray_ToFile':                       (88,),
    # `PyArray_ToFile` 对应的槽位号为 88

    'PyArray_Dump':                         (89,),
    # `PyArray_Dump` 对应的槽位号为 89

    'PyArray_Dumps':                        (90,),
    # `PyArray_Dumps` 对应的槽位号为 90

    'PyArray_ValidType':                    (91,),
    # `PyArray_ValidType` 对应的槽位号为 91

    'PyArray_UpdateFlags':                  (92,),
    # `PyArray_UpdateFlags` 对应的槽位号为 92

    'PyArray_New':                          (93,),
    # `PyArray_New` 对应的槽位号为 93

    'PyArray_NewFromDescr':                 (94, StealRef(2)),
    # `PyArray_NewFromDescr` 对应的槽位号为 94，并且涉及 StealRef(2) 操作

    'PyArray_DescrNew':                     (95,),
    # `PyArray_DescrNew` 对应的槽位号为 95

    'PyArray_DescrNewFromType':             (96,),
    # `PyArray_DescrNewFromType` 对应的槽位号为 96

    'PyArray_GetPriority':                  (97,),
    # `PyArray_GetPriority` 对应的槽位号为 97

    'PyArray_IterNew':                      (98,),
    # `PyArray_IterNew` 对应的槽位号为 98

    'PyArray_MultiIterNew':                 (99,),
    # `PyArray_MultiIterNew` 对应的槽位号为 99

    'PyArray_PyIntAsInt':                   (100,),
    # `PyArray_PyIntAsInt` 对应的槽位号为 100

    'PyArray_PyIntAsIntp':                  (101,),
    # `PyArray_PyIntAsIntp` 对应的槽位号为 101

    'PyArray_Broadcast':                    (102,),
    # `PyArray_Broadcast` 对应的槽位号为 102

    # Unused slot 103, was `PyArray_FillObjectArray`
    # 未使用的槽位 103 是 `PyArray_FillObjectArray`

    'PyArray_FillWithScalar':               (104,),
    # `PyArray_FillWithScalar` 对应的槽位号为 104

    'PyArray_CheckStrides':                 (105,),
    # `PyArray_CheckStrides` 对应的槽位号为 105

    'PyArray_DescrNewByteorder':            (106,),
    # `PyArray_DescrNewByteorder` 对应的槽位号为 106

    'PyArray_IterAllButAxis':               (107,),
    # `PyArray_IterAllButAxis` 对应的槽位号为 107

    'PyArray_CheckFromAny':                 (108, StealRef(2)),
    # `PyArray_CheckFromAny` 对应的槽位号为 108，并且涉及 StealRef(2) 操作

    'PyArray_FromArray':                    (109, StealRef(2)),
    # `PyArray_FromArray` 对应的槽位号为 109，并且涉及 StealRef(2) 操作

    'PyArray_FromInterface':                (110,),
    # `PyArray_FromInterface` 对应的槽位号为 110

    'PyArray_FromStructInterface':          (111,),
    # `PyArray_FromStructInterface` 对应的槽位号为 111

    'PyArray_FromArrayAttr':                (112,),
    # `PyArray_FromArrayAttr` 对应的槽位号为 112

    'PyArray_ScalarKind':                   (113,),
    # `PyArray_ScalarKind` 对应的槽位号为 113

    'PyArray_CanCoerceScalar':              (114,),
    # `PyArray_CanCoerceScalar` 对应的槽位号为 114

    # Unused slot 115, was `PyArray_NewFlagsObject`
    # 未使用的槽位 115 是 `PyArray_NewFlagsObject`

    'PyArray_CanCastScalar':                (116,),
    # `PyArray_CanCastScalar` 对应的槽位号为 116

    # Unused slot 117, was `PyArray_CompareUCS4`
    # 未使用的槽位 117 是 `PyArray_CompareUCS4`

    'PyArray_RemoveSmallest':               (118,),
    # `PyArray_RemoveSmallest` 对应的槽位号为 118

    'PyArray_ElementStrides':               (119,),
    # `PyArray_ElementStrides` 对应的槽位号为 119

    'PyArray_Item_INCREF':                  (120,),
    # `PyArray_Item_INCREF` 对应的槽位号为 120

    'PyArray_Item_XDECREF':                 (121,),
    # `PyArray_Item_XDECREF` 对应的槽位号为 121

    # Unused slot 122, was `PyArray_FieldNames`
    # 未使用的槽位 122 是 `PyArray_FieldNames`

    'PyArray_Transpose':                    (123,),
    # `PyArray_Transpose` 对应的槽位号为 123

    'PyArray_TakeFrom':                     (124,),
    # `PyArray_TakeFrom` 对应的槽位号为 124

    'PyArray_PutTo':                        (125,),
    # `PyArray_PutTo` 对应的槽位号为 125

    'PyArray_PutMask':                      (126,),
    # `PyArray_PutMask` 对应的槽位号为 126

    'PyArray_Repeat':                       (127,),
    # `PyArray_Repeat` 对应的槽位号为 127

    'PyArray_Choose':                       (128,),
    # `PyArray_Choose` 对应的槽位号为 128

    'PyArray_Sort':                         (129,),
    # `PyArray_Sort` 对应的槽位号为 129

    'PyArray_ArgSort':                      (130,),
    # `PyArray_ArgSort` 对应的槽位号为 130

    'PyArray_SearchSorted':                 (131,),
    # `PyArray_SearchSorted` 对应的槽位号为 131

    'PyArray_ArgMax':                       (132,),
    # `PyArray_ArgMax` 对应的槽位号为 132

    'PyArray_ArgMin':                       (133,),
    # `PyArray_ArgMin` 对应的槽位号为 133

    'PyArray_Reshape':                      (134,),
    # `PyArray_Reshape` 对应的槽位号为 134

    'PyArray_Newshape':                     (135,),
    # `PyArray_Newshape` 对应的槽位号为 135

    'PyArray_Squeeze':                      (136,),
    # `PyArray_Squeeze` 对应的槽位
    'PyArray_SwapAxes':                     (138,),
    'PyArray_Max':                          (139,),
    'PyArray_Min':                          (140,),
    'PyArray_Ptp':                          (141,),
    'PyArray_Mean':                         (142,),
    'PyArray_Trace':                        (143,),
    'PyArray_Diagonal':                     (144,),
    'PyArray_Clip':                         (145,),
    'PyArray_Conjugate':                    (146,),
    'PyArray_Nonzero':                      (147,),
    'PyArray_Std':                          (148,),
    'PyArray_Sum':                          (149,),
    'PyArray_CumSum':                       (150,),
    'PyArray_Prod':                         (151,),
    'PyArray_CumProd':                      (152,),
    'PyArray_All':                          (153,),
    'PyArray_Any':                          (154,),
    'PyArray_Compress':                     (155,),
    'PyArray_Flatten':                      (156,),
    'PyArray_Ravel':                        (157,),
    'PyArray_MultiplyList':                 (158,),
    'PyArray_MultiplyIntList':              (159,),
    'PyArray_GetPtr':                       (160,),
    'PyArray_CompareLists':                 (161,),
    'PyArray_AsCArray':                     (162, StealRef(5)),
    # 未使用的槽位 163, 原为 `PyArray_As1D`
    # 未使用的槽位 164, 原为 `PyArray_As2D`
    'PyArray_Free':                         (165,),
    'PyArray_Converter':                    (166,),
    'PyArray_IntpFromSequence':             (167,),
    'PyArray_Concatenate':                  (168,),
    'PyArray_InnerProduct':                 (169,),
    'PyArray_MatrixProduct':                (170,),
    # 未使用的槽位 171, 原为 `PyArray_CopyAndTranspose`
    'PyArray_Correlate':                    (172,),
    # 未使用的槽位 173, 原为 `PyArray_TypestrConvert`
    'PyArray_DescrConverter':               (174,),
    'PyArray_DescrConverter2':              (175,),
    'PyArray_IntpConverter':                (176,),
    'PyArray_BufferConverter':              (177,),
    'PyArray_AxisConverter':                (178,),
    'PyArray_BoolConverter':                (179,),
    'PyArray_ByteorderConverter':           (180,),
    'PyArray_OrderConverter':               (181,),
    'PyArray_EquivTypes':                   (182,),
    'PyArray_Zeros':                        (183, StealRef(3)),
    'PyArray_Empty':                        (184, StealRef(3)),
    'PyArray_Where':                        (185,),
    'PyArray_Arange':                       (186,),
    'PyArray_ArangeObj':                    (187,),
    'PyArray_SortkindConverter':            (188,),
    'PyArray_LexSort':                      (189,),
    'PyArray_Round':                        (190,),
    'PyArray_EquivTypenums':                (191,),
    'PyArray_RegisterDataType':             (192,),
    'PyArray_RegisterCastFunc':             (193,),
    'PyArray_RegisterCanCast':              (194,),
    'PyArray_InitArrFuncs':                 (195,),
    'PyArray_IntTupleFromIntp':             (196,),
    # 未使用的槽位 197，曾是 `PyArray_TypeNumFromName`
    'PyArray_ClipmodeConverter':            (198,),
    'PyArray_OutputConverter':              (199,),
    'PyArray_BroadcastToShape':             (200,),
    # 未使用的槽位 201，曾是 `_PyArray_SigintHandler`
    # 未使用的槽位 202，曾是 `_PyArray_GetSigintBuf`
    'PyArray_DescrAlignConverter':          (203,),
    'PyArray_DescrAlignConverter2':         (204,),
    'PyArray_SearchsideConverter':          (205,),
    'PyArray_CheckAxis':                    (206,),
    'PyArray_OverflowMultiplyList':         (207,),
    # 未使用的槽位 208，曾是 `PyArray_CompareString`
    'PyArray_MultiIterFromObjects':         (209,),
    'PyArray_GetEndianness':                (210,),
    'PyArray_GetNDArrayCFeatureVersion':    (211,),
    'PyArray_Correlate2':                   (212,),
    'PyArray_NeighborhoodIterNew':          (213,),
    # 结束 1.5 版本的 API
    # 未使用的槽位 219，曾是 `PyArray_SetDatetimeParseFunction`
    # 未使用的槽位 220，曾是 `PyArray_DatetimeToDatetimeStruct`
    # 未使用的槽位 221，曾是 `PyArray_TimedeltaToTimedeltaStruct`
    # 未使用的槽位 222，曾是 `PyArray_DatetimeStructToDatetime`
    # 未使用的槽位 223，曾是 `PyArray_TimedeltaStructToTimedelta`
    # NDIter API
    'NpyIter_New':                          (224,),
    'NpyIter_MultiNew':                     (225,),
    'NpyIter_AdvancedNew':                  (226,),
    'NpyIter_Copy':                         (227,),
    'NpyIter_Deallocate':                   (228,),
    'NpyIter_HasDelayedBufAlloc':           (229,),
    'NpyIter_HasExternalLoop':              (230,),
    'NpyIter_EnableExternalLoop':           (231,),
    'NpyIter_GetInnerStrideArray':          (232,),
    'NpyIter_GetInnerLoopSizePtr':          (233,),
    'NpyIter_Reset':                        (234,),
    'NpyIter_ResetBasePointers':            (235,),
    'NpyIter_ResetToIterIndexRange':        (236,),
    'NpyIter_GetNDim':                      (237,),
    'NpyIter_GetNOp':                       (238,),
    'NpyIter_GetIterNext':                  (239,),
    'NpyIter_GetIterSize':                  (240,),
    'NpyIter_GetIterIndexRange':            (241,),
    'NpyIter_GetIterIndex':                 (242,),
    'NpyIter_GotoIterIndex':                (243,),
    'NpyIter_HasMultiIndex':                (244,),
    'NpyIter_GetShape':                     (245,),
    'NpyIter_GetGetMultiIndex':             (246,),
    'NpyIter_GotoMultiIndex':               (247,),
    'NpyIter_RemoveMultiIndex':             (248,),
    'NpyIter_HasIndex':                     (249,),
    'NpyIter_IsBuffered':                   (250,),
    'NpyIter_IsGrowInner':                  (251,),
    'NpyIter_GetBufferSize':                (252,),
    'NpyIter_GetIndexPtr':                  (253,),
    'NpyIter_GotoIndex':                    (254,),
    'NpyIter_GetDataPtrArray':              (255,),
    'NpyIter_GetDescrArray':                (256,),
    # 返回描述符数组的指针
    'NpyIter_GetOperandArray':              (257,),
    # 返回操作数数组的指针
    'NpyIter_GetIterView':                  (258,),
    # 返回迭代器视图对象的指针
    'NpyIter_GetReadFlags':                 (259,),
    # 返回读取标志位
    'NpyIter_GetWriteFlags':                (260,),
    # 返回写入标志位
    'NpyIter_DebugPrint':                   (261,),
    # 打印调试信息
    'NpyIter_IterationNeedsAPI':            (262,),
    # 返回迭代是否需要 API 的标志位
    'NpyIter_GetInnerFixedStrideArray':     (263,),
    # 返回内部固定步长数组的指针
    'NpyIter_RemoveAxis':                   (264,),
    # 移除轴
    'NpyIter_GetAxisStrideArray':           (265,),
    # 返回轴步长数组的指针
    'NpyIter_RequiresBuffering':            (266,),
    # 返回是否需要缓冲的标志位
    'NpyIter_GetInitialDataPtrArray':       (267,),
    # 返回初始数据指针数组的指针
    'NpyIter_CreateCompatibleStrides':      (268,),
    # 创建兼容步长数组
    #
    'PyArray_CastingConverter':             (269,),
    # 数组类型转换器
    'PyArray_CountNonzero':                 (270,),
    # 计算非零元素个数
    'PyArray_PromoteTypes':                 (271,),
    # 提升数组类型
    'PyArray_MinScalarType':                (272,),
    # 获取最小标量类型
    'PyArray_ResultType':                   (273,),
    # 获取结果数组类型
    'PyArray_CanCastArrayTo':               (274,),
    # 判断数组能否转换为指定类型
    'PyArray_CanCastTypeTo':                (275,),
    # 判断类型能否转换为指定类型
    'PyArray_EinsteinSum':                  (276,),
    # 执行爱因斯坦求和
    'PyArray_NewLikeArray':                 (277, StealRef(3)),
    # 创建与给定数组类似的新数组
    # 未使用的槽位 278，曾是 `PyArray_GetArrayParamsFromObject`
    'PyArray_ConvertClipmodeSequence':      (279,),
    # 转换剪切模式序列
    'PyArray_MatrixProduct2':               (280,),
    # 计算矩阵乘积
    # 结束 1.6 版本 API
    'NpyIter_IsFirstVisit':                 (281,),
    # 检查是否第一次访问
    'PyArray_SetBaseObject':                (282, StealRef(2)),
    # 设置数组的基对象
    'PyArray_CreateSortedStridePerm':       (283,),
    # 创建排序后的步长排列
    'PyArray_RemoveAxesInPlace':            (284,),
    # 就地移除轴
    'PyArray_DebugPrint':                   (285,),
    # 打印数组调试信息
    'PyArray_FailUnlessWriteable':          (286,),
    # 如果不可写则失败
    'PyArray_SetUpdateIfCopyBase':          (287, StealRef(2)),
    # 设置如果是拷贝则更新基对象
    'PyDataMem_NEW':                        (288,),
    # 分配新的内存
    'PyDataMem_FREE':                       (289,),
    # 释放内存
    'PyDataMem_RENEW':                      (290,),
    # 重新分配内存
    # 未使用的槽位 291，曾是 `PyDataMem_SetEventHook`
    # 未使用的槽位 293，曾是 `PyArray_MapIterSwapAxes`
    # 未使用的槽位 294，曾是 `PyArray_MapIterArray`
    # 未使用的槽位 295，曾是 `PyArray_MapIterNext`
    # 结束 1.7 版本 API
    'PyArray_Partition':                    (296,),
    # 对数组执行分区
    'PyArray_ArgPartition':                 (297,),
    # 对数组执行参数分区
    'PyArray_SelectkindConverter':          (298,),
    # 选择种类转换器
    'PyDataMem_NEW_ZEROED':                 (299,),
    # 分配新的零填充内存
    # 结束 1.8 版本 API
    # 结束 1.9 版本 API
    'PyArray_CheckAnyScalarExact':          (300,),
    # 检查任意标量是否精确匹配
    # 结束 1.10 版本 API
    # 未使用的槽位 301，曾是 `PyArray_MapIterArrayCopyIfOverlap`
    # 结束 1.13 版本 API
    'PyArray_ResolveWritebackIfCopy':       (302,),
    # 解决如果是拷贝则写回的问题
    'PyArray_SetWritebackIfCopyBase':       (303,),
    # 设置如果是拷贝则写回的基对象
    # 结束 1.14 版本 API
    'PyDataMem_SetHandler':                 (304, MinVersion("1.22")),
    # 设置内存处理器
    'PyDataMem_GetHandler':                 (305, MinVersion("1.22")),
    # 获取内存处理器
    # 结束 1.22 版本 API
    'NpyDatetime_ConvertDatetime64ToDatetimeStruct': (307, MinVersion("2.0")),
    # 转换 datetime64 到 datetime 结构
    'NpyDatetime_ConvertDatetimeStructToDatetime64': (308, MinVersion("2.0")),
    # 转换 datetime 结构到 datetime64
    'NpyDatetime_ConvertPyDateTimeToDatetimeStruct': (309, MinVersion("2.0")),
    'NpyDatetime_GetDatetimeISO8601StrLen':          (310, MinVersion("2.0")),
    'NpyDatetime_MakeISO8601Datetime':               (311, MinVersion("2.0")),
    'NpyDatetime_ParseISO8601Datetime':              (312, MinVersion("2.0")),
    'NpyString_load':                                (313, MinVersion("2.0")),
    'NpyString_pack':                                (314, MinVersion("2.0")),
    'NpyString_pack_null':                           (315, MinVersion("2.0")),
    'NpyString_acquire_allocator':                   (316, MinVersion("2.0")),
    'NpyString_acquire_allocators':                  (317, MinVersion("2.0")),
    'NpyString_release_allocator':                   (318, MinVersion("2.0")),
    'NpyString_release_allocators':                  (319, MinVersion("2.0")),
    # Slots 320-360 reserved for DType classes (see comment in types)
    'PyArray_GetDefaultDescr':                       (361, MinVersion("2.0")),
    'PyArrayInitDTypeMeta_FromSpec':                 (362, MinVersion("2.0")),
    'PyArray_CommonDType':                           (363, MinVersion("2.0")),
    'PyArray_PromoteDTypeSequence':                  (364, MinVersion("2.0")),
    # The actual public API for this is the inline function
    # `PyDataType_GetArrFuncs` checks for the NumPy runtime version.
    '_PyDataType_GetArrFuncs':                       (365,),
    # End 2.0 API



    # 定义一系列的函数名及其对应的版本要求
    'NpyDatetime_ConvertPyDateTimeToDatetimeStruct': (309, MinVersion("2.0")),
    'NpyDatetime_GetDatetimeISO8601StrLen':          (310, MinVersion("2.0")),
    'NpyDatetime_MakeISO8601Datetime':               (311, MinVersion("2.0")),
    'NpyDatetime_ParseISO8601Datetime':              (312, MinVersion("2.0")),
    'NpyString_load':                                (313, MinVersion("2.0")),
    'NpyString_pack':                                (314, MinVersion("2.0")),
    'NpyString_pack_null':                           (315, MinVersion("2.0")),
    'NpyString_acquire_allocator':                   (316, MinVersion("2.0")),
    'NpyString_acquire_allocators':                  (317, MinVersion("2.0")),
    'NpyString_release_allocator':                   (318, MinVersion("2.0")),
    'NpyString_release_allocators':                  (319, MinVersion("2.0")),
    # 320-360 留作 DType 类使用（详见 types 中的注释）
    'PyArray_GetDefaultDescr':                       (361, MinVersion("2.0")),
    'PyArrayInitDTypeMeta_FromSpec':                 (362, MinVersion("2.0")),
    'PyArray_CommonDType':                           (363, MinVersion("2.0")),
    'PyArray_PromoteDTypeSequence':                  (364, MinVersion("2.0")),
    # 实际公共 API 为内联函数
    # `PyDataType_GetArrFuncs` 检查 NumPy 运行时版本。
    '_PyDataType_GetArrFuncs':                       (365,),
    # 2.0 版本 API 结束
# 该字典定义了一些名为 ufunc_types_api 的键值对，每个键都是一个字符串，对应的值是一个包含整数的元组
ufunc_types_api = {
    'PyUFunc_Type':                             (0,)
}

# 该字典定义了一些名为 ufunc_funcs_api 的键值对，每个键都是一个字符串，对应的值是一个包含整数的元组
ufunc_funcs_api = {
    '__unused_indices__': [3, 25, 26, 29, 32],
    'PyUFunc_FromFuncAndData':                  (1,),
    'PyUFunc_RegisterLoopForType':              (2,),
    # 未使用的槽 3，原先是 `PyUFunc_GenericFunction`
    'PyUFunc_f_f_As_d_d':                       (4,),
    'PyUFunc_d_d':                              (5,),
    'PyUFunc_f_f':                              (6,),
    'PyUFunc_g_g':                              (7,),
    'PyUFunc_F_F_As_D_D':                       (8,),
    'PyUFunc_F_F':                              (9,),
    'PyUFunc_D_D':                              (10,),
    'PyUFunc_G_G':                              (11,),
    'PyUFunc_O_O':                              (12,),
    'PyUFunc_ff_f_As_dd_d':                     (13,),
    'PyUFunc_ff_f':                             (14,),
    'PyUFunc_dd_d':                             (15,),
    'PyUFunc_gg_g':                             (16,),
    'PyUFunc_FF_F_As_DD_D':                     (17,),
    'PyUFunc_DD_D':                             (18,),
    'PyUFunc_FF_F':                             (19,),
    'PyUFunc_GG_G':                             (20,),
    'PyUFunc_OO_O':                             (21,),
    'PyUFunc_O_O_method':                       (22,),
    'PyUFunc_OO_O_method':                      (23,),
    'PyUFunc_On_Om':                            (24,),
    # 未使用的槽 25，原先是 `PyUFunc_GetPyValues`
    # 未使用的槽 26，原先是 `PyUFunc_checkfperr`
    'PyUFunc_clearfperr':                       (27,),
    'PyUFunc_getfperr':                         (28,),
    # 未使用的槽 29，原先是 `PyUFunc_handlefperr`
    'PyUFunc_ReplaceLoopBySignature':           (30,),
    'PyUFunc_FromFuncAndDataAndSignature':      (31,),
    # 未使用的槽 32，原先是 `PyUFunc_SetUsesArraysAsData`
    # 1.5 版本 API 结束
    'PyUFunc_e_e':                              (33,),
    'PyUFunc_e_e_As_f_f':                       (34,),
    'PyUFunc_e_e_As_d_d':                       (35,),
    'PyUFunc_ee_e':                             (36,),
    'PyUFunc_ee_e_As_ff_f':                     (37,),
    'PyUFunc_ee_e_As_dd_d':                     (38,),
    # 1.6 版本 API 结束
    'PyUFunc_DefaultTypeResolver':              (39,),
    'PyUFunc_ValidateCasting':                  (40,),
    # 1.7 版本 API 结束
    'PyUFunc_RegisterLoopForDescr':             (41,),
    # 1.8 版本 API 结束
    'PyUFunc_FromFuncAndDataAndSignatureAndIdentity': (42, MinVersion("1.16")),
    # 1.16 版本 API 结束
    'PyUFunc_AddLoopFromSpec':                       (43, MinVersion("2.0")),
    'PyUFunc_AddPromoter':                           (44, MinVersion("2.0")),
    'PyUFunc_AddWrappingLoop':                       (45, MinVersion("2.0")),
    'PyUFunc_GiveFloatingpointErrors':               (46, MinVersion("2.0")),
}

# 列出了定义 C API 的所有字典的列表
# XXX: 不要改变以下元组的顺序！
# 定义一个元组 `multiarray_api`，包含多维数组的全局变量、标量布尔值、类型 API、函数 API
multiarray_api = (
        multiarray_global_vars,   # 多维数组的全局变量
        multiarray_scalar_bool_values,  # 多维数组的标量布尔值
        multiarray_types_api,  # 多维数组的类型 API
        multiarray_funcs_api,  # 多维数组的函数 API
)

# 定义一个元组 `ufunc_api`，包含通用函数的函数 API、类型 API
ufunc_api = (
        ufunc_funcs_api,   # 通用函数的函数 API
        ufunc_types_api    # 通用函数的类型 API
)

# 将多维数组 API 和通用函数 API 合并成一个新的元组 `full_api`
full_api = multiarray_api + ufunc_api
```