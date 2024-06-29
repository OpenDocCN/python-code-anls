# `D:\src\scipysrc\pandas\pandas\io\sas\sas_constants.py`

```
# 导入未来版本兼容模块，确保代码在Python 2和Python 3中的兼容性
from __future__ import annotations

# 导入类型提示模块，用于静态类型检查
from typing import Final

# 定义一个不可变的字节串常量，用于特定的魔数检查
magic: Final = (
    b"\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\xc2\xea\x81\x60"
    b"\xb3\x14\x11\xcf\xbd\x92\x08\x00"
    b"\x09\xc7\x31\x8c\x18\x1f\x10\x11"
)

# 对齐检查1所需的常量定义
align_1_checker_value: Final = b"3"
align_1_offset: Final = 32
align_1_length: Final = 1
align_1_value: Final = 4

# 对齐检查2所需的常量定义
u64_byte_checker_value: Final = b"3"
align_2_offset: Final = 35
align_2_length: Final = 1
align_2_value: Final = 4

# 偏移量和长度定义，用于确定文件头中不同字段的位置和长度
endianness_offset: Final = 37
endianness_length: Final = 1
platform_offset: Final = 39
platform_length: Final = 1
encoding_offset: Final = 70
encoding_length: Final = 1
dataset_offset: Final = 92
dataset_length: Final = 64
file_type_offset: Final = 156
file_type_length: Final = 8
date_created_offset: Final = 164
date_created_length: Final = 8
date_modified_offset: Final = 172
date_modified_length: Final = 8
header_size_offset: Final = 196
header_size_length: Final = 4
page_size_offset: Final = 200
page_size_length: Final = 4
page_count_offset: Final = 204
page_count_length: Final = 4
sas_release_offset: Final = 216
sas_release_length: Final = 8
sas_server_type_offset: Final = 224
sas_server_type_length: Final = 16
os_version_number_offset: Final = 240
os_version_number_length: Final = 16
os_maker_offset: Final = 256
os_maker_length: Final = 16
os_name_offset: Final = 272
os_name_length: Final = 16

# x86和x64平台下的页面位偏移量和子标题指针长度
page_bit_offset_x86: Final = 16
page_bit_offset_x64: Final = 32
subheader_pointer_length_x86: Final = 12
subheader_pointer_length_x64: Final = 24

# 页面结构相关偏移量和长度定义
page_type_offset: Final = 0
page_type_length: Final = 2
block_count_offset: Final = 2
block_count_length: Final = 2
subheader_count_offset: Final = 4
subheader_count_length: Final = 2

# 页面类型掩码定义
page_type_mask: Final = 0x0F00
# 保留“page_comp_type”位的掩码
page_type_mask2: Final = 0xF000 | page_type_mask

# 页面类型常量定义
page_meta_type: Final = 0x0000
page_data_type: Final = 0x0100
page_mix_type: Final = 0x0200
page_amd_type: Final = 0x0400
page_meta2_type: Final = 0x4000
page_comp_type: Final = 0x9000
page_meta_types: Final = [page_meta_type, page_meta2_type]

# 子标题指针偏移量定义
subheader_pointers_offset: Final = 8

# 压缩子标题的ID和类型定义
truncated_subheader_id: Final = 1
compressed_subheader_id: Final = 4
compressed_subheader_type: Final = 1

# 文本块大小长度定义
text_block_size_length: Final = 2

# 行和列计数偏移量的乘法器定义
row_length_offset_multiplier: Final = 5
row_count_offset_multiplier: Final = 6
col_count_p1_multiplier: Final = 9
col_count_p2_multiplier: Final = 10
row_count_on_mix_page_offset_multiplier: Final = 15

# 列名指针长度定义
column_name_pointer_length: Final = 8

# 列名文本子标题偏移量和长度定义
column_name_text_subheader_offset: Final = 0
column_name_text_subheader_length: Final = 2

# 列名偏移量和长度定义
column_name_offset_offset: Final = 2
column_name_offset_length: Final = 2

# 列名长度偏移量和长度定义
column_name_length_offset: Final = 4
column_name_length_length: Final = 2

# 列数据偏移量和长度定义
column_data_offset_offset: Final = 8
column_data_length_offset: Final = 8
column_data_length_length: Final = 4

# 列类型偏移量和长度定义
column_type_offset: Final = 14
column_type_length: Final = 1

# 列格式文本子标题索引偏移量和长度定义
column_format_text_subheader_index_offset: Final = 22
column_format_text_subheader_index_length: Final = 2
column_format_offset_offset: Final = 24
# 定义列格式偏移量的偏移量，即列格式在文件中的起始位置

column_format_offset_length: Final = 2
# 定义列格式偏移量的长度，即列格式在文件中所占用的字节数

column_format_length_offset: Final = 26
# 定义列格式长度的偏移量，即列格式长度信息在文件中的起始位置

column_format_length_length: Final = 2
# 定义列格式长度的长度，即列格式长度信息在文件中所占用的字节数

column_label_text_subheader_index_offset: Final = 28
# 定义列标签文本子标题索引的偏移量，即列标签文本子标题索引信息在文件中的起始位置

column_label_text_subheader_index_length: Final = 2
# 定义列标签文本子标题索引的长度，即列标签文本子标题索引信息在文件中所占用的字节数

column_label_offset_offset: Final = 30
# 定义列标签偏移量的偏移量，即列标签在文件中的起始位置

column_label_offset_length: Final = 2
# 定义列标签偏移量的长度，即列标签在文件中所占用的字节数

column_label_length_offset: Final = 32
# 定义列标签长度的偏移量，即列标签长度信息在文件中的起始位置

column_label_length_length: Final = 2
# 定义列标签长度的长度，即列标签长度信息在文件中所占用的字节数

rle_compression: Final = b"SASYZCRL"
# 定义 RLE 压缩算法的标识符

rdc_compression: Final = b"SASYZCR2"
# 定义 RDC 压缩算法的标识符

compression_literals: Final = [rle_compression, rdc_compression]
# 创建包含所有压缩算法标识符的列表

# 不完整的编码列表，使用 SAS 命名约定对应 Python 标准编码
# 参考 SAS 编码列表：https://support.sas.com/documentation/onlinedoc/dfdmstudio/2.6/dmpdmsug/Content/dfU_Encodings_SAS.html
# 参考 Python 标准编码文档：https://docs.python.org/3/library/codecs.html#standard-encodings
encoding_names: Final = {
    20: "utf-8",
    29: "latin1",
    30: "latin2",
    31: "latin3",
    32: "latin4",
    33: "cyrillic",
    34: "arabic",
    35: "greek",
    36: "hebrew",
    37: "latin5",
    38: "latin6",
    39: "cp874",
    40: "latin9",
    41: "cp437",
    42: "cp850",
    43: "cp852",
    44: "cp857",
    45: "cp858",
    46: "cp862",
    47: "cp864",
    48: "cp865",
    49: "cp866",
    50: "cp869",
    51: "cp874",
    # 52: "",  # 未找到对应编码
    # 53: "",  # 未找到对应编码
    # 54: "",  # 未找到对应编码
    55: "cp720",
    56: "cp737",
    57: "cp775",
    58: "cp860",
    59: "cp863",
    60: "cp1250",
    61: "cp1251",
    62: "cp1252",
    63: "cp1253",
    64: "cp1254",
    65: "cp1255",
    66: "cp1256",
    67: "cp1257",
    68: "cp1258",
    118: "cp950",
    # 119: "",  # 未找到对应编码
    123: "big5",
    125: "gb2312",
    126: "cp936",
    134: "euc_jp",
    136: "cp932",
    138: "shift_jis",
    140: "euc-kr",
    141: "cp949",
    227: "latin8",
    # 228: "",  # 未找到对应编码
    # 229: ""   # 未找到对应编码
}

class SASIndex:
    row_size_index: Final = 0
    column_size_index: Final = 1
    subheader_counts_index: Final = 2
    column_text_index: Final = 3
    column_name_index: Final = 4
    column_attributes_index: Final = 5
    format_and_label_index: Final = 6
    column_list_index: Final = 7
    data_subheader_index: Final = 8
    # 定义 SAS 数据文件中各种索引的偏移量

subheader_signature_to_index: Final = {
    b"\xf7\xf7\xf7\xf7": SASIndex.row_size_index,
    b"\x00\x00\x00\x00\xf7\xf7\xf7\xf7": SASIndex.row_size_index,
    b"\xf7\xf7\xf7\xf7\x00\x00\x00\x00": SASIndex.row_size_index,
    b"\xf7\xf7\xf7\xf7\xff\xff\xfb\xfe": SASIndex.row_size_index,
    b"\xf6\xf6\xf6\xf6": SASIndex.column_size_index,
    b"\x00\x00\x00\x00\xf6\xf6\xf6\xf6": SASIndex.column_size_index,
    b"\xf6\xf6\xf6\xf6\x00\x00\x00\x00": SASIndex.column_size_index,
    b"\xf6\xf6\xf6\xf6\xff\xff\xfb\xfe": SASIndex.column_size_index,
    b"\x00\xfc\xff\xff": SASIndex.subheader_counts_index,
    b"\xff\xff\xfc\x00": SASIndex.subheader_counts_index,
    b"\x00\xfc\xff\xff\xff\xff\xff\xff": SASIndex.subheader_counts_index,
}
# 定义 SAS 数据文件中各种子标题签名对应的索引
    # 下面的代码是一系列的字节序列和对应的索引名称，它们被存储在 SASIndex 类中的不同属性中。

    # 将特定的字节序列映射到 SASIndex 类中的 subheader_counts_index 属性
    b"\xff\xff\xff\xff\xff\xff\xfc\x00": SASIndex.subheader_counts_index,
    # 将特定的字节序列映射到 SASIndex 类中的 column_text_index 属性
    b"\xfd\xff\xff\xff": SASIndex.column_text_index,
    b"\xff\xff\xff\xfd": SASIndex.column_text_index,
    b"\xfd\xff\xff\xff\xff\xff\xff\xff": SASIndex.column_text_index,
    b"\xff\xff\xff\xff\xff\xff\xff\xfd": SASIndex.column_text_index,
    # 将特定的字节序列映射到 SASIndex 类中的 column_name_index 属性
    b"\xff\xff\xff\xff": SASIndex.column_name_index,
    b"\xff\xff\xff\xff\xff\xff\xff\xff": SASIndex.column_name_index,
    # 将特定的字节序列映射到 SASIndex 类中的 column_attributes_index 属性
    b"\xfc\xff\xff\xff": SASIndex.column_attributes_index,
    b"\xff\xff\xff\xfc": SASIndex.column_attributes_index,
    b"\xfc\xff\xff\xff\xff\xff\xff\xff": SASIndex.column_attributes_index,
    b"\xff\xff\xff\xff\xff\xff\xff\xfc": SASIndex.column_attributes_index,
    # 将特定的字节序列映射到 SASIndex 类中的 format_and_label_index 属性
    b"\xfe\xfb\xff\xff": SASIndex.format_and_label_index,
    b"\xff\xff\xfb\xfe": SASIndex.format_and_label_index,
    b"\xfe\xfb\xff\xff\xff\xff\xff\xff": SASIndex.format_and_label_index,
    b"\xff\xff\xff\xff\xff\xff\xfb\xfe": SASIndex.format_and_label_index,
    # 将特定的字节序列映射到 SASIndex 类中的 column_list_index 属性
    b"\xfe\xff\xff\xff": SASIndex.column_list_index,
    b"\xff\xff\xff\xfe": SASIndex.column_list_index,
    b"\xfe\xff\xff\xff\xff\xff\xff\xff": SASIndex.column_list_index,
    b"\xff\xff\xff\xff\xff\xff\xff\xfe": SASIndex.column_list_index,
# List of frequently used SAS date formats
# SAS 日期格式的常用列表
# 参考链接：
# http://support.sas.com/documentation/cdl/en/etsug/60372/HTML/default/viewer.htm#etsug_intervals_sect009.htm
# https://github.com/epam/parso/blob/master/src/main/java/com/epam/parso/impl/SasFileConstants.java
sas_date_formats: Final = (
    "DATE",        # 日期
    "DAY",         # 日
    "DDMMYY",      # 日月年
    "DOWNAME",     # 星期几的全名
    "JULDAY",      # 一年的第几天
    "JULIAN",      # 朱利安日
    "MMDDYY",      # 月日年
    "MMYY",        # 月年
    "MMYYC",       # 带前导零的月年
    "MMYYD",       # 不带前导零的月年
    "MMYYP",       # 月年带前导加号
    "MMYYS",       # 月年带前导减号
    "MMYYN",       # 数字月年
    "MONNAME",     # 月份的全名
    "MONTH",       # 月份的缩写
    "MONYY",       # 月缩写年
    "QTR",         # 季度
    "QTRR",        # 带前导零的季度
    "NENGO",       # 日本年号
    "WEEKDATE",    # 周日期
    "WEEKDATX",    # 周日期扩展格式
    "WEEKDAY",     # 星期几的缩写
    "WEEKV",       # 周
    "WORDDATE",    # 单词日期
    "WORDDATX",    # 单词日期扩展格式
    "YEAR",        # 年
    "YYMM",        # 年月
    "YYMMC",       # 带前导零的年月
    "YYMMD",       # 不带前导零的年月
    "YYMMP",       # 年月带前导加号
    "YYMMS",       # 年月带前导减号
    "YYMMN",       # 数字年月
    "YYMON",       # 年月的缩写
    "YYMMDD",      # 年月日
    "YYQ",         # 年度季度
    "YYQC",        # 带前导零的年度季度
    "YYQD",        # 不带前导零的年度季度
    "YYQP",        # 年度季度带前导加号
    "YYQS",        # 年度季度带前导减号
    "YYQN",        # 数字年度季度
    "YYQR",        # 年度季度带前导零的罗马数字
    "YYQRC",       # 年度季度带前导零的罗马数字
    "YYQRD",       # 不带前导零的年度季度的罗马数字
    "YYQRP",       # 年度季度带前导加号的罗马数字
    "YYQRS",       # 年度季度带前导减号的罗马数字
    "YYQRN",       # 数字年度季度的罗马数字
    "YYMMDDP",     # 年月日带前导加号
    "YYMMDDC",     # 年月日带前导减号
    "E8601DA",     # ISO 8601 扩展日期格式
    "YYMMDDN",     # 年月日数字格式
    "MMDDYYC",     # 月日年带前导零的格式
    "MMDDYYS",     # 月日年带前导减号的格式
    "MMDDYYD",     # 月日年不带前导零的格式
    "YYMMDDS",     # 年月日带前导减号的格式
    "B8601DA",     # ISO 8601 基本日期格式
    "DDMMYYN",     # 日月年数字格式
    "YYMMDDD",     # 年和一年中的第几天
    "DDMMYYB",     # 日月年带后缀的格式
    "DDMMYYP",     # 日月年带前导加号的格式
    "MMDDYYP",     # 月日年带前导加号的格式
    "YYMMDDB",     # 年月日带后缀的格式
    "MMDDYYN",     # 月日年数字格式
    "DDMMYYC",     # 日月年带前导零的格式
    "DDMMYYD",     # 日月年不带前导零的格式
    "DDMMYYS",     # 日月年带前导减号的格式
    "MINGUO",      # 民国年号
)

# List of frequently used SAS datetime formats
# SAS 日期时间格式的常用列表
sas_datetime_formats: Final = (
    "DATETIME",    # 日期时间
    "DTWKDATX",    # 日期时间周扩展格式
    "B8601DN",     # ISO 8601 基本日期时间数字格式
    "B8601DT",     # ISO 8601 基本日期时间格式
    "B8601DX",     # ISO 8601 基本扩展日期时间格式
    "B8601DZ",     # ISO 8601 基本日期时间时区格式
    "B8601LX",     # ISO 8601 基本扩展日期时间时区格式
    "E8601DN",     # ISO 8601 扩展日期时间数字格式
    "E8601DT",     # ISO 8601 扩展日期时间格式
    "E8601DX",     # ISO 8601 扩展扩展日期时间格式
    "E8601DZ",     # ISO 8601 扩展日期时间时区格式
    "E8601LX",     # ISO 8601 扩展扩展日期时间时区格式
    "DATEAMPM",    # 日期 AM/PM 时间格式
    "DTDATE",      # 日期时间日期格式
    "DTMONYY",     # 日期时间月缩写年格式
    "TOD",         # 一天中的时间
    "MDYAMPM",     # 月日年 AM/PM 时间格式
)
```