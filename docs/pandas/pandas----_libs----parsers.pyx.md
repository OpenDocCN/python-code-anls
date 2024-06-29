# `D:\src\scipysrc\pandas\pandas\_libs\parsers.pyx`

```
# 从 collections 模块导入 defaultdict 类
from collections import defaultdict
# 从 csv 模块导入 QUOTE_MINIMAL、QUOTE_NONE、QUOTE_NONNUMERIC 三个常量
from csv import (
    QUOTE_MINIMAL,
    QUOTE_NONE,
    QUOTE_NONNUMERIC,
)
# 导入警告模块
import warnings

# 从 pandas.util._exceptions 模块导入 find_stack_level 函数
from pandas.util._exceptions import find_stack_level

# 从 pandas 模块导入 StringDtype 类
from pandas import StringDtype
# 从 pandas.core.arrays 模块导入 ArrowExtensionArray、BooleanArray、FloatingArray、IntegerArray 类
from pandas.core.arrays import (
    ArrowExtensionArray,
    BooleanArray,
    FloatingArray,
    IntegerArray,
)

# 使用 cython.cimport 导入 Cython 模块
cimport cython
# 使用 cpython.bytes cimport 导入 PyBytes_AsString 函数
from cpython.bytes cimport PyBytes_AsString
# 从 cpython.exc cimport 导入 PyErr_Fetch、PyErr_Occurred 函数
from cpython.exc cimport (
    PyErr_Fetch,
    PyErr_Occurred,
)
# 使用 cpython.object cimport 导入 PyObject 类
from cpython.object cimport PyObject
# 从 cpython.ref cimport 导入 Py_INCREF、Py_XDECREF 函数
from cpython.ref cimport (
    Py_INCREF,
    Py_XDECREF,
)
# 从 cpython.unicode cimport 导入多个与 Unicode 字符串处理相关的函数
from cpython.unicode cimport (
    PyUnicode_AsUTF8String,
    PyUnicode_Decode,
    PyUnicode_DecodeUTF8,
    PyUnicode_FromString,
)
# 使用 cython cimport 导入 Py_ssize_t 类型
from cython cimport Py_ssize_t
# 使用 libc.stdlib cimport 导入 free 函数
from libc.stdlib cimport free
# 使用 libc.string cimport 导入 strcasecmp、strlen、strncpy 函数
from libc.string cimport (
    strcasecmp,
    strlen,
    strncpy,
)

# 导入 numpy 模块，并使用 np 别名
import numpy as np

# 使用 cimport numpy 导入 numpy 模块的多个类型和函数
cimport numpy as cnp
from numpy cimport (
    float64_t,
    int64_t,
    ndarray,
    uint8_t,
    uint64_t,
)

# 调用 numpy 模块的 import_array() 函数
cnp.import_array()

# 从 pandas._libs cimport util 模块导入 INT64_MAX、INT64_MIN、UINT64_MAX 三个常量
from pandas._libs cimport util
from pandas._libs.util cimport (
    INT64_MAX,
    INT64_MIN,
    UINT64_MAX,
)

# 从 pandas._libs 导入 lib 模块
from pandas._libs import lib

# 从 pandas._libs.khash cimport 导入多个与哈希表相关的函数和类型
from pandas._libs.khash cimport (
    kh_destroy_float64,
    kh_destroy_str,
    kh_destroy_str_starts,
    kh_destroy_strbox,
    kh_exist_str,
    kh_float64_t,
    kh_get_float64,
    kh_get_str,
    kh_get_str_starts_item,
    kh_get_strbox,
    kh_init_float64,
    kh_init_str,
    kh_init_str_starts,
    kh_init_strbox,
    kh_put_float64,
    kh_put_str,
    kh_put_str_starts_item,
    kh_put_strbox,
    kh_resize_float64,
    kh_resize_str_starts,
    kh_str_starts_t,
    kh_str_t,
    kh_strbox_t,
    khiter_t,
)

# 从 pandas.errors 模块导入 EmptyDataError、ParserError、ParserWarning 三个异常类
from pandas.errors import (
    EmptyDataError,
    ParserError,
    ParserWarning,
)

# 从 pandas.core.dtypes.dtypes 模块导入 CategoricalDtype、ExtensionDtype 两个数据类型类
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
# 从 pandas.core.dtypes.inference 模块导入 is_dict_like 函数
from pandas.core.dtypes.inference import is_dict_like

# 从 pandas.core.arrays.boolean 模块导入 BooleanDtype 类
from pandas.core.arrays.boolean import BooleanDtype

# 使用 cdef 定义几个 C 语言风格的常量
cdef:
    float64_t INF = <float64_t>np.inf
    float64_t NEGINF = -INF
    int64_t DEFAULT_CHUNKSIZE = 256 * 1024

# Python 风格的常量定义
DEFAULT_BUFFER_HEURISTIC = 2 ** 20

# 使用 cdef extern from 导入外部 C 语言头文件 "pandas/portable.h"
cdef extern from "pandas/portable.h":
    # 这段注释是为了说明为什么需要导入这个头文件

# 使用 cdef extern from 导入外部 C 语言头文件 "pandas/parser/tokenizer.h"
cdef extern from "pandas/parser/tokenizer.h":

    # 定义枚举类型 ParserState
    ctypedef enum ParserState:
        START_RECORD
        START_FIELD
        ESCAPED_CHAR
        IN_FIELD
        IN_QUOTED_FIELD
        ESCAPE_IN_QUOTED_FIELD
        QUOTE_IN_QUOTED_FIELD
        EAT_CRNL
        EAT_CRNL_NOP
        EAT_WHITESPACE
        EAT_COMMENT
        EAT_LINE_COMMENT
        WHITESPACE_LINE
        SKIP_LINE
        FINISHED

    # 定义全局枚举类型 ERROR_OVERFLOW
    enum: ERROR_OVERFLOW

    # 定义枚举类型 BadLineHandleMethod
    ctypedef enum BadLineHandleMethod:
        ERROR,
        WARN,
        SKIP
    # 定义一个函数指针类型，用于处理输入/输出的回调函数
    ctypedef char* (*io_callback)(void *src, size_t nbytes, size_t *bytes_read,
                                  int *status, const char *encoding_errors)
    
    # 定义一个函数指针类型，用于清理资源的回调函数
    ctypedef void (*io_cleanup)(void *src)
    
    # 定义一个结构体 parser_t，用于解析器的配置和状态管理
    ctypedef struct parser_t:
        void *source              # 数据源的指针
        io_callback cb_io         # 处理输入/输出的回调函数指针
        io_cleanup cb_cleanup     # 清理资源的回调函数指针

        int64_t chunksize         # 每个数据块的字节数
        char *data                # 指向待处理数据的指针
        int64_t datalen           # 可用数据长度
        int64_t datapos           # 当前数据处理位置

        char *stream              # 写出标记化数据的目标流
        uint64_t stream_len       # 目标流中当前数据长度
        uint64_t stream_cap       # 目标流的容量

        char **words              # 存储单词的数组（可能是不规则的矩阵）
        int64_t *word_starts     # 单词在流中的起始位置
        uint64_t words_len        # 单词数组当前长度
        uint64_t words_cap        # 单词数组的容量
        uint64_t max_words_cap    # 最大单词容量

        char *pword_start         # 当前字段在流中的起始位置
        int64_t word_start        # 当前字段的起始位置

        int64_t *line_start       # 行的起始位置在单词数组中的索引
        int64_t *line_fields      # 每行包含的字段数
        uint64_t lines            # 观察到的行数
        uint64_t file_lines       # 观察到的行数（包括无效/跳过的）
        uint64_t lines_cap        # 行数数组的容量

        ParserState state         # 解析器的状态
        int doublequote           # 是否使用 "" 表示 "
        char delimiter            # 字段分隔符
        int delim_whitespace      # 是否消耗制表符/空格
        char quotechar            # 引用字符
        char escapechar           # 转义字符
        char lineterminator       # 行终止符
        int skipinitialspace      # 是否忽略分隔符后的空格
        int quoting               # 引用的样式

        char commentchar          # 注释字符
        int allow_embedded_newline # 是否允许嵌入的换行符

        int usecols               # 使用的列数

        Py_ssize_t expected_fields # 预期字段数
        BadLineHandleMethod on_bad_lines # 处理错误行的方法

        char decimal              # 浮点数选项
        char sci                  # 科学计数法选项

        char thousands            # 千位分隔符（逗号、句号等）

        int header                # 是否有表头行
        int64_t header_start      # 表头行的起始位置
        uint64_t header_end       # 表头行的结束位置

        void *skipset             # 要跳过的集合（指针）
        PyObject *skipfunc        # 要跳过的函数对象
        int64_t skip_first_N_rows # 要跳过的前 N 行
        int64_t skipfooter        # 要跳过的末尾行数

        double (*double_converter)(const char *, char **,
                                   char, char, char,
                                   int, int *, int *) noexcept nogil
        # 双精度转换器函数指针，取决于是否需要 GIL

        char *warn_msg            # 警告消息
        char *error_msg           # 错误消息

        int64_t skip_empty_lines  # 跳过空行数
    # 定义一个C语言的结构体`uint_state`，包含三个整型成员变量：`seen_sint`、`seen_uint`、`seen_null`
    ctypedef struct uint_state:
        int seen_sint
        int seen_uint
        int seen_null
    
    # 定义一个C语言的函数原型`COLITER_NEXT`，该函数接受两个参数`coliter_t`和`const char *`，并使用`nogil`来指示函数在调用期间不会释放全局解释器锁（GIL）
    void COLITER_NEXT(coliter_t, const char *) nogil
# 从指定路径导入 C 库中的函数和结构体声明
cdef extern from "pandas/parser/pd_parser.h":
    # 创建一个新的读取源，使用 Python 对象作为参数，返回 void 指针，可能为 NULL
    void *new_rd_source(object obj) except NULL

    # 删除读取源，接收 void 指针作为参数
    void del_rd_source(void *src)

    # 从读取源中读取指定字节数的数据到缓冲区，返回 char 指针
    char* buffer_rd_bytes(void *source, size_t nbytes,
                          size_t *bytes_read, int *status, const char *encoding_errors)

    # 初始化 uint_state 结构体，接收 uint_state 指针作为参数
    void uint_state_init(uint_state *self)
    # 检查 uint_state 结构体中的 uint64_t 是否发生冲突，接收 uint_state 指针作为参数，返回 int
    int uint64_conflict(uint_state *self)

    # 初始化 coliter_t 结构体，接收 coliter_t 指针、parser_t 指针、两个 int64_t 类型整数作为参数
    void coliter_setup(coliter_t *it, parser_t *parser,
                       int64_t i, int64_t start) nogil
    # 获取 coliter_t 结构体的下一个元素，接收 coliter_t 结构体和 const char 指针作为参数，不使用 GIL
    void COLITER_NEXT(coliter_t, const char *) nogil

    # 创建一个新的 parser_t 结构体，返回 parser_t 指针
    parser_t* parser_new()

    # 初始化 parser_t 结构体，接收 parser_t 指针作为参数，不使用 GIL
    int parser_init(parser_t *self) nogil
    # 释放 parser_t 结构体，接收 parser_t 指针作为参数，不使用 GIL
    void parser_free(parser_t *self) nogil
    # 删除 parser_t 结构体，接收 parser_t 指针作为参数，不使用 GIL
    void parser_del(parser_t *self) nogil
    # 向 parser_t 结构体添加要跳过的行数，接收 parser_t 指针和 int64_t 类型整数作为参数，不使用 GIL
    int parser_add_skiprow(parser_t *self, int64_t row)

    # 设置 parser_t 结构体要跳过的前 nrows 行，接收 parser_t 指针和 int64_t 类型整数作为参数
    void parser_set_skipfirstnrows(parser_t *self, int64_t nrows)

    # 设置 parser_t 结构体的默认选项，接收 parser_t 指针作为参数
    void parser_set_default_options(parser_t *self)

    # 消耗 parser_t 结构体中的 nrows 行数据，接收 parser_t 指针和 size_t 类型整数作为参数
    int parser_consume_rows(parser_t *self, size_t nrows)

    # 裁剪 parser_t 结构体中的缓冲区，接收 parser_t 指针作为参数
    int parser_trim_buffers(parser_t *self)

    # 对 parser_t 结构体中的所有行进行标记化，接收 parser_t 指针和 const char 指针作为参数，不使用 GIL
    int tokenize_all_rows(parser_t *self, const char *encoding_errors) nogil
    # 对 parser_t 结构体中的 nrows 行进行标记化，接收 parser_t 指针、size_t 类型整数和 const char 指针作为参数，不使用 GIL
    int tokenize_nrows(parser_t *self, size_t nrows, const char *encoding_errors) nogil

    # 将字符串转换为 int64_t 类型整数，接收 char 指针、int64_t 类型整数、两个 int 指针和 char 类型字符作为参数，不使用 GIL
    int64_t str_to_int64(char *p_item, int64_t int_min,
                         int64_t int_max, int *error, char tsep) nogil
    # 将字符串转换为 uint64_t 类型整数，接收 uint_state 指针、char 指针、int64_t 类型整数、uint64_t 类型整数、两个 int 指针和 char 类型字符作为参数，不使用 GIL
    uint64_t str_to_uint64(uint_state *state, char *p_item, int64_t int_max,
                           uint64_t uint_max, int *error, char tsep) nogil

    # 将字符串转换为 double 类型浮点数，接收 const char 指针、char 指针、四个 char 类型字符、四个 int 指针作为参数，不使用 GIL
    double xstrtod(const char *p, char **q, char decimal,
                   char sci, char tsep, int skip_trailing,
                   int *error, int *maybe_int) nogil
    # 将字符串转换为精确的 double 类型浮点数，接收 const char 指针、char 指针、四个 char 类型字符、四个 int 指针作为参数，不使用 GIL
    double precise_xstrtod(const char *p, char **q, char decimal,
                           char sci, char tsep, int skip_trailing,
                           int *error, int *maybe_int) nogil
    # 将字符串进行往返转换，接收 const char 指针、char 指针、四个 char 类型字符、四个 int 指针作为参数，不使用 GIL
    double round_trip(const char *p, char **q, char decimal,
                      char sci, char tsep, int skip_trailing,
                      int *error, int *maybe_int) nogil

    # 将字符串转换为布尔值，接收 const char 指针和 uint8_t 指针作为参数，不使用 GIL
    int to_boolean(const char *item, uint8_t *val) nogil

    # 导入 PandasParser 模块
    void PandasParser_IMPORT()

# 调用 PandasParser_IMPORT 函数，导入 PandasParser 模块
PandasParser_IMPORT

# 当被分配为函数时，cdef extern 声明似乎会留下未定义的符号
# 定义 xstrtod_wrapper 函数，调用 xstrtod 函数，接收 const char 指针、char 指针、四个 char 类型字符、四个 int 指针作为参数，不抛出异常，不使用 GIL
cdef double xstrtod_wrapper(const char *p, char **q, char decimal,
                            char sci, char tsep, int skip_trailing,
                            int *error, int *maybe_int) noexcept nogil:
    return xstrtod(p, q, decimal, sci, tsep, skip_trailing, error, maybe_int)

# 定义 precise_xstrtod_wrapper 函数，调用 precise_xstrtod 函数，接收 const char 指针、char 指针、四个 char 类型字符、四个 int 指针作为参数，不抛出异常，不使用 GIL
cdef double precise_xstrtod_wrapper(const char *p, char **q, char decimal,
                                    char sci, char tsep, int skip_trailing,
                                    int *error, int *maybe_int) noexcept nogil:
    return precise_xstrtod(p, q, decimal, sci, tsep, skip_trailing, error, maybe_int)
cdef double round_trip_wrapper(const char *p, char **q, char decimal,
                               char sci, char tsep, int skip_trailing,
                               int *error, int *maybe_int) noexcept nogil:
    # 调用 C 函数 `round_trip` 进行数据的来回转换，处理各种转换选项
    return round_trip(p, q, decimal, sci, tsep, skip_trailing, error, maybe_int)


cdef char* buffer_rd_bytes_wrapper(void *source, size_t nbytes,
                                   size_t *bytes_read, int *status,
                                   const char *encoding_errors) noexcept:
    # 调用 C 函数 `buffer_rd_bytes` 读取指定字节数的数据到缓冲区，返回读取的字节数
    return buffer_rd_bytes(source, nbytes, bytes_read, status, encoding_errors)

cdef void del_rd_source_wrapper(void *src) noexcept:
    # 调用 C 函数 `del_rd_source` 释放读取源的资源
    del_rd_source(src)


cdef class TextReader:
    """
    用于读取文本数据的类。

    # source: StringIO or file object

    ..versionchange:: 1.2.0
        removed 'compression', 'memory_map', and 'encoding' argument.
        These arguments are outsourced to CParserWrapper.
        'source' has to be a file handle.
    """

    cdef:
        parser_t *parser  # C 解析器对象指针
        object na_fvalues  # 用于表示缺失值的特殊值
        object true_values, false_values  # 布尔值的真和假的可能取值
        object handle  # 文件句柄或类似对象
        object orig_header  # 原始的文件头信息
        bint na_filter, keep_default_na, has_usecols, has_mi_columns  # 布尔值，表示是否进行特定的数据处理
        bint allow_leading_cols  # 布尔值，表示是否允许额外的列
        uint64_t parser_start  # 解析器开始工作的时间戳，初始化后修改
        const char *encoding_errors  # 编码错误处理选项
        kh_str_starts_t *false_set  # 哈希集合，存储可能的假值的前缀
        kh_str_starts_t *true_set  # 哈希集合，存储可能的真值的前缀
        int64_t buffer_lines, skipfooter  # 缓冲区行数，尾部跳过的行数
        list dtype_cast_order  # 数据类型转换的顺序
        list names   # 列名列表，可能为 None
        set noconvert  # 不转换的列的集合，存储索引

    cdef public:
        int64_t leading_cols, table_width  # 开始列数，表格宽度
        object delimiter  # 分隔符，可以是字节或字符串
        object converters  # 数据转换器
        object na_values  # 表示缺失值的可能取值
        list header  # 头部信息，表示列索引列表的列表
        object index_col  # 索引列
        object skiprows  # 要跳过的行数或行号
        object dtype  # 数据类型
        object usecols  # 要解析的列
        set unnamed_cols  # 未命名列的集合，存储列名
        str dtype_backend  # 数据类型后端

    def __init__(self, *args, **kwargs):
        # 初始化方法，暂时未实现具体逻辑
        pass

    def __dealloc__(self):
        # 析构方法，关闭对象相关的资源
        _close(self)  # 调用 `_close` 函数关闭对象
        parser_del(self.parser)  # 调用 `parser_del` 函数释放解析器对象

    def close(self):
        # 关闭对象的公共方法，调用内部 `_close` 函数关闭对象
        _close(self)
    # 设置引用字符和引用方式
    def _set_quoting(self, quote_char: str | bytes | None, quoting: int):
        # 检查引用方式是否为整数，如果不是则引发类型错误
        if not isinstance(quoting, int):
            raise TypeError('"quoting" must be an integer')

        # 检查引用方式值是否在允许范围内
        if not QUOTE_MINIMAL <= quoting <= QUOTE_NONE:
            raise TypeError('bad "quoting" value')

        # 检查引用字符类型是否为字符串或字节，并且不能为空
        if not isinstance(quote_char, (str, bytes)) and quote_char is not None:
            dtype = type(quote_char).__name__
            raise TypeError(f'"quotechar" must be string, not {dtype}')

        # 如果引用字符为空或者为""，并且引用方式不为QUOTE_NONE，则引发类型错误
        if quote_char is None or quote_char == "":
            if quoting != QUOTE_NONE:
                raise TypeError("quotechar must be set if quoting enabled")
            self.parser.quoting = quoting
            self.parser.quotechar = -1
        elif len(quote_char) > 1:  # 处理长度大于1的引用字符
            raise TypeError('"quotechar" must be a 1-character string')
        else:
            # 设置引用方式和引用字符的 ASCII 值
            self.parser.quoting = quoting
            self.parser.quotechar = <char>ord(quote_char)

    # 创建跳过行的集合
    cdef _make_skiprow_set(self):
        # 如果跳过行数是整数对象，则调用解析器设置跳过的第一行
        if util.is_integer_object(self.skiprows):
            parser_set_skipfirstnrows(self.parser, self.skiprows)
        # 如果跳过行数不是可调用对象，则逐行添加跳过行数
        elif not callable(self.skiprows):
            for i in self.skiprows:
                parser_add_skiprow(self.parser, i)
        else:
            # 否则，将跳过行数设置为解析器的跳过函数
            self.parser.skipfunc = <PyObject *>self.skiprows

    # 设置解析器的数据源
    cdef _setup_parser_source(self, source):
        cdef:
            void *ptr

        # 创建一个新的数据源，并将其指针赋给解析器的数据源
        ptr = new_rd_source(source)
        self.parser.source = ptr
        self.parser.cb_io = buffer_rd_bytes_wrapper
        self.parser.cb_cleanup = del_rd_source_wrapper

    # 读取数据的方法，返回以整数索引为键、"ArrayLike"对象为值的字典
    def read(self, rows: int | None = None) -> dict[int, "ArrayLike"]:
        """
        rows=None --> read all rows
        """
        # 不考虑内存使用，读取指定行数的数据
        columns = self._read_rows(rows, 1)

        return columns
    def read_low_memory(self, rows: int | None)-> list[dict[int, "ArrayLike"]]:
        """
        rows=None --> read all rows
        """
        # Conserve intermediate space
        # Caller is responsible for concatenating chunks,
        #  see c_parser_wrapper._concatenate_chunks
        # 定义变量，记录已读取的行数和存储数据块的列表
        cdef:
            size_t rows_read = 0
            list chunks = []

        # 如果参数 rows 为 None，则循环读取所有行直至结束
        if rows is None:
            while True:
                try:
                    # 调用 _read_rows 方法读取数据块
                    chunk = self._read_rows(self.buffer_lines, 0)
                    # 如果读取的数据块为空，跳出循环
                    if len(chunk) == 0:
                        break
                except StopIteration:
                    break
                else:
                    # 将读取的数据块添加到 chunks 列表中
                    chunks.append(chunk)
        # 如果指定了具体的行数 rows，则循环读取至指定行数或数据结束
        else:
            while rows_read < rows:
                try:
                    # 计算本次循环要读取的行数 crows
                    crows = min(self.buffer_lines, rows - rows_read)
                    # 调用 _read_rows 方法读取数据块
                    chunk = self._read_rows(crows, 0)
                    # 如果读取的数据块为空，跳出循环
                    if len(chunk) == 0:
                        break

                    # 更新已读取的行数
                    rows_read += len(list(chunk.values())[0])
                except StopIteration:
                    break
                else:
                    # 将读取的数据块添加到 chunks 列表中
                    chunks.append(chunk)

        # 清空解析器的缓冲区
        parser_trim_buffers(self.parser)

        # 如果 chunks 列表为空，抛出 StopIteration 异常
        if len(chunks) == 0:
            raise StopIteration

        # 返回所有数据块组成的列表
        return chunks

    # 使用 C 语言定义的方法，用于处理 nrows 行数据的分词操作
    cdef _tokenize_rows(self, size_t nrows):
        cdef:
            int status

        # 在没有全局解释器锁的情况下执行，调用底层的 C 函数进行行数据的分词
        with nogil:
            status = tokenize_nrows(self.parser, nrows, self.encoding_errors)

        # 检查分词操作的状态
        self._check_tokenize_status(status)

    # 检查分词操作的状态，并进行相应的处理
    cdef _check_tokenize_status(self, int status):
        # 如果解析器有警告信息，解码并发出警告
        if self.parser.warn_msg != NULL:
            warnings.warn(
                PyUnicode_DecodeUTF8(
                    self.parser.warn_msg,
                    strlen(self.parser.warn_msg),
                    self.encoding_errors
                ),
                ParserWarning,
                stacklevel=find_stack_level()
            )
            # 释放警告信息占用的内存并重置为 NULL
            free(self.parser.warn_msg)
            self.parser.warn_msg = NULL

        # 如果分词操作的状态小于 0，抛出解析器错误异常
        if status < 0:
            raise_parser_error("Error tokenizing data", self.parser)
    # 定义一个 C 函数，用于从数据流中读取行，可选地进行修剪操作
    cdef _read_rows(self, rows, bint trim):
        # 声明变量：缓冲行数和读取行数
        cdef:
            int64_t buffered_lines  # 缓冲区中已有的行数
            int64_t irows  # 要读取的行数

        # 如果指定了要读取的行数
        if rows is not None:
            # 将要读取的行数赋值给 irows
            irows = rows
            # 计算缓冲区中还剩余多少行未处理
            buffered_lines = self.parser.lines - self.parser_start
            # 如果缓冲区中未处理的行数少于要读取的行数，需重新进行分词
            if buffered_lines < irows:
                self._tokenize_rows(irows - buffered_lines)

            # 如果设置了 skipfooter 参数大于 0，则抛出数值错误异常
            if self.skipfooter > 0:
                raise ValueError("skipfooter can only be used to read "
                                 "the whole file")
        else:
            # 否则，在没有全局解释器锁的情况下，调用 C 函数 tokenize_all_rows 解析所有行
            with nogil:
                status = tokenize_all_rows(self.parser, self.encoding_errors)

            # 检查解析状态，确保成功
            self._check_tokenize_status(status)

        # 如果已经处理的行数超过了总行数，则抛出迭代停止异常
        if self.parser_start >= self.parser.lines:
            raise StopIteration

        # 将列数据转换成指定格式，并返回
        columns = self._convert_column_data(rows)

        # 如果成功转换了列数据
        if len(columns) > 0:
            # 获取实际读取的行数
            rows_read = len(list(columns.values())[0])
            # 从解析器中消耗掉这些行数
            parser_consume_rows(self.parser, rows_read)
            # 如果需要进行修剪操作，则调用解析器的修剪缓冲区函数
            if trim:
                parser_trim_buffers(self.parser)
            # 更新起始行号
            self.parser_start -= rows_read

        # 返回处理后的列数据
        return columns

    # 设置不进行转换的列索引
    def set_noconvert(self, i: int) -> None:
        self.noconvert.add(i)

    # 移除不进行转换的列索引
    def remove_noconvert(self, i: int) -> None:
        self.noconvert.remove(i)

    # 定义一个 C 函数，用于将指定范围内的字符串转换为 UTF-8 编码的对象
    # 返回值为转换后的对象及其结束位置
    # 参数说明：
    #   i: 行索引
    #   start: 起始位置
    #   end: 结束位置
    #   na_filter: 是否进行 NA 过滤
    #   na_hashset: NA 值的哈希集合
    cdef _string_convert(self, Py_ssize_t i, int64_t start, int64_t end,
                         bint na_filter, kh_str_starts_t *na_hashset):

        # 调用 C 函数 _string_box_utf8 对数据进行 UTF-8 编码转换，并返回结果
        return _string_box_utf8(self.parser, i, start, end, na_filter,
                                na_hashset, self.encoding_errors)

    # 获取指定索引位置的转换器函数
    def _get_converter(self, i: int, name):
        # 如果转换器列表为空，直接返回 None
        if self.converters is None:
            return None

        # 如果提供了列名，并且该列名在转换器字典中存在，则返回对应的转换器函数
        if name is not None and name in self.converters:
            return self.converters[name]

        # 否则，返回对应位置的转换器函数（如果存在）
        # 注：这里使用 get 方法，如果不存在对应位置的转换器，则返回 None
        #     这也包括 name 为 None 或者 converters 字典中没有 name 对应的键的情况
        #     返回的是位置 i 对应的转换器函数
        #     如果没有找到 name 对应的键，get 方法也返回 None
        #     如果 converters 为 None，则 get 方法同样返回 None
        #     综上所述，这里的逻辑是返回第 i 位置的转换器函数，如果不存在则返回 None
        #     该方法用于根据位置或名称获取列的转换器函数
        # Converter for position, if any
        return self.converters.get(i)
    # 获取指定位置和列名的缺失值定义列表和强制类型转换后的集合
    cdef _get_na_list(self, Py_ssize_t i, name):
        # 如果没有提供缺失值定义，则返回空值和空集合
        if self.na_values is None:
            return None, set()

        # 如果缺失值定义是一个字典
        if isinstance(self.na_values, dict):
            key = None
            values = None

            # 根据列名或索引查找缺失值定义的键
            if name is not None and name in self.na_values:
                key = name
            elif i in self.na_values:
                key = i
            else:  # 如果未提供该列的缺失值定义
                if self.keep_default_na:
                    return _NA_VALUES, set()  # 返回默认的缺失值列表和空集合

                return list(), set()  # 返回空列表和空集合

            # 获取缺失值定义及其强制类型转换后的集合
            values = self.na_values[key]
            if values is not None and not isinstance(values, list):
                values = list(values)

            fvalues = self.na_fvalues[key]
            if fvalues is not None and not isinstance(fvalues, set):
                fvalues = set(fvalues)

            return _ensure_encoded(values), fvalues  # 返回编码后的缺失值列表和集合
        else:
            # 如果缺失值定义不是字典，则将其转换为列表和集合并返回
            if not isinstance(self.na_values, list):
                self.na_values = list(self.na_values)
            if not isinstance(self.na_fvalues, set):
                self.na_fvalues = set(self.na_fvalues)

            return _ensure_encoded(self.na_values), self.na_fvalues  # 返回编码后的缺失值列表和集合

    # 释放使用字符串起始表的内存
    cdef _free_na_set(self, kh_str_starts_t *table):
        kh_destroy_str_starts(table)

    # 获取列名
    cdef _get_column_name(self, Py_ssize_t i, Py_ssize_t nused):
        cdef int64_t j
        # 如果使用指定列和列名，并且名称列表不为空
        if self.has_usecols and self.names is not None:
            if (not callable(self.usecols) and
                    len(self.names) == len(self.usecols)):
                return self.names[nused]  # 返回指定列的名称
            else:
                return self.names[i - self.leading_cols]  # 返回指定索引的名称
        else:
            # 如果存在表头信息
            if self.header is not None:
                j = i - self.leading_cols
                # 如果列数多于表头数量，则生成额外的虚构表头
                # 应该是字符串，以避免与可调用对象在usecols中的问题
                if j >= len(self.header[0]):
                    return str(j)  # 返回虚构的列名字符串
                elif self.has_mi_columns:
                    return tuple(header_row[j] for header_row in self.header)  # 返回多重索引列的名称元组
                else:
                    return self.header[0][j]  # 返回普通表头的列名
            else:
                return None  # 如果没有表头信息，则返回空
# 将 TextReader.__dealloc__ 和 TextReader.close 公共的代码抽取出来
# 由于在 __dealloc__ 中调用 self.close() 会导致类属性查找，违反最佳实践
# https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#finalization-method-dealloc
cdef _close(TextReader reader):
    # 释放解析器所分配的内存
    parser_free(reader.parser)
    # 如果 reader.true_set 不为空，则销毁其哈希表
    if reader.true_set:
        kh_destroy_str_starts(reader.true_set)
        reader.true_set = NULL
    # 如果 reader.false_set 不为空，则销毁其哈希表
    if reader.false_set:
        kh_destroy_str_starts(reader.false_set)
        reader.false_set = NULL


cdef:
    # 包含真值的字节字符串列表
    object _true_values = [b"True", b"TRUE", b"true"]
    # 包含假值的字节字符串列表
    object _false_values = [b"False", b"FALSE", b"false"]


def _ensure_encoded(list lst):
    # 确保输入列表中的每个元素被编码为 UTF-8 字符串并返回
    cdef:
        list result = []
    for x in lst:
        if isinstance(x, str):
            x = PyUnicode_AsUTF8String(x)
        elif not isinstance(x, bytes):
            x = str(x).encode("utf-8")

        result.append(x)
    return result


# 常见的 NA 值集合
# 不再排除无穷大表示
# '1.#INF','-1.#INF', '1.#INF000000',
STR_NA_VALUES = {
    "-1.#IND",
    "1.#QNAN",
    "1.#IND",
    "-1.#QNAN",
    "#N/A N/A",
    "#N/A",
    "N/A",
    "n/a",
    "NA",
    "<NA>",
    "#NA",
    "NULL",
    "null",
    "NaN",
    "-NaN",
    "nan",
    "-nan",
    "",
    "None",
}
# 对 STR_NA_VALUES 中的每个字符串进行编码处理，确保转换为 UTF-8 字符串列表
_NA_VALUES = _ensure_encoded(list(STR_NA_VALUES))


def _maybe_upcast(
    arr, use_dtype_backend: bool = False, dtype_backend: str = "numpy"
):
    """Sets nullable dtypes or upcasts if nans are present.

    Upcast, if use_dtype_backend is false and nans are present so that the
    current dtype can not hold the na value. We use nullable dtypes if the
    flag is true for every array.

    Parameters
    ----------
    arr: ndarray
        Numpy array that is potentially being upcast.

    use_dtype_backend: bool, default False
        If true, we cast to the associated nullable dtypes.

    Returns
    -------
    The casted array.
    """
    # 如果 arr 的 dtype 是 ExtensionDtype 类型，则直接返回 arr
    if isinstance(arr.dtype, ExtensionDtype):
        return arr

    # 获取 arr 的 NA 值
    na_value = na_values[arr.dtype]

    # 如果 arr 的 dtype 是 np.integer 的子类
    if issubclass(arr.dtype.type, np.integer):
        # 创建一个布尔掩码，标记所有等于 na_value 的位置
        mask = arr == na_value

        # 如果 use_dtype_backend 为 True，则使用 IntegerArray 进行类型转换
        if use_dtype_backend:
            arr = IntegerArray(arr, mask)
        else:
            # 否则将 arr 转换为 float 类型，将 mask 的位置设置为 np.nan
            arr = arr.astype(float)
            np.putmask(arr, mask, np.nan)

    # 如果 arr 的 dtype 是 np.bool_
    elif arr.dtype == np.bool_:
        # 创建一个布尔掩码，使用 np.uint8 进行视图转换，标记所有等于 na_value 的位置
        mask = arr.view(np.uint8) == na_value

        # 如果 use_dtype_backend 为 True，则使用 BooleanArray 进行类型转换
        if use_dtype_backend:
            arr = BooleanArray(arr, mask)
        else:
            # 否则将 arr 转换为 object 类型，将 mask 的位置设置为 np.nan
            arr = arr.astype(object)
            np.putmask(arr, mask, np.nan)

    # 如果 arr 的 dtype 是 np.float 的子类或者是 np.float32
    elif issubclass(arr.dtype.type, float) or arr.dtype.type == np.float32:
        # 如果 use_dtype_backend 为 True，则创建一个浮点数数组，并使用 FloatingArray 进行类型转换
        if use_dtype_backend:
            mask = np.isnan(arr)
            arr = FloatingArray(arr, mask)
    # 如果数组的数据类型是 np.object_ （即包含对象类型），并且使用了 dtype 后端
    elif arr.dtype == np.object_:
        # 如果使用了 dtype 后端，则创建一个 StringDtype 对象
        dtype = StringDtype()
        # 从 StringDtype 类型构造出数组对应的类型
        cls = dtype.construct_array_type()
        # 使用构造的类型从 arr 序列中创建数组对象，指定数据类型为 dtype
        arr = cls._from_sequence(arr, dtype=dtype)

    # 如果使用了 dtype 后端，并且指定了 dtype 后端为 "pyarrow"
    if use_dtype_backend and dtype_backend == "pyarrow":
        import pyarrow as pa
        # 如果 arr 是 IntegerArray 类型且所有值都是缺失值（NA）
        if isinstance(arr, IntegerArray) and arr.isna().all():
            # 在 pyarrow 中使用 null 代替 int64 类型
            arr = arr.to_numpy(na_value=None)
        # 将 arr 转换为 ArrowExtensionArray 类型，使用 pyarrow 中的数组对象 pa.array 创建
        arr = ArrowExtensionArray(pa.array(arr, from_pandas=True))

    # 返回处理后的数组 arr
    return arr
# ----------------------------------------------------------------------
# Type conversions / inference support code

# 定义一个Cython函数，返回一个元组，包含一个对象类型的NumPy数组和一个整数
cdef _string_box_utf8(parser_t *parser, int64_t col,
                      int64_t line_start, int64_t line_end,
                      bint na_filter, kh_str_starts_t *na_hashset,
                      const char *encoding_errors):
    cdef:
        int na_count = 0  # 计数NA值的数量
        Py_ssize_t i, lines  # 迭代器和行数
        coliter_t it  # 列迭代器
        const char *word = NULL  # 当前单词

        ndarray[object] result  # 结果数组，对象类型

        int ret = 0  # 返回值
        kh_strbox_t *table  # 字符串哈希表

        object pyval  # Python对象

        object NA = na_values[np.object_]  # NA值的Python对象
        khiter_t k  # 哈希表迭代器

    table = kh_init_strbox()  # 初始化字符串哈希表
    lines = line_end - line_start  # 计算行数
    result = np.empty(lines, dtype=np.object_)  # 创建空的对象数组

    coliter_setup(&it, parser, col, line_start)  # 设置列迭代器

    # 遍历每一行
    for i in range(lines):
        COLITER_NEXT(it, word)  # 获取下一个单词

        # 如果需要过滤NA值
        if na_filter:
            if kh_get_str_starts_item(na_hashset, word):
                # 在NA哈希表中
                na_count += 1
                result[i] = NA  # 将结果设置为NA值
                continue

        k = kh_get_strbox(table, word)  # 获取哈希表中的键

        # 如果在哈希表中存在
        if k != table.n_buckets:
            # 这会增加引用计数，但需要测试
            pyval = <object>table.vals[k]
        else:
            # 将单词转换为Python Unicode对象，并存入哈希表
            pyval = PyUnicode_Decode(word, strlen(word), "utf-8", encoding_errors)
            k = kh_put_strbox(table, word, &ret)
            table.vals[k] = <PyObject *>pyval

        result[i] = pyval  # 将结果存入结果数组

    kh_destroy_strbox(table)  # 销毁哈希表

    return result, na_count  # 返回结果数组和NA计数

# Cython装饰器，关闭边界检查
@cython.boundscheck(False)
cdef _categorical_convert(parser_t *parser, int64_t col,
                          int64_t line_start, int64_t line_end,
                          bint na_filter, kh_str_starts_t *na_hashset):
    "Convert column data into codes, categories"
    cdef:
        int na_count = 0  # 计数NA值的数量
        Py_ssize_t i, lines  # 迭代器和行数
        coliter_t it  # 列迭代器
        const char *word = NULL  # 当前单词

        int64_t NA = -1  # NA值的标志
        int64_t[::1] codes  # 代码数组，64位整数
        int64_t current_category = 0  # 当前类别

        int ret = 0  # 返回值
        kh_str_t *table  # 字符串哈希表
        khiter_t k  # 哈希表迭代器

    lines = line_end - line_start  # 计算行数
    codes = np.empty(lines, dtype=np.int64)  # 创建空的64位整数数组

    # 对解析后的值进行因子化，创建哈希表
    # 字节 -> 类别代码
    with nogil:
        table = kh_init_str()  # 初始化字符串哈希表
        coliter_setup(&it, parser, col, line_start)  # 设置列迭代器

        # 遍历每一行
        for i in range(lines):
            COLITER_NEXT(it, word)  # 获取下一个单词

            # 如果需要过滤NA值
            if na_filter:
                if kh_get_str_starts_item(na_hashset, word):
                    # 在NA值中
                    na_count += 1
                    codes[i] = NA  # 将代码设置为NA值
                    continue

            k = kh_get_str(table, word)  # 获取哈希表中的键

            # 如果不在哈希表中
            if k == table.n_buckets:
                k = kh_put_str(table, word, &ret)
                table.vals[k] = current_category  # 将当前类别存入哈希表
                current_category += 1

            codes[i] = table.vals[k]  # 将代码存入代码数组
    # 创建一个空的 NumPy 数组用于存储字符串对象，数组大小由 table.n_occupied 决定
    result = np.empty(table.n_occupied, dtype=np.object_)
    
    # 遍历哈希表中的每个桶
    for k in range(table.n_buckets):
        # 检查哈希表中第 k 个位置是否存在有效的字符串键值对
        if kh_exist_str(table, k):
            # 如果存在有效键值对，则将对应的字符串键值转换为 Python 的 Unicode 对象，并存储在 result 数组中
            result[table.vals[k]] = PyUnicode_FromString(table.keys[k])
    
    # 销毁哈希表，释放其占用的内存空间
    kh_destroy_str(table)
    
    # 将结果以 NumPy 数组的形式返回：codes 数组、result 数组和 na_count 变量
    return np.asarray(codes), result, na_count
# -> ndarray[f'|S{width}']
cdef _to_fw_string(parser_t *parser, int64_t col, int64_t line_start,
                   int64_t line_end, int64_t width):
    cdef:
        char *data  # 定义一个指向字符的指针，用于存储结果数据
        ndarray result  # 定义一个NumPy数组，用于存储结果字符串

    # 创建一个空的NumPy数组，用于存储字符串结果，dtype为指定宽度的字符串
    result = np.empty(line_end - line_start, dtype=f"|S{width}")
    # 将结果数组的数据部分转换为字符指针
    data = <char*>result.data

    # 使用 nogil 上下文，调用无GIL版本的字符串格式化函数，填充结果数据
    with nogil:
        _to_fw_string_nogil(parser, col, line_start, line_end, width, data)

    # 返回填充后的结果数组
    return result


cdef void _to_fw_string_nogil(parser_t *parser, int64_t col,
                              int64_t line_start, int64_t line_end,
                              size_t width, char *data) noexcept nogil:
    cdef:
        int64_t i  # 迭代器变量
        coliter_t it  # 列迭代器对象
        const char *word = NULL  # 字符指针，用于存储从迭代器获取的单词

    # 设置列迭代器，准备迭代处理数据
    coliter_setup(&it, parser, col, line_start)

    # 遍历每一行数据
    for i in range(line_end - line_start):
        COLITER_NEXT(it, word)  # 从迭代器获取下一个单词
        strncpy(data, word, width)  # 将单词复制到指定宽度的字符串中
        data += width  # 移动数据指针到下一个位置


cdef:
    char* cinf = b"inf"  # 定义常量字符串，表示正无穷
    char* cposinf = b"+inf"  # 定义常量字符串，表示正无穷
    char* cneginf = b"-inf"  # 定义常量字符串，表示负无穷

    char* cinfty = b"Infinity"  # 定义常量字符串，表示大写的无穷
    char* cposinfty = b"+Infinity"  # 定义常量字符串，表示大写的正无穷
    char* cneginfty = b"-Infinity"  # 定义常量字符串，表示大写的负无穷


# -> tuple[ndarray[float64_t], int]  | tuple[None, None]
cdef _try_double(parser_t *parser, int64_t col,
                 int64_t line_start, int64_t line_end,
                 bint na_filter, kh_str_starts_t *na_hashset, object na_flist):
    cdef:
        int error, na_count = 0  # 错误标志和缺失值计数初始化
        Py_ssize_t lines  # Python对象大小类型，用于存储行数
        float64_t *data  # 指向浮点数的指针，用于存储结果数据
        float64_t NA = na_values[np.float64]  # 定义浮点数NA值
        kh_float64_t *na_fset  # 定义浮点数集合的哈希表
        ndarray[float64_t] result  # 定义NumPy数组，用于存储结果数据
        bint use_na_flist = len(na_flist) > 0  # 判断是否使用na_flist

    # 计算行数
    lines = line_end - line_start
    # 创建一个空的NumPy数组，用于存储浮点数结果
    result = np.empty(lines, dtype=np.float64)
    # 将结果数组的数据部分转换为浮点数指针
    data = <float64_t *>result.data
    # 将na_flist转换为浮点数哈希集合
    na_fset = kset_float64_from_list(na_flist)
    
    # 使用 nogil 上下文，调用无GIL版本的浮点数转换函数，填充结果数据
    with nogil:
        error = _try_double_nogil(parser, parser.double_converter,
                                  col, line_start, line_end,
                                  na_filter, na_hashset, use_na_flist,
                                  na_fset, NA, data, &na_count)

    kh_destroy_float64(na_fset)  # 销毁浮点数哈希集合
    # 如果有错误发生，则返回空值
    if error != 0:
        return None, None
    # 返回填充后的结果数组和缺失值计数
    return result, na_count


cdef int _try_double_nogil(parser_t *parser,
                           float64_t (*double_converter)(
                               const char *, char **, char,
                               char, char, int, int *, int *) noexcept nogil,
                           int64_t col, int64_t line_start, int64_t line_end,
                           bint na_filter, kh_str_starts_t *na_hashset,
                           bint use_na_flist,
                           const kh_float64_t *na_flist,
                           float64_t NA, float64_t *data,
                           int *na_count) nogil:
    cdef:
        int error = 0  # 错误标志初始化
        Py_ssize_t i, lines = line_end - line_start  # 行数和迭代器初始化
        coliter_t it  # 列迭代器对象
        const char *word = NULL  # 字符指针，用于存储从迭代器获取的单词
        char *p_end  # 字符指针，用于标记转换结束位置
        khiter_t k64  # 哈希表迭代器

    na_count[0] = 0  # 缺失值计数初始化
    coliter_setup(&it, parser, col, line_start)  # 设置列迭代器，准备迭代处理数据
    // 如果需要进行缺失值处理
    if (na_filter) {
        // 遍历数据行数次数
        for (i = 0; i < lines; i++) {
            // 从迭代器中获取下一个单词
            COLITER_NEXT(it, word)

            // 检查单词是否存在于哈希集合中
            if (kh_get_str_starts_item(na_hashset, word)) {
                // 如果存在于哈希集合中，则增加缺失值计数器并设置数据为NA
                na_count[0] += 1
                data[0] = NA
            } else {
                // 否则，将单词转换为双精度浮点数，处理十进制、科学计数法、千分位等，并检查错误
                data[0] = double_converter(word, &p_end, parser.decimal,
                                           parser.sci, parser.thousands,
                                           1, &error, NULL)
                // 如果出现错误或转换后单词为空或包含错误信息
                if (error != 0 || p_end == word || p_end[0]) {
                    error = 0
                    // 检查特殊情况，设置数据为正无穷大或负无穷大
                    if (strcasecmp(word, cinf) == 0 ||
                        strcasecmp(word, cposinf) == 0 ||
                        strcasecmp(word, cinfty) == 0 ||
                        strcasecmp(word, cposinfty) == 0) {
                        data[0] = INF
                    } else if (strcasecmp(word, cneginf) == 0 ||
                               strcasecmp(word, cneginfty) == 0) {
                        data[0] = NEGINF
                    } else {
                        return 1
                    }
                }
                // 如果使用缺失值浮点列表，并且转换后的数值存在于列表中
                if (use_na_flist) {
                    k64 = kh_get_float64(na_flist, data[0])
                    // 如果存在于列表中，增加缺失值计数器并设置数据为NA
                    if (k64 != na_flist.n_buckets) {
                        na_count[0] += 1
                        data[0] = NA
                    }
                }
            }
            // 移动数据指针到下一个位置
            data += 1
        }
    } else {
        // 如果不需要进行缺失值处理，遍历数据行数次数
        for (i = 0; i < lines; i++) {
            // 从迭代器中获取下一个单词
            COLITER_NEXT(it, word)
            // 将单词转换为双精度浮点数，处理十进制、科学计数法、千分位等，并检查错误
            data[0] = double_converter(word, &p_end, parser.decimal,
                                       parser.sci, parser.thousands,
                                       1, &error, NULL)
            // 如果出现错误或转换后单词为空或包含错误信息
            if (error != 0 || p_end == word || p_end[0]) {
                error = 0
                // 检查特殊情况，设置数据为正无穷大或负无穷大
                if (strcasecmp(word, cinf) == 0 ||
                    strcasecmp(word, cposinf) == 0 ||
                    strcasecmp(word, cinfty) == 0 ||
                    strcasecmp(word, cposinfty) == 0) {
                    data[0] = INF
                } else if (strcasecmp(word, cneginf) == 0 ||
                           strcasecmp(word, cneginfty) == 0) {
                    data[0] = NEGINF
                } else {
                    return 1
                }
            }
            // 移动数据指针到下一个位置
            data += 1
        }
    }

    // 返回处理结果，无错误返回0
    return 0
# 尝试读取给定列中的 uint64 数据，处理在无 NA 值过滤和 NA 哈希集情况下的逻辑
cdef _try_uint64(parser_t *parser, int64_t col,
                 int64_t line_start, int64_t line_end,
                 bint na_filter, kh_str_starts_t *na_hashset):
    cdef:
        int error  # 错误码
        Py_ssize_t lines  # 行数
        coliter_t it  # 列迭代器
        uint64_t *data  # uint64 数据数组指针
        ndarray result  # 结果数组
        uint_state state  # uint 状态结构体

    lines = line_end - line_start  # 计算行数
    result = np.empty(lines, dtype=np.uint64)  # 创建一个空的 uint64 数组
    data = <uint64_t *>result.data  # 获取结果数组的数据指针

    uint_state_init(&state)  # 初始化 uint 状态
    coliter_setup(&it, parser, col, line_start)  # 设置列迭代器
    with nogil:
        # 在无 GIL 的情况下调用 _try_uint64_nogil 函数处理数据
        error = _try_uint64_nogil(parser, col, line_start, line_end,
                                  na_filter, na_hashset, data, &state)
    if error != 0:
        if error == ERROR_OVERFLOW:
            # 如果发生溢出错误，则抛出 OverflowError 异常
            raise OverflowError("Overflow")
        return None

    if uint64_conflict(&state):
        # 如果存在 uint64 类型冲突，则抛出 ValueError 异常
        raise ValueError("Cannot convert to numerical dtype")

    if state.seen_sint:
        # 如果存在有符号整数类型，则抛出 OverflowError 异常
        raise OverflowError("Overflow")

    return result  # 返回处理后的 uint64 结果数组


# 在无 GIL 的情况下尝试读取 uint64 数据的具体实现
cdef int _try_uint64_nogil(parser_t *parser, int64_t col,
                           int64_t line_start,
                           int64_t line_end, bint na_filter,
                           const kh_str_starts_t *na_hashset,
                           uint64_t *data, uint_state *state) nogil:
    cdef:
        int error  # 错误码
        Py_ssize_t i, lines = line_end - line_start  # 行数和行索引变量
        coliter_t it  # 列迭代器
        const char *word = NULL  # 单词字符串指针

    coliter_setup(&it, parser, col, line_start)  # 设置列迭代器起始位置

    if na_filter:
        # 如果需要进行 NA 值过滤
        for i in range(lines):
            COLITER_NEXT(it, word)  # 获取下一个单词
            if kh_get_str_starts_item(na_hashset, word):
                # 如果单词存在于 NA 哈希集中，则标记为 null，并设置数据为 0
                state.seen_null = 1
                data[i] = 0
                continue

            # 否则将单词转换为 uint64，并写入数据数组
            data[i] = str_to_uint64(state, word, INT64_MAX, UINT64_MAX,
                                    &error, parser.thousands)
            if error != 0:
                return error  # 如果转换出错，返回错误码
    else:
        # 如果不需要进行 NA 值过滤
        for i in range(lines):
            COLITER_NEXT(it, word)  # 获取下一个单词
            # 将单词转换为 uint64，并写入数据数组
            data[i] = str_to_uint64(state, word, INT64_MAX, UINT64_MAX,
                                    &error, parser.thousands)
            if error != 0:
                return error  # 如果转换出错，返回错误码

    return 0  # 返回 0 表示处理成功


# 尝试读取给定列中的 int64 数据，处理在无 NA 值过滤和 NA 哈希集情况下的逻辑
cdef _try_int64(parser_t *parser, int64_t col,
                int64_t line_start, int64_t line_end,
                bint na_filter, kh_str_starts_t *na_hashset):
    cdef:
        int error, na_count = 0  # 错误码和 NA 值计数
        Py_ssize_t lines  # 行数
        coliter_t it  # 列迭代器
        int64_t *data  # int64 数据数组指针
        ndarray result  # 结果数组
        int64_t NA = na_values[np.int64]  # NA 值设定为 int64 类型的 NA

    lines = line_end - line_start  # 计算行数
    result = np.empty(lines, dtype=np.int64)  # 创建一个空的 int64 数组
    data = <int64_t *>result.data  # 获取结果数组的数据指针
    coliter_setup(&it, parser, col, line_start)  # 设置列迭代器起始位置
    with nogil:
        # 在无 GIL 的情况下调用 _try_int64_nogil 函数处理数据
        error = _try_int64_nogil(parser, col, line_start, line_end,
                                 na_filter, na_hashset, NA, data, &na_count)
    # 如果错误码不为0，则处理错误情况
    if error != 0:
        # 如果错误码为ERROR_OVERFLOW，表示溢出错误
        if error == ERROR_OVERFLOW:
            # 抛出溢出错误异常
            raise OverflowError("Overflow")
        # 返回空结果和空计数
        return None, None

    # 返回正常的结果和计数
    return result, na_count
# 定义一个 Cython 函数 _try_int64_nogil，用于尝试将文本解析为 int64_t 类型数据，操作在无全局解锁（nogil）状态下进行
cdef int _try_int64_nogil(parser_t *parser, int64_t col,
                          int64_t line_start,
                          int64_t line_end, bint na_filter,
                          const kh_str_starts_t *na_hashset, int64_t NA,
                          int64_t *data, int *na_count) nogil:
    cdef:
        int error  # 错误码
        Py_ssize_t i, lines = line_end - line_start  # 循环变量 i 和 lines，计算行数

        coliter_t it  # 列迭代器对象
        const char *word = NULL  # 当前单词指针

    na_count[0] = 0  # 初始化缺失值计数为 0
    coliter_setup(&it, parser, col, line_start)  # 设置列迭代器，从指定列和起始行开始

    # 如果启用缺失值过滤
    if na_filter:
        for i in range(lines):  # 遍历行数
            COLITER_NEXT(it, word)  # 获取下一个单词
            if kh_get_str_starts_item(na_hashset, word):  # 检查单词是否在缺失值哈希表中
                # 在哈希表中，增加缺失值计数，将数据设置为 NA
                na_count[0] += 1
                data[i] = NA
                continue

            # 尝试将单词转换为 int64_t 类型数据，如果出错则返回错误码
            data[i] = str_to_int64(word, INT64_MIN, INT64_MAX,
                                   &error, parser.thousands)
            if error != 0:
                return error
    else:
        for i in range(lines):  # 遍历行数
            COLITER_NEXT(it, word)  # 获取下一个单词
            # 尝试将单词转换为 int64_t 类型数据，如果出错则返回错误码
            data[i] = str_to_int64(word, INT64_MIN, INT64_MAX,
                                   &error, parser.thousands)
            if error != 0:
                return error

    return 0  # 成功解析所有行，返回 0 表示无错误


# 定义一个 Cython 函数 _try_bool_flex，用于尝试将文本解析为灵活布尔值类型，并返回结果数组和缺失值计数
# 返回值类型为 tuple[ndarray[bool], int]
cdef _try_bool_flex(parser_t *parser, int64_t col,
                    int64_t line_start, int64_t line_end,
                    bint na_filter, const kh_str_starts_t *na_hashset,
                    const kh_str_starts_t *true_hashset,
                    const kh_str_starts_t *false_hashset):
    cdef:
        int error, na_count = 0  # 错误码和缺失值计数
        Py_ssize_t lines  # 行数
        uint8_t *data  # 数据指针
        ndarray result  # 结果数组
        uint8_t NA = na_values[np.bool_]  # 布尔类型的缺失值

    lines = line_end - line_start  # 计算行数
    result = np.empty(lines, dtype=np.uint8)  # 创建空的 uint8 类型数组
    data = <uint8_t *>result.data  # 获取数组的数据指针
    with nogil:  # 在无全局解锁（nogil）状态下执行以下代码块
        # 调用 _try_bool_flex_nogil 函数进行布尔类型的灵活解析
        error = _try_bool_flex_nogil(parser, col, line_start, line_end,
                                     na_filter, na_hashset, true_hashset,
                                     false_hashset, NA, data, &na_count)
    if error != 0:  # 如果解析过程中出错，则返回 None 和 None
        return None, None
    return result.view(np.bool_), na_count  # 成功解析，返回结果数组和缺失值计数


# 定义一个 Cython 函数 _try_bool_flex_nogil，用于在无全局解锁状态下尝试将文本解析为布尔类型数据
cdef int _try_bool_flex_nogil(parser_t *parser, int64_t col,
                              int64_t line_start,
                              int64_t line_end, bint na_filter,
                              const kh_str_starts_t *na_hashset,
                              const kh_str_starts_t *true_hashset,
                              const kh_str_starts_t *false_hashset,
                              uint8_t NA, uint8_t *data,
                              int *na_count) nogil:
    cdef:
        int error = 0  # 错误码初始化为 0
        Py_ssize_t i, lines = line_end - line_start  # 循环变量 i 和行数

        coliter_t it  # 列迭代器对象
        const char *word = NULL  # 当前单词指针

    na_count[0] = 0  # 初始化缺失值计数为 0
    coliter_setup(&it, parser, col, line_start)  # 设置列迭代器，从指定列和起始行开始
    # 如果需要进行空值过滤，则执行以下代码块
    if na_filter:
        # 遍历指定行数的数据
        for i in range(lines):
            # 使用迭代器从数据中获取单词
            COLITER_NEXT(it, word)

            # 如果单词存在于na_hashset中
            if kh_get_str_starts_item(na_hashset, word):
                # 增加空值计数器
                na_count[0] += 1
                # 将数据中当前位置标记为NA（表示缺失值）
                data[0] = NA
                # 移动数据指针到下一个位置
                data += 1
                # 继续处理下一个单词
                continue

            # 如果单词存在于true_hashset中
            if kh_get_str_starts_item(true_hashset, word):
                # 将数据中当前位置标记为1（表示True）
                data[0] = 1
                # 移动数据指针到下一个位置
                data += 1
                # 继续处理下一个单词
                continue

            # 如果单词存在于false_hashset中
            if kh_get_str_starts_item(false_hashset, word):
                # 将数据中当前位置标记为0（表示False）
                data[0] = 0
                # 移动数据指针到下一个位置
                data += 1
                # 继续处理下一个单词
                continue

            # 将单词转换为布尔值，并将结果存入data中
            error = to_boolean(word, data)
            # 如果转换过程中出现错误
            if error != 0:
                # 返回错误代码
                return error
            # 移动数据指针到下一个位置
            data += 1
    # 如果不需要进行空值过滤，则执行以下代码块
    else:
        # 遍历指定行数的数据
        for i in range(lines):
            # 使用迭代器从数据中获取单词
            COLITER_NEXT(it, word)

            # 如果单词存在于true_hashset中
            if kh_get_str_starts_item(true_hashset, word):
                # 将数据中当前位置标记为1（表示True）
                data[0] = 1
                # 移动数据指针到下一个位置
                data += 1
                # 继续处理下一个单词
                continue

            # 如果单词存在于false_hashset中
            if kh_get_str_starts_item(false_hashset, word):
                # 将数据中当前位置标记为0（表示False）
                data[0] = 0
                # 移动数据指针到下一个位置
                data += 1
                # 继续处理下一个单词
                continue

            # 将单词转换为布尔值，并将结果存入data中
            error = to_boolean(word, data)
            # 如果转换过程中出现错误
            if error != 0:
                # 返回错误代码
                return error
            # 移动数据指针到下一个位置
            data += 1

    # 如果所有数据处理完毕，返回0表示成功
    return 0
# 定义一个 C 函数，用于从 Python 列表创建 kh_str_starts_t* 类型的哈希表，如果出现异常则返回 NULL
cdef kh_str_starts_t* kset_from_list(list values) except NULL:
    # 定义 C 语言的变量
    cdef:
        Py_ssize_t i  # Python 中的整数大小
        kh_str_starts_t *table  # kh_str_starts_t 类型的哈希表指针
        int ret = 0  # 函数返回值，默认为 0
        object val  # Python 对象类型

    # 初始化一个 kh_str_starts_t* 类型的哈希表
    table = kh_init_str_starts()

    # 遍历传入的 Python 列表 values
    for i in range(len(values)):
        # 获取当前索引处的值
        val = values[i]

        # 检查值是否为字节类型，如果不是则销毁哈希表并抛出 ValueError 异常
        if not isinstance(val, bytes):
            kh_destroy_str_starts(table)
            raise ValueError("Must be all encoded bytes")

        # 将字节数据插入到哈希表中
        kh_put_str_starts_item(table, PyBytes_AsString(val), &ret)

    # 如果哈希表的桶数小于等于 128，则调整哈希表大小以减少哈希冲突，提高查找速度
    if table.table.n_buckets <= 128:
        # 调整哈希表大小为当前大小的 8 倍
        kh_resize_str_starts(table, table.table.n_buckets * 8)

    # 返回创建的哈希表指针
    return table


# 定义一个 C 函数，用于从 Python 列表创建 kh_float64_t* 类型的哈希表，如果出现异常则返回 NULL
cdef kh_float64_t* kset_float64_from_list(values) except NULL:
    # 定义 C 语言的变量
    cdef:
        kh_float64_t *table  # kh_float64_t 类型的哈希表指针
        int ret = 0  # 函数返回值，默认为 0
        float64_t val  # C 语言中的 double 类型
        object value  # Python 对象类型

    # 初始化一个 kh_float64_t* 类型的哈希表
    table = kh_init_float64()

    # 遍历传入的 Python 列表 values
    for value in values:
        # 将 Python 对象转换为 double 类型
        val = float(value)

        # 将 double 类型数据插入到哈希表中
        kh_put_float64(table, val, &ret)

    # 如果哈希表的桶数小于等于 128，则调整哈希表大小以减少哈希冲突，提高查找速度
    if table.n_buckets <= 128:
        # 调整哈希表大小为当前大小的 8 倍
        kh_resize_float64(table, table.n_buckets * 8)

    # 返回创建的哈希表指针
    return table


# 定义一个 C 函数，用于处理解析器错误的异常情况
cdef raise_parser_error(object base, parser_t *parser):
    cdef:
        object old_exc  # 旧的异常对象
        object exc_type  # 异常类型
        PyObject *type  # CPython 中的 PyObject 类型指针
        PyObject *value  # 异常值
        PyObject *traceback  # 异常回溯信息

    # 如果当前已经有异常发生，则获取异常信息并清除
    if PyErr_Occurred():
        PyErr_Fetch(&type, &value, &traceback)
        Py_XDECREF(traceback)

        # 如果异常值不为空，则处理异常
        if value != NULL:
            # 保存旧的异常对象，并释放异常值
            old_exc = <object>value
            Py_XDECREF(value)

            # 如果旧异常对象是字符串类型，则根据异常类型或默认创建 ParserError 异常并抛出
            if isinstance(old_exc, str):
                if type != NULL:
                    exc_type = <object>type
                else:
                    exc_type = ParserError

                Py_XDECREF(type)
                raise exc_type(old_exc)
            else:
                # 否则直接抛出旧的异常对象
                Py_XDECREF(type)
                raise old_exc

    # 构建错误消息，包括基础信息和解析器的 C 错误消息（如果有）
    message = f"{base}. C error: "
    if parser.error_msg != NULL:
        message += parser.error_msg.decode("utf-8")
    else:
        message += "no error message set"

    # 抛出 ParserError 异常，包含错误消息
    raise ParserError(message)


# ----------------------------------------------------------------------
# NA values
# 定义一个函数用于计算 NA 值
def _compute_na_values():
    # 获取不同整数类型的最小和最大值信息
    int64info = np.iinfo(np.int64)
    int32info = np.iinfo(np.int32)
    int16info = np.iinfo(np.int16)
    int8info = np.iinfo(np.int8)
    uint64info = np.iinfo(np.uint64)
    uint32info = np.iinfo(np.uint32)
    uint16info = np.iinfo(np.uint16)
    uint8info = np.iinfo(np.uint8)
    # 定义一个字典，用于存储不同 NumPy 数据类型对应的缺失值
    na_values = {
        np.float32: np.nan,        # 对应 np.float32 类型的值设为 NaN
        np.float64: np.nan,        # 对应 np.float64 类型的值设为 NaN
        np.int64: int64info.min,   # 对应 np.int64 类型的值设为 int64info 的最小值
        np.int32: int32info.min,   # 对应 np.int32 类型的值设为 int32info 的最小值
        np.int16: int16info.min,   # 对应 np.int16 类型的值设为 int16info 的最小值
        np.int8: int8info.min,     # 对应 np.int8 类型的值设为 int8info 的最小值
        np.uint64: uint64info.max, # 对应 np.uint64 类型的值设为 uint64info 的最大值
        np.uint32: uint32info.max, # 对应 np.uint32 类型的值设为 uint32info 的最大值
        np.uint16: uint16info.max, # 对应 np.uint16 类型的值设为 uint16info 的最大值
        np.uint8: uint8info.max,   # 对应 np.uint8 类型的值设为 uint8info 的最大值
        np.bool_: uint8info.max,   # 对应 np.bool_ 类型的值设为 uint8info 的最大值
        np.object_: np.nan,        # 对应 np.object_ 类型的值设为 NaN
    }
    # 返回定义的 na_values 字典，用于处理 NumPy 数组中的缺失值
    return na_values
# 计算并返回缺失值的集合
na_values = _compute_na_values()

# 遍历缺失值集合的副本
for k in list(na_values):
    # 将每个键的数据类型作为键，将其值复制到相应数据类型的条目中
    na_values[np.dtype(k)] = na_values[k]


# 定义一个 C 语言扩展函数 _apply_converter，返回一个 ArrayLike 对象
cdef _apply_converter(object f, parser_t *parser, int64_t col,
                      int64_t line_start, int64_t line_end):
    cdef:
        Py_ssize_t i, lines
        coliter_t it
        const char *word = NULL
        ndarray[object] result
        object val

    lines = line_end - line_start
    # 创建一个对象数组 result，用于存储转换后的结果
    result = np.empty(lines, dtype=np.object_)

    # 初始化列迭代器 it，从 parser 的指定列和行范围开始
    coliter_setup(&it, parser, col, line_start)

    # 遍历行数范围内的每一行
    for i in range(lines):
        # 使用列迭代器获取下一个单词
        COLITER_NEXT(it, word)
        # 将单词转换为 Python 的 Unicode 对象
        val = PyUnicode_FromString(word)
        # 对每个单词应用函数 f，并将结果存储到 result 数组中的相应位置
        result[i] = f(val)

    # 返回经过处理的对象数组 result
    return lib.maybe_convert_objects(result)


# 定义一个 C 语言扩展函数 _maybe_encode，接受一个列表 values 作为参数
cdef list _maybe_encode(list values):
    # 如果 values 是 None，则返回空列表
    if values is None:
        return []
    # 否则，对 values 中的每个元素进行 UTF-8 编码（如果是字符串类型），并返回结果列表
    return [x.encode("utf-8") if isinstance(x, str) else x for x in values]


# 定义 Python 函数 sanitize_objects，接受一个对象数组 values 和一个集合 na_values 作为参数
def sanitize_objects(ndarray[object] values, set na_values) -> int:
    """
    Convert specified values, including the given set na_values to np.nan.

    Parameters
    ----------
    values : ndarray[object]
    na_values : set

    Returns
    -------
    na_count : int
    """
    cdef:
        Py_ssize_t i, n
        object val, onan
        Py_ssize_t na_count = 0
        dict memo = {}

    # 获取 values 数组的长度 n
    n = len(values)
    # 设置 onan 为 np.nan
    onan = np.nan

    # 遍历 values 数组
    for i in range(n):
        val = values[i]
        # 如果当前值 val 在 na_values 集合中，将其替换为 np.nan，并增加 na_count 计数
        if val in na_values:
            values[i] = onan
            na_count += 1
        # 否则，如果当前值 val 已经在 memo 字典中，则将其替换为 memo 中对应的值
        elif val in memo:
            values[i] = memo[val]
        # 否则，将当前值 val 添加到 memo 字典中，并保持其不变
        else:
            memo[val] = val

    # 返回 na_count，表示被替换为 np.nan 的值的数量
    return na_count
```