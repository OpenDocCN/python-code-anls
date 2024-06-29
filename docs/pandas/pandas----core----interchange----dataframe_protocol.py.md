# `D:\src\scipysrc\pandas\pandas\core\interchange\dataframe_protocol.py`

```
"""
A verbatim copy (vendored) of the spec from https://github.com/data-apis/dataframe-api
"""

# 导入必要的模块和类
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import enum
from typing import (
    TYPE_CHECKING,
    Any,
    TypedDict,
)

# 如果是类型检查阶段，导入额外的模块
if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )


# 枚举类型：DLPack设备类型的整数编码
class DlpackDeviceType(enum.IntEnum):
    """Integer enum for device type codes matching DLPack."""

    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10


# 枚举类型：数据类型的整数编码
class DtypeKind(enum.IntEnum):
    """
    Integer enum for data types.

    Attributes
    ----------
    INT : int
        Matches to signed integer data type.
    UINT : int
        Matches to unsigned integer data type.
    FLOAT : int
        Matches to floating point data type.
    BOOL : int
        Matches to boolean data type.
    STRING : int
        Matches to string data type (UTF-8 encoded).
    DATETIME : int
        Matches to datetime data type.
    CATEGORICAL : int
        Matches to categorical data type.
    """

    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


# 枚举类型：列的空值类型的整数编码
class ColumnNullType(enum.IntEnum):
    """
    Integer enum for null type representation.

    Attributes
    ----------
    NON_NULLABLE : int
        Non-nullable column.
    USE_NAN : int
        Use explicit float NaN value.
    USE_SENTINEL : int
        Sentinel value besides NaN/NaT.
    USE_BITMASK : int
        The bit is set/unset representing a null on a certain position.
    USE_BYTEMASK : int
        The byte is set/unset representing a null on a certain position.
    """

    NON_NULLABLE = 0
    USE_NAN = 1
    USE_SENTINEL = 2
    USE_BITMASK = 3
    USE_BYTEMASK = 4


# 类型字典：列缓冲区的结构描述
class ColumnBuffers(TypedDict):
    # 数据缓冲区和其关联的数据类型组成的元组
    data: tuple[Buffer, Any]

    # 有效性缓冲区和其关联的数据类型组成的元组
    # 如果空值表示不是位掩码或字节掩码，则为None
    validity: tuple[Buffer, Any] | None

    # 偏移量缓冲区和其关联的数据类型组成的元组
    # 如果数据缓冲区没有关联的偏移量缓冲区，则为None
    offsets: tuple[Buffer, Any] | None


# 类型字典：分类描述的结构
class CategoricalDescription(TypedDict):
    # 字典索引顺序是否语义上有意义
    is_ordered: bool
    # 是否存在将分类值映射到其他对象的字典样式映射
    is_dictionary: bool
    # 仅Python级别有效（例如``{int: str}``）
    # 如果不是字典样式分类，则为None
    categories: Column | None


# 抽象类：缓冲区的基类
class Buffer(ABC):
    """
    Abstract base class for buffers.
    Placeholder for future elaboration.
    """
    """
    数据在缓冲区中保证在内存中是连续的。

    注意，这里没有 dtype 属性，缓冲区可以简单地被视为一块内存块。但是，如果附加到缓冲区的列具有 DLPack 支持的 dtype 并且实现了 ``__dlpack__``，那么 dtype 信息将包含在 ``__dlpack__`` 的返回值中。

    这种区别对于支持通过 DLPack 在缓冲区之间进行数据交换以及不具有固定字节数的元素的 dtype（如可变长度字符串）是有用的。
    """
    
    @property
    @abstractmethod
    def bufsize(self) -> int:
        """
        返回缓冲区的字节大小。
        """

    @property
    @abstractmethod
    def ptr(self) -> int:
        """
        返回缓冲区起始位置的指针，作为一个整数。
        """

    @abstractmethod
    def __dlpack__(self):
        """
        生成 DLPack 胶囊（参见数组 API 标准）。

        Raises:
            - TypeError: 如果缓冲区包含不支持的 dtype。
            - NotImplementedError: 如果未实现 DLPack 支持。

        对于连接到数组库非常有用。支持是可选的，因为对于仅基于 Python 的库来说，实现起来并不完全平凡。
        """
        raise NotImplementedError("__dlpack__")

    @abstractmethod
    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """
        返回缓冲区中数据所在的设备类型和设备 ID。使用与 DLPack 匹配的设备类型代码。
        注意：即使未实现 ``__dlpack__``，也必须实现此方法。
        """

        """
        返回缓冲区中数据所在的设备类型和设备 ID。使用与 DLPack 匹配的设备类型代码。
        注意：即使未实现 ``__dlpack__``，也必须实现此方法。
        """
# 定义一个抽象基类（Abstract Base Class），表示数据交换协议中的列对象
class Column(ABC):
    """
    A column object, with only the methods and properties required by the
    interchange protocol defined.

    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length
    strings).

    TBD: Arrow has a separate "null" dtype, and has no separate mask concept.
         Instead, it seems to use "children" for both columns with a bit mask,
         and for nested dtypes. Unclear whether this is elegant or confusing.
         This design requires checking the null representation explicitly.

         The Arrow design requires checking:
         1. the ARROW_FLAG_NULLABLE (for sentinel values)
         2. if a column has two children, combined with one of those children
            having a null dtype.

         Making the mask concept explicit seems useful. One null dtype would
         not be enough to cover both bit and byte masks, so that would mean
         even more checking if we did it the Arrow way.

    TBD: there's also the "chunk" concept here, which is implicit in Arrow as
         multiple buffers per array (= column here). Semantically it may make
         sense to have both: chunks were meant for example for lazy evaluation
         of data which doesn't fit in memory, while multiple buffers per column
         could also come from doing a selection operation on a single
         contiguous buffer.

         Given these concepts, one would expect chunks to be all of the same
         size (say a 10,000 row dataframe could have 10 chunks of 1,000 rows),
         while multiple buffers could have data-dependent lengths. Not an issue
         in pandas if one column is backed by a single NumPy array, but in
         Arrow it seems possible.
         Are multiple chunks *and* multiple buffers per column necessary for
         the purposes of this interchange protocol, or must producers either
         reuse the chunk concept for this or copy the data?

    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.
    """

    @abstractmethod
    # 抽象方法，返回列的大小（元素个数）
    def size(self) -> int:
        """
        Size of the column, in elements.

        Corresponds to DataFrame.num_rows() if column is a single chunk;
        equal to size of this current chunk otherwise.
        """

    @property
    @abstractmethod
    # 抽象属性，返回第一个元素的偏移量
    def offset(self) -> int:
        """
        Offset of first element.

        May be > 0 if using chunks; for example for a column with N chunks of
        equal size M (only the last chunk may be shorter),
        ``offset = n * M``, ``n = 0 .. N-1``.
        """

    @property
    @abstractmethod
    # 抽象属性，表示数据缓冲区
    def dtype(self) -> tuple[DtypeKind, int, str, str]:
        """
        Dtype description as a tuple ``(kind, bit-width, format string, endianness)``.

        Bit-width : the number of bits as an integer
        Format string : data type description format string in Apache Arrow C
                        Data Interface format.
        Endianness : current only native endianness (``=``) is supported

        Notes:
            - Kind specifiers are aligned with DLPack where possible (hence the
              jump to 20, leave enough room for future extension)
            - Masks must be specified as boolean with either bit width 1 (for bit
              masks) or 8 (for byte masks).
            - Dtype width in bits was preferred over bytes
            - Endianness isn't too useful, but included now in case in the future
              we need to support non-native endianness
            - Went with Apache Arrow format strings over NumPy format strings
              because they're more complete from a dataframe perspective
            - Format strings are mostly useful for datetime specification, and
              for categoricals.
            - For categoricals, the format string describes the type of the
              categorical in the data buffer. In case of a separate encoding of
              the categorical (e.g. an integer to string mapping), this can
              be derived from ``self.describe_categorical``.
            - Data types not included: complex, Arrow-style null, binary, decimal,
              and nested (list, struct, map, union) dtypes.
        """
        
    @property
    @abstractmethod
    def describe_categorical(self) -> CategoricalDescription:
        """
        If the dtype is categorical, there are two options:
        - There are only values in the data buffer.
        - There is a separate non-categorical Column encoding for categorical values.

        Raises TypeError if the dtype is not categorical

        Returns the dictionary with description on how to interpret the data buffer:
            - "is_ordered" : bool, whether the ordering of dictionary indices is
                             semantically meaningful.
            - "is_dictionary" : bool, whether a mapping of
                                categorical values to other objects exists
            - "categories" : Column representing the (implicit) mapping of indices to
                             category values (e.g. an array of cat1, cat2, ...).
                             None if not a dictionary-style categorical.

        TBD: are there any other in-memory representations that are needed?
        """
    @property
    @abstractmethod
    def null_count(self) -> int | None:
        """
        返回列中空值（或“null”）的表示形式，作为元组``(kind, value)``。

        Value: 如果kind是“sentinel value”，则是实际值；如果kind是位掩码或字节掩码，则是表示空值的值（0或1）。否则为None。
        """

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """
        返回列的元数据。

        详见`DataFrame.metadata`了解更多详情。
        """

    @abstractmethod
    def num_chunks(self) -> int:
        """
        返回列包含的块（chunks）数。
        """

    @abstractmethod
    def get_chunks(self, n_chunks: int | None = None) -> Iterable[Column]:
        """
        返回一个迭代器，产生列的块（chunks）。

        详见`DataFrame.get_chunks`了解关于``n_chunks``的更多详情。
        """

    @abstractmethod
    def get_buffers(self) -> ColumnBuffers:
        """
        返回一个包含底层缓冲区的字典。

        返回的字典包含以下内容：

            - "data": 一个包含数据的缓冲区的两元素元组，第一个元素是数据缓冲区，第二个元素是数据缓冲区的关联dtype。
            - "validity": 一个包含掩码值的缓冲区的两元素元组，指示缺失数据，第一个元素是掩码值缓冲区，第二个元素是掩码值缓冲区的关联dtype。
                          如果null表示不是位或字节掩码，则为None。
            - "offsets": 一个包含变长二进制数据（例如变长字符串）的偏移值的缓冲区的两元素元组，第一个元素是偏移值缓冲区，第二个元素是偏移值缓冲区的关联dtype。
                         如果数据缓冲区没有关联的偏移缓冲区，则为None。
        """
# 定义一个抽象基类 DataFrame，表示一个数据框架，只包含交换协议所需的方法。
class DataFrame(ABC):
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.

    A "data frame" represents an ordered collection of named columns.
    A column's "name" must be a unique string.
    Columns may be accessed by name or by position.

    This could be a public data frame class, or an object with the methods and
    attributes defined on this DataFrame class could be returned from the
    ``__dataframe__`` method of a public data frame class in a library adhering
    to the dataframe interchange protocol specification.
    """

    version = 0  # 版本号，表示协议的版本

    @abstractmethod
    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):
        """Construct a new interchange object, potentially changing the parameters."""
        # 抽象方法，用于构建一个新的交换对象，可能会改变参数。

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """
        The metadata for the data frame, as a dictionary with string keys. The
        contents of `metadata` may be anything, they are meant for a library
        to store information that it needs to, e.g., roundtrip losslessly or
        for two implementations to share data that is not (yet) part of the
        interchange protocol specification. For avoiding collisions with other
        entries, please add name the keys with the name of the library
        followed by a period and the desired name, e.g, ``pandas.indexcol``.
        """
        # 抽象属性，返回数据框架的元数据字典，键为字符串类型，值为任意类型。

    @abstractmethod
    def num_columns(self) -> int:
        """
        Return the number of columns in the DataFrame.
        """
        # 抽象方法，返回数据框架中的列数。

    @abstractmethod
    def num_rows(self) -> int | None:
        # TODO: not happy with Optional, but need to flag it may be expensive
        #       why include it if it may be None - what do we expect consumers
        #       to do here?
        """
        Return the number of rows in the DataFrame, if available.
        """
        # 抽象方法，返回数据框架中的行数，如果可用。

    @abstractmethod
    def num_chunks(self) -> int:
        """
        Return the number of chunks the DataFrame consists of.
        """
        # 抽象方法，返回数据框架的分块数。

    @abstractmethod
    def column_names(self) -> Iterable[str]:
        """
        Return an iterator yielding the column names.
        """
        # 抽象方法，返回一个迭代器，迭代列名。

    @abstractmethod
    def get_column(self, i: int) -> Column:
        """
        Return the column at the indicated position.
        """
        # 抽象方法，根据位置返回指定位置的列。

    @abstractmethod
    def get_column_by_name(self, name: str) -> Column:
        """
        Return the column whose name is the indicated name.
        """
        # 抽象方法，根据列名返回指定名称的列。

    @abstractmethod
    def get_columns(self) -> Iterable[Column]:
        """
        Return an iterator yielding the columns.
        """
        # 抽象方法，返回一个迭代器，迭代所有列。
    @abstractmethod
    def select_columns(self, indices: Sequence[int]) -> DataFrame:
        """
        Create a new DataFrame by selecting a subset of columns by index.
        """

    @abstractmethod
    def select_columns_by_name(self, names: Sequence[str]) -> DataFrame:
        """
        Create a new DataFrame by selecting a subset of columns by name.
        """

    @abstractmethod
    def get_chunks(self, n_chunks: int | None = None) -> Iterable[DataFrame]:
        """
        Return an iterator yielding the chunks.

        By default (None), yields the chunks that the data is stored as by the
        producer. If given, ``n_chunks`` must be a multiple of
        ``self.num_chunks()``, meaning the producer must subdivide each chunk
        before yielding it.
        """
```