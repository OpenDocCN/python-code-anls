# `D:\src\scipysrc\scipy\scipy\io\_netcdf.py`

```
# 定义了一个文档字符串，描述了这个模块的作用和支持的NetCDF文件的版本信息
"""
NetCDF reader/writer module.

This module is used to read and create NetCDF files. NetCDF files are
accessed through the `netcdf_file` object. Data written to and from NetCDF
files are contained in `netcdf_variable` objects. Attributes are given
as member variables of the `netcdf_file` and `netcdf_variable` objects.

This module implements the Scientific.IO.NetCDF API to read and create
NetCDF files. The same API is also used in the PyNIO and pynetcdf
modules, allowing these modules to be used interchangeably when working
with NetCDF files.

Only NetCDF3 is supported here; for NetCDF4 see
`netCDF4-python <http://unidata.github.io/netcdf4-python/>`__,
which has a similar API.
"""

# TODO:
# * properly implement ``_FillValue``.
# * fix character variables.
# * implement PAGESIZE for Python 2.6?

# 定义了一个包含模块公开接口的列表
__all__ = ['netcdf_file', 'netcdf_variable']

# 导入必要的模块和库
import warnings
import weakref
from operator import mul
from platform import python_implementation

# 导入 mmap 模块，并且将其重命名为 mm
import mmap as mm

# 导入 NumPy 库，从中导入一些函数和类
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce

# 判断当前 Python 解释器是否为 PyPy
IS_PYPY = python_implementation() == 'PyPy'

# 定义了一系列常量，代表了不同类型的 NetCDF 数据类型和填充值
ABSENT = b'\x00\x00\x00\x00\x00\x00\x00\x00'
ZERO = b'\x00\x00\x00\x00'
NC_BYTE = b'\x00\x00\x00\x01'
NC_CHAR = b'\x00\x00\x00\x02'
NC_SHORT = b'\x00\x00\x00\x03'
NC_INT = b'\x00\x00\x00\x04'
NC_FLOAT = b'\x00\x00\x00\x05'
NC_DOUBLE = b'\x00\x00\x00\x06'
NC_DIMENSION = b'\x00\x00\x00\n'
NC_VARIABLE = b'\x00\x00\x00\x0b'
NC_ATTRIBUTE = b'\x00\x00\x00\x0c'
FILL_BYTE = b'\x81'
FILL_CHAR = b'\x00'
FILL_SHORT = b'\x80\x01'
FILL_INT = b'\x80\x00\x00\x01'
FILL_FLOAT = b'\x7C\xF0\x00\x00'
FILL_DOUBLE = b'\x47\x9E\x00\x00\x00\x00\x00\x00'

# 映射了 NetCDF 类型到 Python 类型和字节数的字典
TYPEMAP = {NC_BYTE: ('b', 1),
           NC_CHAR: ('c', 1),
           NC_SHORT: ('h', 2),
           NC_INT: ('i', 4),
           NC_FLOAT: ('f', 4),
           NC_DOUBLE: ('d', 8)}

# 映射了 NetCDF 类型到填充值的字典
FILLMAP = {NC_BYTE: FILL_BYTE,
           NC_CHAR: FILL_CHAR,
           NC_SHORT: FILL_SHORT,
           NC_INT: FILL_INT,
           NC_FLOAT: FILL_FLOAT,
           NC_DOUBLE: FILL_DOUBLE}
# 定义一个映射，将数据类型字符和其对应字节数的组合映射到NetCDF数据类型常量
REVERSE = {('b', 1): NC_BYTE,        # 1字节有符号整数，映射为NC_BYTE
           ('B', 1): NC_CHAR,        # 1字节字符，映射为NC_CHAR
           ('c', 1): NC_CHAR,        # 1字节字符，映射为NC_CHAR
           ('h', 2): NC_SHORT,       # 2字节有符号短整数，映射为NC_SHORT
           ('i', 4): NC_INT,         # 4字节有符号整数，映射为NC_INT
           ('f', 4): NC_FLOAT,       # 4字节浮点数，映射为NC_FLOAT
           ('d', 8): NC_DOUBLE,      # 8字节双精度浮点数，映射为NC_DOUBLE

           # 下面的映射来自asarray(1).dtype.char和asarray('foo').dtype.char，
           # 用于从通用属性中获取类型信息。
           ('l', 4): NC_INT,         # 4字节有符号整数，映射为NC_INT
           ('S', 1): NC_CHAR}        # 1字节字符，映射为NC_CHAR

class netcdf_file:
    """
    用于NetCDF数据的文件对象。

    一个`netcdf_file`对象有两个标准属性：`dimensions` 和 `variables`。
    它们的值都是字典，分别将维度名称映射到它们的长度和将变量名称映射到变量。

    应用程序不应修改这些字典。

    所有其他属性对应于在NetCDF文件中定义的全局属性。
    全局文件属性通过给`netcdf_file`对象的属性赋值来创建。

    Parameters
    ----------
    filename : string or file-like
        字符串 -> 文件名
    mode : {'r', 'w', 'a'}, optional
        读取-写入-追加模式，默认为'r'
    mmap : None or bool, optional
        当读取时是否使用mmap`filename`。当`filename`是文件名时，默认为True，
        当`filename`是文件类对象时，默认为False。注意，当使用mmap时，
        返回的数据数组直接引用磁盘上的mmapped数据，只要存在对它的引用，
        就无法关闭文件。
    version : {1, 2}, optional
        要读取/写入的NetCDF版本，其中1表示“经典格式”，2表示“64位偏移格式”。默认为1。
        更多信息请参见`这里<https://docs.unidata.ucar.edu/nug/current/netcdf_introduction.html#select_format>`__。
    maskandscale : bool, optional
        是否基于属性自动缩放和/或屏蔽数据。默认为False。

    Notes
    -----
    该模块相对于其他模块的主要优势在于，它不需要将代码链接到NetCDF库。
    该模块源自`pupynere <https://bitbucket.org/robertodealmeida/pupynere/>`_。

    NetCDF文件是一种自描述的二进制数据格式。文件包含描述文件中维度和变量的元数据。
    可以在`这里<https://www.unidata.ucar.edu/software/netcdf/guide_toc.html>`__找到有关NetCDF文件的更多详细信息。
    NetCDF数据结构的主要部分有三个：

    1. Dimensions（维度）
    2. Variables（变量）
    3. Attributes（属性）

    维度部分记录了变量使用的每个维度的名称和长度。
    然后，变量将指示它使用哪些维度以及任何属性，例如数据单位，同时包含变量的数据值。
    包含与维度同名的变量是一种良好的实践，可以提供变量的值。

    """
    that axes. Lastly, the attributes section would contain additional
    information such as the name of the file creator or the instrument used to
    collect the data.

    When writing data to a NetCDF file, there is often the need to indicate the
    'record dimension'. A record dimension is the unbounded dimension for a
    variable. For example, a temperature variable may have dimensions of
    latitude, longitude and time. If one wants to add more temperature data to
    the NetCDF file as time progresses, then the temperature variable should
    have the time dimension flagged as the record dimension.

    In addition, the NetCDF file header contains the position of the data in
    the file, so access can be done in an efficient manner without loading
    unnecessary data into memory. It uses the ``mmap`` module to create
    Numpy arrays mapped to the data on disk, for the same purpose.

    Note that when `netcdf_file` is used to open a file with mmap=True
    (default for read-only), arrays returned by it refer to data
    directly on the disk. The file should not be closed, and cannot be cleanly
    closed when asked, if such arrays are alive. You may want to copy data arrays
    obtained from mmapped Netcdf file if they are to be processed after the file
    is closed, see the example below.

    Examples
    --------
    To create a NetCDF file:

    >>> from scipy.io import netcdf_file
    >>> import numpy as np
    >>> f = netcdf_file('simple.nc', 'w')
    >>> f.history = 'Created for a test'
    >>> f.createDimension('time', 10)
    >>> time = f.createVariable('time', 'i', ('time',))
    >>> time[:] = np.arange(10)
    >>> time.units = 'days since 2008-01-01'
    >>> f.close()

    Note the assignment of ``arange(10)`` to ``time[:]``.  Exposing the slice
    of the time variable allows for the data to be set in the object, rather
    than letting ``arange(10)`` overwrite the ``time`` variable.

    To read the NetCDF file we just created:

    >>> from scipy.io import netcdf_file
    >>> f = netcdf_file('simple.nc', 'r')
    >>> print(f.history)
    b'Created for a test'
    >>> time = f.variables['time']
    >>> print(time.units)
    b'days since 2008-01-01'
    >>> print(time.shape)
    (10,)
    >>> print(time[-1])
    9

    NetCDF files, when opened read-only, return arrays that refer
    directly to memory-mapped data on disk:

    >>> data = time[:]

    If the data is to be processed after the file is closed, it needs
    to be copied to main memory:

    >>> data = time[:].copy()
    >>> del time
    >>> f.close()
    >>> data.mean()
    4.5

    A NetCDF file can also be used as context manager:

    >>> from scipy.io import netcdf_file
    >>> with netcdf_file('simple.nc', 'r') as f:
    ...     print(f.history)
    b'Created for a test'

    """
    def __init__(self, filename, mode='r', mmap=None, version=1,
                 maskandscale=False):
        """
        Initialize netcdf_file from fileobj (str or file-like).

        Args:
            filename: 文件名或文件对象 (str 或类文件对象)
            mode: 打开文件的模式 ('r', 'w' 或 'a')
            mmap: 是否使用内存映射 (默认为 None，根据情况选择)
            version: 文件版本号 (默认为 1)
            maskandscale: 是否使用掩码和缩放 (默认为 False)
        """
        if mode not in 'rwa':
            raise ValueError("Mode must be either 'r', 'w' or 'a'.")

        if hasattr(filename, 'seek'):  # 如果是类文件对象
            self.fp = filename
            self.filename = 'None'  # 文件名设为字符串 'None'
            if mmap is None:
                mmap = False  # 默认不使用内存映射
            elif mmap and not hasattr(filename, 'fileno'):
                raise ValueError('Cannot use file object for mmap')
        else:  # 否则假设是字符串文件名
            self.filename = filename
            omode = 'r+' if mode == 'a' else mode
            self.fp = open(self.filename, '%sb' % omode)  # 打开文件对象
            if mmap is None:
                # 对于 PyPy，内存映射的文件通常不能在 GC 运行之前关闭，
                # 所以默认最好设为 mmap=False。
                mmap = (not IS_PYPY)  # 根据情况决定是否使用内存映射

        if mode != 'r':
            mmap = False  # 写入模式下不能使用内存映射

        self.use_mmap = mmap  # 记录是否使用内存映射
        self.mode = mode  # 记录文件打开模式
        self.version_byte = version  # 记录文件版本号
        self.maskandscale = maskandscale  # 记录是否使用掩码和缩放

        self.dimensions = {}  # 初始化维度字典
        self.variables = {}  # 初始化变量字典

        self._dims = []  # 初始化维度列表
        self._recs = 0  # 初始化记录数
        self._recsize = 0  # 初始化记录大小

        self._mm = None  # 初始化内存映射对象
        self._mm_buf = None  # 初始化内存映射缓冲区
        if self.use_mmap:
            self._mm = mm.mmap(self.fp.fileno(), 0, access=mm.ACCESS_READ)
            self._mm_buf = np.frombuffer(self._mm, dtype=np.int8)

        self._attributes = {}  # 初始化属性字典

        if mode in 'ra':
            self._read()  # 如果是读取模式，则调用读取方法

    def __setattr__(self, attr, value):
        """
        Store user defined attributes in a separate dict,
        so we can save them to file later.

        将用户定义的属性存储在单独的字典中，以便稍后保存到文件中。

        Args:
            attr: 属性名
            value: 属性值
        """
        try:
            self._attributes[attr] = value  # 将属性存储在特定字典中
        except AttributeError:
            pass
        self.__dict__[attr] = value  # 设置属性值
    # 定义一个方法，用于关闭 NetCDF 文件。
    def close(self):
        """Closes the NetCDF file."""
        # 检查是否存在文件指针并且文件未关闭
        if hasattr(self, 'fp') and not self.fp.closed:
            try:
                # 刷新数据到文件
                self.flush()
            finally:
                # 清空变量字典
                self.variables = {}
                # 如果存在内存映射缓冲区
                if self._mm_buf is not None:
                    # 创建对 self._mm_buf 的弱引用
                    ref = weakref.ref(self._mm_buf)
                    self._mm_buf = None
                    # 如果引用为 None，即 self._mm_buf 被垃圾回收了，可以关闭内存映射
                    if ref() is None:
                        self._mm.close()
                    else:
                        # 无法关闭内存映射，因为 self._mm_buf 仍然存在引用
                        warnings.warn(
                            "Cannot close a netcdf_file opened with mmap=True, when "
                            "netcdf_variables or arrays referring to its data still "
                            "exist. All data arrays obtained from such files refer "
                            "directly to data on disk, and must be copied before the "
                            "file can be cleanly closed. "
                            "(See netcdf_file docstring for more information on mmap.)",
                            category=RuntimeWarning, stacklevel=2,
                        )
                # 清空内存映射对象
                self._mm = None
                # 关闭文件指针
                self.fp.close()
    
    # 使用 __del__ 方法调用 close 方法，确保在对象被销毁时关闭文件
    __del__ = close

    # 实现上下文管理器的 __enter__ 方法
    def __enter__(self):
        return self
    
    # 实现上下文管理器的 __exit__ 方法
    def __exit__(self, type, value, traceback):
        self.close()

    # 定义一个方法，用于向 NetCDF 数据结构的 Dimension 部分添加一个维度
    def createDimension(self, name, length):
        """
        Adds a dimension to the Dimension section of the NetCDF data structure.

        Note that this function merely adds a new dimension that the variables can
        reference. The values for the dimension, if desired, should be added as
        a variable using `createVariable`, referring to this dimension.

        Parameters
        ----------
        name : str
            Name of the dimension (Eg, 'lat' or 'time').
        length : int
            Length of the dimension.

        See Also
        --------
        createVariable

        """
        # 如果长度为 None 并且已经存在维度，则抛出 ValueError
        if length is None and self._dims:
            raise ValueError("Only first dimension may be unlimited!")
        
        # 将维度名和长度添加到 dimensions 字典和 _dims 列表中
        self.dimensions[name] = length
        self._dims.append(name)
    # 创建一个新变量并添加到 netcdf_file 对象中，指定其数据类型和使用的维度
    def createVariable(self, name, type, dimensions):
        """
        Create an empty variable for the `netcdf_file` object, specifying its data
        type and the dimensions it uses.

        Parameters
        ----------
        name : str
            Name of the new variable.
        type : dtype or str
            Data type of the variable.
        dimensions : sequence of str
            List of the dimension names used by the variable, in the desired order.

        Returns
        -------
        variable : netcdf_variable
            The newly created ``netcdf_variable`` object.
            This object has also been added to the `netcdf_file` object as well.

        See Also
        --------
        createDimension

        Notes
        -----
        Any dimensions to be used by the variable should already exist in the
        NetCDF data structure or should be created by `createDimension` prior to
        creating the NetCDF variable.

        """
        # 根据给定的维度名称获取各个维度的长度，组成一个元组
        shape = tuple([self.dimensions[dim] for dim in dimensions])
        # 将 shape 中的 None 替换为 0，以便适应 NumPy 的要求
        shape_ = tuple([dim or 0 for dim in shape])  # replace None with 0 for NumPy

        # 转换变量的数据类型为大端字节顺序
        type = dtype(type)
        typecode, size = type.char, type.itemsize
        # 如果数据类型不在支持范围内，抛出错误
        if (typecode, size) not in REVERSE:
            raise ValueError("NetCDF 3 does not support type %s" % type)

        # 始终将数据转换为大端字节顺序，适用于 NetCDF 3
        data = empty(shape_, dtype=type.newbyteorder("B"))
        # 创建 netcdf_variable 对象并将其添加到 netcdf_file 对象中
        self.variables[name] = netcdf_variable(
                data, typecode, size, shape, dimensions,
                maskandscale=self.maskandscale)
        return self.variables[name]

    # 如果 netcdf_file 对象处于写入模式，则执行同步到磁盘的操作
    def flush(self):
        """
        Perform a sync-to-disk flush if the `netcdf_file` object is in write mode.

        See Also
        --------
        sync : Identical function

        """
        if hasattr(self, 'mode') and self.mode in 'wa':
            # 调用 _write 方法执行实际的写入操作
            self._write()
    sync = flush  # 同步方法与 flush 方法相同

    # 执行实际的写入操作，将 NetCDF 文件写入磁盘
    def _write(self):
        # 定位文件指针到开头位置并写入文件类型标识符 'CDF'
        self.fp.seek(0)
        self.fp.write(b'CDF')
        # 将版本字节序列转换为字节并写入文件
        self.fp.write(array(self.version_byte, '>b').tobytes())

        # 写入头信息和数据
        self._write_numrecs()
        self._write_dim_array()
        self._write_gatt_array()
        self._write_var_array()

    # 写入记录数量到文件中
    def _write_numrecs(self):
        # 获取所有记录变量中的最高记录数
        for var in self.variables.values():
            if var.isrec and len(var.data) > self._recs:
                self.__dict__['_recs'] = len(var.data)
        # 将记录数写入文件
        self._pack_int(self._recs)

    # 写入维度数组到文件中
    def _write_dim_array(self):
        if self.dimensions:
            # 写入维度数组标识符
            self.fp.write(NC_DIMENSION)
            # 将维度的数量打包写入文件
            self._pack_int(len(self.dimensions))
            # 遍历维度名称并写入名称和长度到文件
            for name in self._dims:
                self._pack_string(name)
                length = self.dimensions[name]
                # 将 None 替换为 0，以便于记录维度的情况
                self._pack_int(length or 0)  # replace None with 0 for record dimension
        else:
            # 若没有维度，则写入 ABSENT 标识符
            self.fp.write(ABSENT)
    # 调用 _write_att_array 方法，将 self._attributes 写入文件流 self.fp
    def _write_gatt_array(self):
        self._write_att_array(self._attributes)

    # 写入属性数组到文件流 self.fp
    def _write_att_array(self, attributes):
        if attributes:
            # 写入 NC_ATTRIBUTE 标志到文件流 self.fp
            self.fp.write(NC_ATTRIBUTE)
            # 将属性的数量打包成整数并写入文件流 self.fp
            self._pack_int(len(attributes))
            # 遍历属性字典，每个属性名打包成字符串，属性值调用 _write_att_values 方法写入文件流
            for name, values in attributes.items():
                self._pack_string(name)
                self._write_att_values(values)
        else:
            # 如果没有属性，则写入 ABSENT 标志到文件流 self.fp
            self.fp.write(ABSENT)

    # 写入变量数组到文件流 self.fp
    def _write_var_array(self):
        if self.variables:
            # 写入 NC_VARIABLE 标志到文件流 self.fp
            self.fp.write(NC_VARIABLE)
            # 将变量数量打包成整数并写入文件流 self.fp
            self._pack_int(len(self.variables))

            # 按照非记录变量优先，然后记录变量的顺序排序变量名
            def sortkey(n):
                v = self.variables[n]
                if v.isrec:
                    return (-1,)
                return v._shape
            variables = sorted(self.variables, key=sortkey, reverse=True)

            # 设置所有变量的元数据
            for name in variables:
                self._write_var_metadata(name)
            # 现在已经有了元数据，可以计算每个记录变量的 vsize，进而计算 recsize
            self.__dict__['_recsize'] = sum([
                    var._vsize for var in self.variables.values()
                    if var.isrec])
            # 设置所有变量的数据
            for name in variables:
                self._write_var_data(name)
        else:
            # 如果没有变量，则写入 ABSENT 标志到文件流 self.fp
            self.fp.write(ABSENT)

    # 写入变量的元数据到文件流 self.fp
    def _write_var_metadata(self, name):
        var = self.variables[name]

        # 变量名打包成字符串并写入文件流 self.fp
        self._pack_string(name)
        # 变量的维度数量打包成整数并写入文件流 self.fp
        self._pack_int(len(var.dimensions))
        # 遍历变量的维度列表，将维度名对应的维度索引打包成整数并写入文件流 self.fp
        for dimname in var.dimensions:
            dimid = self._dims.index(dimname)
            self._pack_int(dimid)

        # 写入变量的属性数组到文件流 self.fp
        self._write_att_array(var._attributes)

        # 根据变量的类型和字节大小获取相应的 NC 类型码并写入文件流 self.fp
        nc_type = REVERSE[var.typecode(), var.itemsize()]
        self.fp.write(nc_type)

        # 计算变量的 vsize
        if not var.isrec:
            vsize = var.data.size * var.data.itemsize
            vsize += -vsize % 4
        else:  # 记录变量
            try:
                vsize = var.data[0].size * var.data.itemsize
            except IndexError:
                vsize = 0
            # 统计所有记录变量的数量
            rec_vars = len([v for v in self.variables.values()
                            if v.isrec])
            if rec_vars > 1:
                vsize += -vsize % 4
        # 将计算得到的 vsize 存入变量的 _vsize 属性中，并将其打包成整数写入文件流 self.fp
        self.variables[name].__dict__['_vsize'] = vsize
        self._pack_int(vsize)

        # 打包一个假的 begin 值，并稍后设置真实值
        self.variables[name].__dict__['_begin'] = self.fp.tell()
        self._pack_begin(0)
    # 定义一个方法 `_write_var_data`，用于将变量数据写入文件中
    def _write_var_data(self, name):
        # 获取指定名称的变量
        var = self.variables[name]

        # 设置文件头的起始位置
        the_beguine = self.fp.tell()
        # 将文件指针移动到变量的起始位置
        self.fp.seek(var._begin)
        # 将起始位置打包写入文件
        self._pack_begin(the_beguine)
        # 将文件指针移回起始位置
        self.fp.seek(the_beguine)

        # 写入数据部分
        if not var.isrec:  # 如果不是记录变量
            # 将变量数据转换为字节并写入文件
            self.fp.write(var.data.tobytes())
            # 计算数据大小并写入填充数据
            count = var.data.size * var.data.itemsize
            self._write_var_padding(var, var._vsize - count)
        else:  # 如果是记录变量
            # 处理形状[0]小于记录数的记录变量
            if self._recs > len(var.data):
                shape = (self._recs,) + var.data.shape[1:]
                # 尝试原地调整大小，但不一定成功，因为数组可能不是单一段
                try:
                    var.data.resize(shape)
                except ValueError:
                    # 如果失败，复制并调整大小
                    dtype = var.data.dtype
                    var.__dict__['data'] = np.resize(var.data, shape).astype(dtype)

            # 记录当前位置及其初始值
            pos0 = pos = self.fp.tell()
            # 遍历记录变量的数据
            for rec in var.data:
                # 处理标量不能转换为大端序的情况
                if not rec.shape and (rec.dtype.byteorder == '<' or
                        (rec.dtype.byteorder == '=' and LITTLE_ENDIAN)):
                    rec = rec.byteswap()
                # 将记录转换为字节并写入文件
                self.fp.write(rec.tobytes())
                # 写入填充数据
                count = rec.size * rec.itemsize
                self._write_var_padding(var, var._vsize - count)
                # 移动到下一个记录的位置
                pos += self._recsize
                self.fp.seek(pos)
            # 将文件指针移动到变量结束位置
            self.fp.seek(pos0 + var._vsize)

    # 定义一个方法 `_write_var_padding`，用于向文件中写入填充数据
    def _write_var_padding(self, var, size):
        # 获取编码填充值
        encoded_fill_value = var._get_encoded_fill_value()
        # 计算填充次数
        num_fills = size // len(encoded_fill_value)
        # 将填充值写入文件指定次数
        self.fp.write(encoded_fill_value * num_fills)
    def _write_att_values(self, values):
        # 如果 values 具有 dtype 属性，则确定其对应的 NetCDF 类型
        if hasattr(values, 'dtype'):
            nc_type = REVERSE[values.dtype.char, values.dtype.itemsize]
        else:
            # 定义可能的数据类型和对应的 NetCDF 类型
            types = [(int, NC_INT), (float, NC_FLOAT), (str, NC_CHAR)]

            # 在 Python 3 中，bytes 索引为标量。检查是否为字符串类型
            if isinstance(values, (str, bytes)):
                sample = values
            else:
                try:
                    sample = values[0]  # 是否可以进行下标访问？
                except TypeError:
                    sample = values     # 标量

            # 确定 values 的类别，并找到对应的 NetCDF 类型
            for class_, nc_type in types:
                if isinstance(sample, class_):
                    break

        # 根据 NetCDF 类型获取数据类型码和大小
        typecode, size = TYPEMAP[nc_type]
        dtype_ = '>%s' % typecode
        # 在 Python 3 中，asarray() 无法处理 bytes 和 '>c'，需要转换为 'S'
        dtype_ = 'S' if dtype_ == '>c' else dtype_

        # 将 values 转换为指定类型的数组
        values = asarray(values, dtype=dtype_)

        # 将 NetCDF 类型写入文件流
        self.fp.write(nc_type)

        # 计算数据元素个数并写入文件流
        if values.dtype.char == 'S':
            nelems = values.itemsize
        else:
            nelems = values.size
        self._pack_int(nelems)

        # 如果 values 是标量或者其字节顺序需要调整，则进行相应处理
        if not values.shape and (values.dtype.byteorder == '<' or
                (values.dtype.byteorder == '=' and LITTLE_ENDIAN)):
            values = values.byteswap()
        self.fp.write(values.tobytes())
        count = values.size * values.itemsize
        # 对齐数据，以 4 字节为单位填充
        self.fp.write(b'\x00' * (-count % 4))  # pad

    def _read(self):
        # 检查文件头的魔数和版本信息
        magic = self.fp.read(3)
        if not magic == b'CDF':
            raise TypeError("Error: %s is not a valid NetCDF 3 file" %
                            self.filename)
        # 读取版本字节
        self.__dict__['version_byte'] = frombuffer(self.fp.read(1), '>b')[0]

        # 读取文件头和设置数据
        self._read_numrecs()
        self._read_dim_array()
        self._read_gatt_array()
        self._read_var_array()

    def _read_numrecs(self):
        # 读取记录数
        self.__dict__['_recs'] = self._unpack_int()

    def _read_dim_array(self):
        # 读取维度数组的头部信息
        header = self.fp.read(4)
        if header not in [ZERO, NC_DIMENSION]:
            raise ValueError("Unexpected header.")
        count = self._unpack_int()

        # 逐个读取维度信息并存储
        for dim in range(count):
            name = self._unpack_string().decode('latin1')
            length = self._unpack_int() or None  # 如果是记录维度，则长度为 None
            self.dimensions[name] = length
            self._dims.append(name)  # 保持维度的顺序

    def _read_gatt_array(self):
        # 读取全局属性数组，并将其作为对象的属性
        for k, v in self._read_att_array().items():
            self.__setattr__(k, v)

    def _read_att_array(self):
        # 读取属性数组的头部信息
        header = self.fp.read(4)
        if header not in [ZERO, NC_ATTRIBUTE]:
            raise ValueError("Unexpected header.")
        count = self._unpack_int()

        attributes = {}
        # 逐个读取属性并存储其值
        for attr in range(count):
            name = self._unpack_string().decode('latin1')
            attributes[name] = self._read_att_values()
        return attributes
    # 从文件流中解包字符串，并以Latin-1解码为Unicode字符串
    def _read_var(self):
        name = self._unpack_string().decode('latin1')
        # 初始化维度列表和形状列表
        dimensions = []
        shape = []
        # 从文件流中解包整数，获取维度数量
        dims = self._unpack_int()

        # 遍历维度数量，逐个解包维度标识符和维度名称，并将维度名称添加到列表中
        for i in range(dims):
            dimid = self._unpack_int()
            dimname = self._dims[dimid]
            dimensions.append(dimname)
            # 根据维度名称获取对应的维度大小，并将其添加到形状列表中
            dim = self.dimensions[dimname]
            shape.append(dim)
        
        # 将维度列表和形状列表转换为元组
        dimensions = tuple(dimensions)
        shape = tuple(shape)

        # 读取属性数组
        attributes = self._read_att_array()
        # 从文件流中读取4字节的数据类型标识符
        nc_type = self.fp.read(4)
        # 解包整数，获取变量大小
        vsize = self._unpack_int()
        # 解包整数或整数64位，获取起始位置
        begin = [self._unpack_int, self._unpack_int64][self.version_byte-1]()

        # 根据数据类型标识符从类型映射表中获取对应的类型码和大小
        typecode, size = TYPEMAP[nc_type]
        # 构造大端序的数据类型字符串
        dtype_ = '>%s' % typecode

        # 返回变量名称、维度、形状、属性、类型码、大小、数据类型字符串、起始位置、变量大小
        return name, dimensions, shape, attributes, typecode, size, dtype_, begin, vsize

    # 从文件流中读取属性值
    def _read_att_values(self):
        # 从文件流中读取4字节的数据类型标识符
        nc_type = self.fp.read(4)
        # 解包整数，获取属性值数量
        n = self._unpack_int()

        # 根据数据类型标识符从类型映射表中获取对应的类型码和大小
        typecode, size = TYPEMAP[nc_type]

        # 计算需要读取的总字节数
        count = n * size
        # 从文件流中读取相应数量的数据
        values = self.fp.read(int(count))
        # 读取填充字节，确保数据对齐
        self.fp.read(-count % 4)  # read padding

        # 如果数据类型不是字符型，则将数据转换为相应的NumPy数组，并处理单元素数组情况
        if typecode != 'c':
            values = frombuffer(values, dtype='>%s' % typecode).copy()
            if values.shape == (1,):
                values = values[0]
        else:
            # 如果数据类型是字符型，则去除末尾的空字节
            values = values.rstrip(b'\x00')
        
        # 返回属性值
        return values

    # 根据版本字节打包起始位置
    def _pack_begin(self, begin):
        if self.version_byte == 1:
            # 如果版本字节为1，则打包为32位整数
            self._pack_int(begin)
        elif self.version_byte == 2:
            # 如果版本字节为2，则打包为64位整数
            self._pack_int64(begin)

    # 打包32位整数
    def _pack_int(self, value):
        self.fp.write(array(value, '>i').tobytes())
    _pack_int32 = _pack_int  # 别名

    # 从文件流中解包32位整数
    def _unpack_int(self):
        return int(frombuffer(self.fp.read(4), '>i')[0])
    _unpack_int32 = _unpack_int  # 别名

    # 打包64位整数
    def _pack_int64(self, value):
        self.fp.write(array(value, '>q').tobytes())

    # 从文件流中解包64位整数
    def _unpack_int64(self):
        return frombuffer(self.fp.read(8), '>q')[0]

    # 打包字符串
    def _pack_string(self, s):
        # 获取字符串长度，并打包为32位整数
        count = len(s)
        self._pack_int(count)
        # 将字符串编码为Latin-1并写入文件流
        self.fp.write(s.encode('latin1'))
        # 写入填充字节，确保数据对齐
        self.fp.write(b'\x00' * (-count % 4))  # pad

    # 从文件流中解包字符串
    def _unpack_string(self):
        # 解包32位整数，获取字符串长度
        count = self._unpack_int()
        # 从文件流中读取对应长度的数据，并去除末尾的空字节
        s = self.fp.read(count).rstrip(b'\x00')
        # 读取填充字节，确保数据对齐
        self.fp.read(-count % 4)  # read padding
        # 返回解码后的字符串
        return s
class netcdf_variable:
    """
    A data object for netcdf files.

    `netcdf_variable` objects are constructed by calling the method
    `netcdf_file.createVariable` on the `netcdf_file` object. `netcdf_variable`
    objects behave much like array objects defined in numpy, except that their
    data resides in a file. Data is read by indexing and written by assigning
    to an indexed subset; the entire array can be accessed by the index ``[:]``
    or (for scalars) by using the methods `getValue` and `assignValue`.
    `netcdf_variable` objects also have attribute `shape` with the same meaning
    as for arrays, but the shape cannot be modified. There is another read-only
    attribute `dimensions`, whose value is the tuple of dimension names.

    All other attributes correspond to variable attributes defined in
    the NetCDF file. Variable attributes are created by assigning to an
    attribute of the `netcdf_variable` object.

    Parameters
    ----------
    data : array_like
        The data array that holds the values for the variable.
        Typically, this is initialized as empty, but with the proper shape.
    typecode : dtype character code
        Desired data-type for the data array.
    size : int
        Desired element size for the data array.
    shape : sequence of ints
        The shape of the array. This should match the lengths of the
        variable's dimensions.
    dimensions : sequence of strings
        The names of the dimensions used by the variable. Must be in the
        same order of the dimension lengths given by `shape`.
    attributes : dict, optional
        Attribute values (any type) keyed by string names. These attributes
        become attributes for the netcdf_variable object.
    maskandscale : bool, optional
        Whether to automatically scale and/or mask data based on attributes.
        Default is False.


    Attributes
    ----------
    dimensions : list of str
        List of names of dimensions used by the variable object.
    isrec, shape
        Properties

    See also
    --------
    isrec, shape

    """

    def __init__(self, data, typecode, size, shape, dimensions,
                 attributes=None,
                 maskandscale=False):
        # Initialize a new netcdf_variable object with provided parameters
        self.data = data
        self._typecode = typecode
        self._size = size
        self._shape = shape
        self.dimensions = dimensions
        self.maskandscale = maskandscale

        # Set attributes dictionary; if attributes are provided, add them as object attributes
        self._attributes = attributes or {}
        for k, v in self._attributes.items():
            self.__dict__[k] = v

    def __setattr__(self, attr, value):
        # Override setattr to store user-defined attributes in a separate dictionary
        # `_attributes`, enabling later saving to a file
        try:
            self._attributes[attr] = value
        except AttributeError:
            pass
        self.__dict__[attr] = value
    # 返回变量是否具有记录维度的布尔值。
    #
    # 记录维度是指可以在netcdf数据结构中轻松追加额外数据的维度，而不需要重写数据文件。
    # 这是`netcdf_variable`的只读属性。
    def isrec(self):
        return bool(self.data.shape) and not self._shape[0]
    isrec = property(isrec)

    # 返回数据变量的形状元组。
    #
    # 这是一个只读属性，不能像其他numpy数组那样修改。
    def shape(self):
        return self.data.shape
    shape = property(shape)

    # 从长度为一的`netcdf_variable`中检索标量值。
    #
    # Raises
    # ------
    # ValueError
    #     如果netcdf变量是长度大于一的数组，则会引发此异常。
    def getValue(self):
        return self.data.item()

    # 将标量值分配给长度为一的`netcdf_variable`。
    #
    # Parameters
    # ----------
    # value : scalar
    #     要分配给长度为一的netcdf变量的标量值（兼容类型）。此值将写入文件。
    #
    # Raises
    # ------
    # ValueError
    #     如果输入不是标量，或者目标不是长度为一的netcdf变量。
    def assignValue(self, value):
        if not self.data.flags.writeable:
            # NumPy中的一个bug的解决方法。在只读的内存映射数组上调用itemset()会导致段错误。
            # 参见NumPy票号＃1622和SciPy票号＃1202。
            # 当SciPy支持的最旧版本的NumPy包含＃1622的修复时，可以删除此`writeable`检查。
            raise RuntimeError("variable is not writeable")

        self.data[:] = value

    # 返回变量的类型码。
    #
    # Returns
    # -------
    # typecode : char
    #     变量的字符类型码（例如，'i'表示整数）。
    def typecode(self):
        return self._typecode

    # 返回变量的元素大小。
    #
    # Returns
    # -------
    # itemsize : int
    #     变量的元素大小（例如，float64为8）。
    def itemsize(self):
        return self._size
    # 当没有设置掩码和缩放时，直接返回数据数组中的指定索引位置的数据
    def __getitem__(self, index):
        if not self.maskandscale:
            return self.data[index]

        # 复制数据以避免修改原始数据
        data = self.data[index].copy()

        # 获取缺失值并应用到数据中
        missing_value = self._get_missing_value()
        data = self._apply_missing_value(data, missing_value)

        # 获取缩放因子和偏移量
        scale_factor = self._attributes.get('scale_factor')
        add_offset = self._attributes.get('add_offset')

        # 如果缩放因子或偏移量存在，则将数据类型转换为 np.float64
        if add_offset is not None or scale_factor is not None:
            data = data.astype(np.float64)

        # 如果存在缩放因子，则将数据乘以缩放因子
        if scale_factor is not None:
            data = data * scale_factor

        # 如果存在偏移量，则将偏移量加到数据上
        if add_offset is not None:
            data += add_offset

        return data

    # 设置指定索引位置的数据为给定数据
    def __setitem__(self, index, data):
        # 如果需要掩码和缩放
        if self.maskandscale:
            # 获取缺失值，若数据有填充值则使用填充值，否则使用默认值 999999
            missing_value = (
                    self._get_missing_value() or
                    getattr(data, 'fill_value', 999999))
            
            # 设置缺失值和 _FillValue 属性
            self._attributes.setdefault('missing_value', missing_value)
            self._attributes.setdefault('_FillValue', missing_value)
            
            # 根据缩放因子和偏移量对数据进行缩放和偏移处理
            data = ((data - self._attributes.get('add_offset', 0.0)) /
                    self._attributes.get('scale_factor', 1.0))
            
            # 将数据转换为 Masked Array，并用缺失值填充未掩码的部分
            data = np.ma.asarray(data).filled(missing_value)
            
            # 如果数据类型不是 'f' 或 'd'，且数据类型为浮点数，则将数据四舍五入
            if self._typecode not in 'fd' and data.dtype.kind == 'f':
                data = np.round(data)

        # 如果数据是记录变量，则根据索引扩展数据数组的大小
        if self.isrec:
            if isinstance(index, tuple):
                rec_index = index[0]
            else:
                rec_index = index
            
            # 计算需要扩展的记录数
            if isinstance(rec_index, slice):
                recs = (rec_index.start or 0) + len(data)
            else:
                recs = rec_index + 1
            
            # 如果需要扩展的记录数超过当前数据数组的长度，则进行扩展操作
            if recs > len(self.data):
                shape = (recs,) + self._shape[1:]
                # 尝试原地调整数组大小，若失败则重新分配空间
                try:
                    self.data.resize(shape)
                except ValueError:
                    dtype = self.data.dtype
                    self.__dict__['data'] = np.resize(self.data, shape).astype(dtype)

        # 将给定数据赋值到指定索引位置
        self.data[index] = data

    # 返回当前变量数据类型的默认编码填充值
    def _default_encoded_fill_value(self):
        """
        The default encoded fill-value for this Variable's data type.
        """
        # 获取当前变量数据类型对应的 NetCDF 数据类型
        nc_type = REVERSE[self.typecode(), self.itemsize()]
        
        # 返回对应的填充值
        return FILLMAP[nc_type]
    def _get_encoded_fill_value(self):
        """
        Returns the encoded fill value for this variable as bytes.

        This is taken from either the _FillValue attribute, or the default fill
        value for this variable's data type.
        """
        # 检查是否存在 _FillValue 属性
        if '_FillValue' in self._attributes:
            # 将 _FillValue 属性的值转换为与数据类型相匹配的 numpy 数组，并转换为字节序列
            fill_value = np.array(self._attributes['_FillValue'],
                                  dtype=self.data.dtype).tobytes()
            # 如果字节长度与变量数据类型的大小相等，则返回该填充值
            if len(fill_value) == self.itemsize():
                return fill_value
            else:
                # 否则调用默认的编码填充值函数
                return self._default_encoded_fill_value()
        else:
            # 如果不存在 _FillValue 属性，则调用默认的编码填充值函数
            return self._default_encoded_fill_value()

    def _get_missing_value(self):
        """
        Returns the value denoting "no data" for this variable.

        If this variable does not have a missing/fill value, returns None.

        If both _FillValue and missing_value are given, give precedence to
        _FillValue. The netCDF standard gives special meaning to _FillValue;
        missing_value is  just used for compatibility with old datasets.
        """
        # 检查是否存在 _FillValue 属性
        if '_FillValue' in self._attributes:
            # 返回 _FillValue 属性的值作为缺失值
            missing_value = self._attributes['_FillValue']
        elif 'missing_value' in self._attributes:
            # 否则返回 missing_value 属性的值作为缺失值
            missing_value = self._attributes['missing_value']
        else:
            # 如果不存在任何缺失值属性，则返回 None
            missing_value = None

        return missing_value

    @staticmethod
    def _apply_missing_value(data, missing_value):
        """
        Applies the given missing value to the data array.

        Returns a numpy.ma array, with any value equal to missing_value masked
        out (unless missing_value is None, in which case the original array is
        returned).
        """
        # 如果缺失值为 None，则直接返回原始数据数组
        if missing_value is None:
            newdata = data
        else:
            try:
                # 尝试检查缺失值是否为 NaN
                missing_value_isnan = np.isnan(missing_value)
            except (TypeError, NotImplementedError):
                # 捕获可能的异常，如某些数据类型无法测试 NaN
                missing_value_isnan = False

            # 如果缺失值是 NaN，则创建掩码，标记数据数组中 NaN 的位置
            if missing_value_isnan:
                mymask = np.isnan(data)
            else:
                # 否则创建掩码，标记数据数组中等于缺失值的位置
                mymask = (data == missing_value)

            # 根据掩码创建一个 numpy.ma 数组，将标记位置的值掩盖
            newdata = np.ma.masked_where(mymask, data)

        return newdata
# 将 netcdf_file 赋值给 NetCDFFile，用于简化代码中的引用
NetCDFFile = netcdf_file
# 将 netcdf_variable 赋值给 NetCDFVariable，用于简化代码中的引用
NetCDFVariable = netcdf_variable
```