# `.\numpy\numpy\ma\mrecords.py`

```
"""
Defines the equivalent of numpy.recarrays for masked arrays,
where fields can be accessed as attributes.
Note that numpy.ma.MaskedArray already supports structured datatypes
and the masking of individual fields.
"""

# 导入所需的模块和函数
from numpy.ma import (
    MAError, MaskedArray, masked, nomask, masked_array, getdata,
    getmaskarray, filled
)
import numpy.ma as ma  # 导入numpy.ma模块，并简写为ma
import warnings  # 导入警告模块

import numpy as np  # 导入numpy库，并简写为np
from numpy import dtype, ndarray, array as narray  # 导入dtype, ndarray, array，并给array起别名narray

from numpy._core.records import (
    recarray, fromarrays as recfromarrays, fromrecords as recfromrecords
)  # 导入numpy._core.records中的recarray, fromarrays和fromrecords函数

_byteorderconv = np._core.records._byteorderconv  # 设置_byteorderconv变量为numpy._core.records._byteorderconv

_check_fill_value = ma.core._check_fill_value  # 设置_check_fill_value变量为ma.core._check_fill_value

__all__ = [
    'MaskedRecords', 'mrecarray', 'fromarrays', 'fromrecords',
    'fromtextfile', 'addfield',
]  # 设置__all__变量，包含对外暴露的类和函数名

reserved_fields = ['_data', '_mask', '_fieldmask', 'dtype']  # 定义保留字段列表


def _checknames(descr, names=None):
    """
    Checks that field names `descr` are not reserved keywords.

    If this is the case, a default 'f%i' is substituted.  If the argument
    `names` is not None, updates the field names to valid names.
    """
    # 检查字段名是否为保留关键字，并进行必要的更改
    ndescr = len(descr)
    default_names = ['f%i' % i for i in range(ndescr)]
    if names is None:
        new_names = default_names
    else:
        if isinstance(names, (tuple, list)):
            new_names = names
        elif isinstance(names, str):
            new_names = names.split(',')
        else:
            raise NameError(f'illegal input names {names!r}')
        nnames = len(new_names)
        if nnames < ndescr:
            new_names += default_names[nnames:]
    ndescr = []
    for (n, d, t) in zip(new_names, default_names, descr.descr):
        if n in reserved_fields:
            if t[0] in reserved_fields:
                ndescr.append((d, t[1]))
            else:
                ndescr.append(t)
        else:
            ndescr.append((n, t[1]))
    return np.dtype(ndescr)


def _get_fieldmask(self):
    """
    Creates and returns a boolean recarray for field masking.

    This recarray sets the mask of each individual field of each record.
    """
    mdescr = [(n, '|b1') for n in self.dtype.names]
    fdmask = np.empty(self.shape, dtype=mdescr)
    fdmask.flat = tuple([False] * len(mdescr))
    return fdmask


class MaskedRecords(MaskedArray):
    """
    Subclass of MaskedArray representing masked records.

    Attributes
    ----------
    _data : recarray
        Underlying data, as a record array.
    _mask : boolean array
        Mask of the records. A record is masked when all its fields are
        masked.
    _fieldmask : boolean recarray
        Record array of booleans, setting the mask of each individual field
        of each record.
    _fill_value : record
        Filling values for each field.
    """
    def __new__(cls, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False,
                mask=nomask, hard_mask=False, fill_value=None, keep_mask=True,
                copy=False,
                **options):
        """
        Create a new instance of a masked array with specified parameters.

        """
        # Call the __new__ method of the base class 'recarray' to create the instance
        self = recarray.__new__(cls, shape, dtype=dtype, buf=buf, offset=offset,
                                strides=strides, formats=formats, names=names,
                                titles=titles, byteorder=byteorder,
                                aligned=aligned,)
        
        # Create a mask descriptor for the dtype of the array
        mdtype = ma.make_mask_descr(self.dtype)
        
        # Check if the mask parameter is 'nomask' or an empty array
        if mask is nomask or not np.size(mask):
            # If not keeping the mask, set _mask attribute to tuple of 'False' values
            if not keep_mask:
                self._mask = tuple([False] * len(mdtype))
        else:
            # Convert mask to a numpy array, with optional copy
            mask = np.array(mask, copy=copy)
            
            # Check if mask shape matches the shape of the array
            if mask.shape != self.shape:
                (nd, nm) = (self.size, mask.size)
                
                # Resize mask if its size is 1 or reshape if size matches data size
                if nm == 1:
                    mask = np.resize(mask, self.shape)
                elif nm == nd:
                    mask = np.reshape(mask, self.shape)
                else:
                    # Raise an error if mask and data sizes are incompatible
                    msg = "Mask and data not compatible: data size is %i, " + \
                          "mask size is %i."
                    raise MAError(msg % (nd, nm))
            
            # If not keeping the mask, set the mask using __setmask__ method
            if not keep_mask:
                self.__setmask__(mask)
                self._sharedmask = True
            else:
                # Convert mask to the appropriate dtype if necessary
                if mask.dtype == mdtype:
                    _mask = mask
                else:
                    _mask = np.array([tuple([m] * len(mdtype)) for m in mask],
                                     dtype=mdtype)
                self._mask = _mask
        
        # Return the created instance
        return self

    def __array_finalize__(self, obj):
        """
        Finalizes the creation of the masked array instance.

        """
        # Ensure _mask attribute exists by default
        _mask = getattr(obj, '_mask', None)
        
        # If _mask does not exist, create it based on obj's _mask or make a new one
        if _mask is None:
            objmask = getattr(obj, '_mask', nomask)
            _dtype = ndarray.__getattribute__(self, 'dtype')
            if objmask is nomask:
                _mask = ma.make_mask_none(self.shape, dtype=_dtype)
            else:
                mdescr = ma.make_mask_descr(_dtype)
                _mask = narray([tuple([m] * len(mdescr)) for m in objmask],
                               dtype=mdescr).view(recarray)
        
        # Update attributes dictionary with _mask attribute
        _dict = self.__dict__
        _dict.update(_mask=_mask)
        
        # Update attributes from the obj
        self._update_from(obj)
        
        # Change _baseclass attribute to 'recarray' if it was originally 'ndarray'
        if _dict['_baseclass'] == ndarray:
            _dict['_baseclass'] = recarray
        
        return

    @property
    def _data(self):
        """
        Property that returns the data as a recarray.

        """
        return ndarray.view(self, recarray)

    @property
    def _fieldmask(self):
        """
        Property that aliases to _mask attribute.

        """
        return self._mask
    def __len__(self):
        """
        Returns the length of the object.

        """
        # If the object has more than one dimension
        if self.ndim:
            # Return the length of the underlying data array
            return len(self._data)
        # If the object has only one record: return the number of fields
        return len(self.dtype)

    def __getattribute__(self, attr):
        try:
            # Try to get the attribute from the base class
            return object.__getattribute__(self, attr)
        except AttributeError:
            # If the attribute is not found, it must be a field name
            pass
        
        # Get the field dictionary from the data type of the ndarray
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        
        try:
            # Try to retrieve the field description from the dictionary
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            # Raise AttributeError if the field is not found
            raise AttributeError(
                f'record array has no attribute {attr}') from e
        
        # At this point, everything is going well
        
        # Access the local dictionary of the ndarray
        _localdict = ndarray.__getattribute__(self, '__dict__')
        # Get the view of the ndarray as a specified base class
        _data = ndarray.view(self, _localdict['_baseclass'])
        # Get the field data object
        obj = _data.getfield(*res)
        
        # Check if the dtype has names (structured array with named fields)
        if obj.dtype.names is not None:
            # Currently not implemented for masked records
            raise NotImplementedError("MaskedRecords is currently limited to simple records.")
        
        # Check for special attributes
        
        # Reset the object's mask if applicable
        hasmasked = False
        _mask = _localdict.get('_mask', None)
        if _mask is not None:
            try:
                _mask = _mask[attr]
            except IndexError:
                # If mask for the field is not found, use the default (nomask)
                pass
            # Check if there are any masked values in the mask
            tp_len = len(_mask.dtype)
            hasmasked = _mask.view((bool, ((tp_len,) if tp_len else ()))).any()
        
        # Check if the object has a shape or is masked
        if (obj.shape or hasmasked):
            # Convert the object to a MaskedArray
            obj = obj.view(MaskedArray)
            obj._baseclass = ndarray
            obj._isfield = True
            obj._mask = _mask
            
            # Reset the field values
            _fill_value = _localdict.get('_fill_value', None)
            if _fill_value is not None:
                try:
                    obj._fill_value = _fill_value[attr]
                except ValueError:
                    obj._fill_value = None
        else:
            # If neither shape nor mask is present, convert obj to a single item
            obj = obj.item()
        
        return obj
    # 定义特殊方法 __setattr__，用于设置对象的属性 attr 为值 val
    def __setattr__(self, attr, val):
        """
        Sets the attribute attr to the value val.

        """
        # 如果要设置的属性是 'mask' 或 'fieldmask'，则调用 __setmask__ 方法
        if attr in ['mask', 'fieldmask']:
            self.__setmask__(val)
            return
        # 创建一个快捷方式（避免每次调用都要获取属性）
        _localdict = object.__getattribute__(self, '__dict__')
        # 检查是否正在创建新的字段
        newattr = attr not in _localdict
        try:
            # 尝试设置属性，判断是否是通用属性
            ret = object.__setattr__(self, attr, val)
        except Exception:
            # 如果不是通用属性，检查是否是有效字段，否则抛出异常
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            optinfo = ndarray.__getattribute__(self, '_optinfo') or {}
            if not (attr in fielddict or attr in optinfo):
                raise
        else:
            # 获取字段字典
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            # 检查属性是否是字段
            if attr not in fielddict:
                return ret
            if newattr:
                # 如果是新添加的字段或者 setattr 操作作用于内部属性
                try:
                    object.__delattr__(self, attr)
                except Exception:
                    return ret
        # 尝试设置字段
        try:
            res = fielddict[attr][:2]  # 获取字段的信息
        except (TypeError, KeyError) as e:
            raise AttributeError(
                f'record array has no attribute {attr}') from e

        # 如果值是 masked
        if val is masked:
            _fill_value = _localdict['_fill_value']
            if _fill_value is not None:
                dval = _localdict['_fill_value'][attr]
            else:
                dval = val
            mval = True
        else:
            # 使用 filled 函数填充值
            dval = filled(val)
            # 获取值的掩码数组
            mval = getmaskarray(val)
        # 设置字段值
        obj = ndarray.__getattribute__(self, '_data').setfield(dval, *res)
        # 设置掩码属性
        _localdict['_mask'].__setitem__(attr, mval)
        return obj
    # 定义特殊方法 __getitem__，用于获取具有相同字段名基础的所有字段数据
    def __getitem__(self, indx):
        """
        Returns all the fields sharing the same fieldname base.

        The fieldname base is either `_data` or `_mask`.
        """
        # 获取当前对象的属性字典
        _localdict = self.__dict__
        # 获取当前对象的 _mask 属性，这是一个 ndarray 对象
        _mask = ndarray.__getattribute__(self, '_mask')
        # 获取当前对象的 _data 属性，并将其视图转换为指定的基础类
        _data = ndarray.view(self, _localdict['_baseclass'])

        # 如果 indx 是字符串类型，表示获取特定字段的数据
        if isinstance(indx, str):
            # 获取指定字段的数据，并将其视图转换为 MaskedArray 类型的对象
            obj = _data[indx].view(MaskedArray)
            # 设置 obj 的 _mask 属性为对应字段的 _mask 值
            obj._mask = _mask[indx]
            # 将 obj 的 _sharedmask 属性设置为 True，以便将掩码信息传播回 _fieldmask
            obj._sharedmask = True
            # 获取当前对象的填充值
            fval = _localdict['_fill_value']
            # 如果填充值不为 None，则设置 obj 的 _fill_value 属性为对应字段的填充值
            if fval is not None:
                obj._fill_value = fval[indx]
            # 如果 obj 的维度为 0 并且其 _mask 属性为 True，则返回 masked 常量
            if not obj.ndim and obj._mask:
                return masked
            # 否则返回 obj 对象
            return obj
        
        # 如果 indx 不是字符串类型，则表示获取部分元素
        # 首先获取数据部分
        obj = np.asarray(_data[indx]).view(mrecarray)
        # 设置 obj 的 _mask 属性为对应部分的 _mask 值
        obj._mask = np.asarray(_mask[indx]).view(recarray)
        # 返回 obj 对象
        return obj

    # 定义特殊方法 __setitem__，用于设置给定索引位置的记录值
    def __setitem__(self, indx, value):
        """
        Sets the given record to value.
        """
        # 调用父类 MaskedArray 的 __setitem__ 方法设置记录值
        MaskedArray.__setitem__(self, indx, value)
        # 如果 indx 是字符串类型，则更新对应字段的 _mask 值
        if isinstance(indx, str):
            self._mask[indx] = ma.getmaskarray(value)

    # 定义特殊方法 __str__，用于计算对象的字符串表示
    def __str__(self):
        """
        Calculates the string representation.
        """
        # 如果对象的大小大于 1，则生成包含所有字段数据的字符串列表
        if self.size > 1:
            mstr = [f"({','.join([str(i) for i in s])})"
                    for s in zip(*[getattr(self, f) for f in self.dtype.names])]
            return f"[{', '.join(mstr)}]"
        else:
            # 否则，生成包含所有字段数据的字符串列表
            mstr = [f"{','.join([str(i) for i in s])}"
                    for s in zip([getattr(self, f) for f in self.dtype.names])]
            return f"({', '.join(mstr)})"

    # 定义特殊方法 __repr__，用于计算对象的详细表示
    def __repr__(self):
        """
        Calculates the repr representation.
        """
        # 获取对象的字段名列表
        _names = self.dtype.names
        # 格式化字符串，用于生成每个字段的格式化表示
        fmt = "%%%is : %%s" % (max([len(n) for n in _names]) + 4,)
        # 生成对象的详细表示字符串列表
        reprstr = [fmt % (f, getattr(self, f)) for f in self.dtype.names]
        reprstr.insert(0, 'masked_records(')
        # 添加对象的填充值信息到详细表示字符串列表中
        reprstr.extend([fmt % ('    fill_value', self.fill_value),
                        '              )'])
        # 返回完整的详细表示字符串
        return str("\n".join(reprstr))
    # 返回一个 mrecarray 的视图。

    def view(self, dtype=None, type=None):
        """
        Returns a view of the mrecarray.

        """
        # 如果未指定 dtype，则根据 type 返回 ndarray 的视图
        if dtype is None:
            if type is None:
                output = ndarray.view(self)
            else:
                output = ndarray.view(self, type)
        # 如果指定了 dtype 但未指定 type
        elif type is None:
            try:
                # 如果 dtype 是 ndarray 的子类，则返回相应 dtype 的视图
                if issubclass(dtype, ndarray):
                    output = ndarray.view(self, dtype)
                else:
                    output = ndarray.view(self, dtype)
            # 处理 TypeError 异常
            except TypeError:
                dtype = np.dtype(dtype)
                # 若 dtype 没有 fields，则强制将数组视图设置为 dtype，使用第一个父类 basetype
                if dtype.fields is None:
                    basetype = self.__class__.__bases__[0]
                    output = self.__array__().view(dtype, basetype)
                    output._update_from(self)
                else:
                    output = ndarray.view(self, dtype)
                output._fill_value = None
        else:
            # 如果同时指定了 dtype 和 type，则返回指定 dtype 和 type 的视图
            output = ndarray.view(self, dtype, type)
        
        # 更新 mask，与 MaskedArray.view 的操作类似
        if (getattr(output, '_mask', nomask) is not nomask):
            mdtype = ma.make_mask_descr(output.dtype)
            output._mask = self._mask.view(mdtype, ndarray)
            output._mask.shape = output.shape
        
        # 返回输出结果
        return output

    # 强制将 mask 设置为硬掩码
    def harden_mask(self):
        """
        Forces the mask to hard.

        """
        self._hardmask = True

    # 将 mask 设置为软掩码
    def soften_mask(self):
        """
        Forces the mask to soft

        """
        self._hardmask = False

    # 返回 masked record 的副本
    def copy(self):
        """
        Returns a copy of the masked record.

        """
        # 复制 _data 的视图，并转换为与当前实例相同类型的视图
        copied = self._data.copy().view(type(self))
        copied._mask = self._mask.copy()
        return copied

    # 将数组的数据部分转换为列表
    def tolist(self, fill_value=None):
        """
        Return the data portion of the array as a list.

        Data items are converted to the nearest compatible Python type.
        Masked values are converted to fill_value. If fill_value is None,
        the corresponding entries in the output list will be ``None``.

        """
        # 如果指定了 fill_value，则用 fill_value 替换掩码值后返回列表
        if fill_value is not None:
            return self.filled(fill_value).tolist()
        
        # 否则，将填充后的数据部分转换为对象类型的数组 result
        result = narray(self.filled().tolist(), dtype=object)
        mask = narray(self._mask.tolist())
        result[mask] = None
        return result.tolist()
    def __getstate__(self):
        """
        Return the internal state of the masked array.

        This is for pickling.
        """
        # 定义一个包含对象状态的元组，用于序列化
        state = (1,                      # 版本号
                 self.shape,             # 数组形状
                 self.dtype,             # 数组数据类型
                 self.flags.fnc,         # 数组的 flags 属性
                 self._data.tobytes(),   # 数据部分转换为字节序列
                 self._mask.tobytes(),   # 掩码部分转换为字节序列
                 self._fill_value,       # 填充值
                 )
        return state

    def __setstate__(self, state):
        """
        Restore the internal state of the masked array.

        This is for pickling.  ``state`` is typically the output of the
        ``__getstate__`` output, and is a 5-tuple:

        - class name
        - a tuple giving the shape of the data
        - a typecode for the data
        - a binary string for the data
        - a binary string for the mask.
        """
        # 解析状态元组，恢复对象的内部状态
        (ver, shp, typ, isf, raw, msk, flv) = state
        # 调用父类的 __setstate__ 方法，恢复数据部分
        ndarray.__setstate__(self, (shp, typ, isf, raw))
        # 定义掩码数据类型
        mdtype = dtype([(k, np.bool) for (k, _) in self.dtype.descr])
        # 恢复掩码数据
        self.__dict__['_mask'].__setstate__((shp, mdtype, isf, msk))
        # 恢复填充值
        self.fill_value = flv

    def __reduce__(self):
        """
        Return a 3-tuple for pickling a MaskedArray.
        """
        # 返回一个3元组，用于序列化 MaskedArray 对象
        return (_mrreconstruct,
                (self.__class__, self._baseclass, (0,), 'b',),
                self.__getstate__())
def _mrreconstruct(subtype, baseclass, baseshape, basetype,):
    """
    Build a new MaskedArray from the information stored in a pickle.

    """
    # 使用给定的参数构建一个新的 MaskedArray 对象
    _data = ndarray.__new__(baseclass, baseshape, basetype).view(subtype)
    # 创建一个新的 ndarray 对象作为 mask，dtype 为 'b1'
    _mask = ndarray.__new__(ndarray, baseshape, 'b1')
    # 使用给定的 subtype、_data、_mask 和 basetype 创建并返回一个新的 subtype 对象
    return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)

mrecarray = MaskedRecords


###############################################################################
#                             Constructors                                    #
###############################################################################


def fromarrays(arraylist, dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None,
               fill_value=None):
    """
    Creates a mrecarray from a (flat) list of masked arrays.

    Parameters
    ----------
    arraylist : sequence
        A list of (masked) arrays. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : {None, dtype}, optional
        Data type descriptor.
    shape : {None, integer}, optional
        Number of records. If None, shape is defined from the shape of the
        first array in the list.
    formats : {None, sequence}, optional
        Sequence of formats for each individual field. If None, the formats will
        be autodetected by inspecting the fields and selecting the highest dtype
        possible.
    names : {None, sequence}, optional
        Sequence of the names of each field.
    fill_value : {None, sequence}, optional
        Sequence of data to be used as filling values.

    Notes
    -----
    Lists of tuples should be preferred over lists of lists for faster processing.

    """
    # 获取每个 arraylist 中元素的数据部分
    datalist = [getdata(x) for x in arraylist]
    # 获取每个 arraylist 中元素的 mask 部分，并确保至少是一维数组
    masklist = [np.atleast_1d(getmaskarray(x)) for x in arraylist]
    # 使用 recfromarrays 函数从 datalist 创建一个结构化数组 _array
    _array = recfromarrays(datalist,
                           dtype=dtype, shape=shape, formats=formats,
                           names=names, titles=titles, aligned=aligned,
                           byteorder=byteorder).view(mrecarray)
    # 将 _mask 属性的扁平视图设置为 masklist 的转置
    _array._mask.flat = list(zip(*masklist))
    # 如果提供了 fill_value，则将其赋给 _array 的 fill_value 属性
    if fill_value is not None:
        _array.fill_value = fill_value
    # 返回创建的 MaskedRecords 对象 _array
    return _array


def fromrecords(reclist, dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None,
                fill_value=None, mask=nomask):
    """
    Creates a MaskedRecords from a list of records.

    Parameters
    ----------
    reclist : sequence
        A list of records. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : {None, dtype}, optional
        Data type descriptor.
    
    """
    # 将给定的 reclist 转换为 MaskedRecords 对象并返回
    # 使用 recfromrecords 函数从 reclist 创建结构化数组
    _array = recfromrecords(reclist,
                            dtype=dtype, shape=shape, formats=formats,
                            names=names, titles=titles, aligned=aligned,
                            byteorder=byteorder, mask=mask).view(MaskedRecords)
    # 如果提供了 fill_value，则将其赋给 _array 的 fill_value 属性
    if fill_value is not None:
        _array.fill_value = fill_value
    # 返回创建的 MaskedRecords 对象 _array
    return _array
    # 定义函数的参数及其可选值范围，形状参数用于指定记录的数量，如果为 None，则从列表中第一个数组的形状推断得出
    # formats 参数用于指定每个字段的格式，如果为 None，则通过检查字段并选择可能的最高数据类型进行自动检测
    # names 参数用于指定每个字段的名称序列，如果为 None，则将根据字段自动检测
    # fill_value 参数用于指定用作填充值的数据序列，如果为 None，则不使用填充值
    # mask 参数用于指定外部掩码在数据上的应用情况，如果为 nomask，则表示无掩码
    # 注：建议使用元组列表而不是列表列表，以提高处理速度。
    
    """
    shape : {None,int}, optional
        记录的数量。如果为 None，则从列表中第一个数组的形状推断得出。
    formats : {None, sequence}, optional
        每个字段的格式序列。如果为 None，则会自动检测并选择可能的最高数据类型。
    names : {None, sequence}, optional
        每个字段的名称序列。
    fill_value : {None, sequence}, optional
        用作填充值的数据序列。
    mask : {nomask, sequence}, optional
        外部掩码在数据上的应用情况。
    
    Notes
    -----
    建议使用元组列表而不是列表列表以提高处理速度。
    """
    
    # 获取初始的 _fieldmask，如果有的话
    _mask = getattr(reclist, '_mask', None)
    
    # 获取记录列表
    if isinstance(reclist, ndarray):
        # 确保没有隐藏的掩码
        if isinstance(reclist, MaskedArray):
            # 填充掩码并将其视为 ndarray
            reclist = reclist.filled().view(ndarray)
        # 如果 dtype 为 None，则获取初始的数据类型
        if dtype is None:
            dtype = reclist.dtype
        # 将 ndarray 转换为列表
        reclist = reclist.tolist()
    
    # 从记录列表创建记录数组 mrec，使用指定的 dtype、shape、formats、names、titles、aligned、byteorder 参数
    mrec = recfromrecords(reclist, dtype=dtype, shape=shape, formats=formats,
                          names=names, titles=titles,
                          aligned=aligned, byteorder=byteorder).view(mrecarray)
    
    # 如果需要，设置填充值
    if fill_value is not None:
        mrec.fill_value = fill_value
    
    # 处理掩码
    if mask is not nomask:
        mask = np.asarray(mask)  # 将 mask 转换为 ndarray
        maskrecordlength = len(mask.dtype)
        if maskrecordlength:
            mrec._mask.flat = mask  # 如果 mask 的记录长度不为零，则将其展开并设置为 mrec 的掩码
        elif mask.ndim == 2:
            mrec._mask.flat = [tuple(m) for m in mask]  # 如果 mask 是二维数组，则将其转换为元组列表并设置为 mrec 的掩码
        else:
            mrec.__setmask__(mask)  # 否则，调用 __setmask__ 方法设置掩码
    
    # 如果之前有 _mask，则将其复制到 mrec 的掩码中
    if _mask is not None:
        mrec._mask[:] = _mask
    
    # 返回创建的结构化数组 mrec
    return mrec
def _guessvartypes(arr):
    """
    Tries to guess the dtypes of the str_ ndarray `arr`.

    Guesses by testing element-wise conversion. Returns a list of dtypes.
    The array is first converted to ndarray. If the array is 2D, the test
    is performed on the first line. An exception is raised if the file is
    3D or more.
    """
    # Initialize an empty list to store guessed dtypes
    vartypes = []
    # Convert `arr` to a NumPy ndarray
    arr = np.asarray(arr)
    # Check if `arr` is 2D, if so, use only the first row for dtype guessing
    if arr.ndim == 2:
        arr = arr[0]
    # Raise an error if `arr` has more than 2 dimensions
    elif arr.ndim > 2:
        raise ValueError("The array should be 2D at most!")
    # Start looping through elements in `arr` for dtype guessing
    for f in arr:
        try:
            int(f)
        except (ValueError, TypeError):
            try:
                float(f)
            except (ValueError, TypeError):
                try:
                    complex(f)
                except (ValueError, TypeError):
                    vartypes.append(arr.dtype)
                else:
                    vartypes.append(np.dtype(complex))
            else:
                vartypes.append(np.dtype(float))
        else:
            vartypes.append(np.dtype(int))
    # Return the list of guessed dtypes
    return vartypes


def openfile(fname):
    """
    Opens the file handle of file `fname`.

    """
    # Check if `fname` already has a `readline` attribute, indicating it's a file handle
    if hasattr(fname, 'readline'):
        return fname
    # Attempt to open the file `fname`
    try:
        f = open(fname)
    except FileNotFoundError as e:
        # Raise a more informative error if the file `fname` is not found
        raise FileNotFoundError(f"No such file: '{fname}'") from e
    # Check if the first line of the file does not start with '\\x'
    if f.readline()[:2] != "\\x":
        # Reset file pointer to the beginning of the file
        f.seek(0, 0)
        # Return the file handle `f`
        return f
    # Close the file `f` and raise an error indicating binary file type
    f.close()
    raise NotImplementedError("Wow, binary file")


def fromtextfile(fname, delimiter=None, commentchar='#', missingchar='',
                 varnames=None, vartypes=None,
                 *, delimitor=np._NoValue):  # backwards compatibility
    """
    Creates a mrecarray from data stored in the file `filename`.

    Parameters
    ----------
    fname : {file name/handle}
        Handle of an opened file.
    delimiter : {None, string}, optional
        Alphanumeric character used to separate columns in the file.
        If None, any (group of) white spacestring(s) will be used.
    commentchar : {'#', string}, optional
        Alphanumeric character used to mark the start of a comment.
    missingchar : {'', string}, optional
        String indicating missing data, and used to create the masks.
    varnames : {None, sequence}, optional
        Sequence of the variable names. If None, a list will be created from
        the first non empty line of the file.
    vartypes : {None, sequence}, optional
        Sequence of the variables dtypes. If None, it will be estimated from
        the first non-commented line.


    Ultra simple: the varnames are in the header, one line
    """
    # 检查是否指定了 delimitor 参数，并且不是 np._NoValue
    if delimitor is not np._NoValue:
        # 如果同时指定了 delimiter 参数，则抛出 TypeError 异常
        if delimiter is not None:
            raise TypeError("fromtextfile() got multiple values for argument "
                            "'delimiter'")
        # 发出警告，提示 'delimitor' 参数已被弃用，建议使用 'delimiter' 替代
        # NumPy 1.22.0, 2021-09-23
        warnings.warn("The 'delimitor' keyword argument of "
                      "numpy.ma.mrecords.fromtextfile() is deprecated "
                      "since NumPy 1.22.0, use 'delimiter' instead.",
                      DeprecationWarning, stacklevel=2)
        # 使用 delimitor 值来设定 delimiter
        delimiter = delimitor

    # 尝试打开文件
    ftext = openfile(fname)

    # 获取第一行不为空的数据作为变量名（varnames）
    while True:
        line = ftext.readline()
        # 获取第一行中注释字符之前的内容，并去除首尾空格
        firstline = line[:line.find(commentchar)].strip()
        # 使用 delimiter 分割第一行内容，作为变量名列表（_varnames）
        _varnames = firstline.split(delimiter)
        # 如果变量名列表（_varnames）长度大于 1，则跳出循环
        if len(_varnames) > 1:
            break
    # 如果未指定 varnames，则使用 _varnames
    if varnames is None:
        varnames = _varnames

    # 获取数据
    # 从文件中读取每行数据，使用 delimiter 分割，生成 masked_array 数组（_variables）
    _variables = masked_array([line.strip().split(delimiter) for line in ftext
                               if line[0] != commentchar and len(line) > 1])
    # 获取 _variables 的形状信息 (_, nfields)
    (_, nfields) = _variables.shape
    # 关闭文件
    ftext.close()

    # 尝试猜测数据类型（vartypes）
    if vartypes is None:
        # 如果未指定 vartypes，则根据第一行数据猜测数据类型
        vartypes = _guessvartypes(_variables[0])
    else:
        # 如果指定了 vartypes，则将其转换为 np.dtype 对象
        vartypes = [np.dtype(v) for v in vartypes]
        # 如果指定的 vartypes 长度与字段数不一致，则发出警告并使用默认值
        if len(vartypes) != nfields:
            msg = "Attempting to %i dtypes for %i fields!"
            msg += " Reverting to default."
            warnings.warn(msg % (len(vartypes), nfields), stacklevel=2)
            vartypes = _guessvartypes(_variables[0])

    # 构造描述符（mdescr）
    # 将变量名（varnames）和数据类型（vartypes）一一对应，组成描述符列表（mdescr）
    mdescr = [(n, f) for (n, f) in zip(varnames, vartypes)]
    # 获取填充值列表（mfillv），用于创建 masked_array 对象
    mfillv = [ma.default_fill_value(f) for f in vartypes]

    # 获取数据和掩码
    # 创建 masked_array 对象列表（_datalist）
    # 使用 _variables.T == missingchar 生成掩码（_mask），然后根据变量类型和填充值创建 masked_array 对象
    _mask = (_variables.T == missingchar)
    _datalist = [masked_array(a, mask=m, dtype=t, fill_value=f)
                 for (a, m, t, f) in zip(_variables.T, _mask, vartypes, mfillv)]

    # 使用 fromarrays 函数根据描述符（mdescr）创建 masked_records 对象并返回
    return fromarrays(_datalist, dtype=mdescr)
def addfield(mrecord, newfield, newfieldname=None):
    """Adds a new field to the masked record array

    Uses `newfield` as data and `newfieldname` as name. If `newfieldname`
    is None, the new field name is set to 'f%i' where `i` is the number of
    existing fields.

    Args:
    - mrecord (MaskedRecords): The masked record array to which the field will be added.
    - newfield: The data to be added as a new field.
    - newfieldname (str, optional): The name of the new field. Defaults to None.

    Returns:
    MaskedRecords: A new masked record array with the added field.

    """
    _data = mrecord._data  # Retrieve the data from the masked record array
    _mask = mrecord._mask  # Retrieve the mask from the masked record array

    # Determine the name for the new field if not provided or if it's in reserved fields
    if newfieldname is None or newfieldname in reserved_fields:
        newfieldname = 'f%i' % len(_data.dtype)

    newfield = ma.array(newfield)  # Convert newfield to a masked array

    # Create a new dtype with the existing fields and the new field
    newdtype = np.dtype(_data.dtype.descr + [(newfieldname, newfield.dtype)])
    # Create a new empty recarray with the updated dtype
    newdata = recarray(_data.shape, newdtype)

    # Add the existing fields to the new recarray
    [newdata.setfield(_data.getfield(*f), *f) for f in _data.dtype.fields.values()]

    # Add the new field to the new recarray
    newdata.setfield(newfield._data, *newdata.dtype.fields[newfieldname])

    # Convert newdata to MaskedRecords format
    newdata = newdata.view(MaskedRecords)

    # Create a new mask dtype for the new recarray
    newmdtype = np.dtype([(n, np.bool) for n in newdtype.names])
    # Create a new empty recarray for the mask
    newmask = recarray(_data.shape, newmdtype)

    # Add the existing masks to the new mask recarray
    [newmask.setfield(_mask.getfield(*f), *f) for f in _mask.dtype.fields.values()]

    # Add the mask of the new field to the new mask recarray
    newmask.setfield(getmaskarray(newfield), *newmask.dtype.fields[newfieldname])

    # Assign the new mask to newdata
    newdata._mask = newmask

    return newdata  # Return the new masked record array with the added field
```