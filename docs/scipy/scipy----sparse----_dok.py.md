# `D:\src\scipysrc\scipy\scipy\sparse\_dok.py`

```
# 定义一个基于字典的稀疏矩阵基类 _dok_base
class _dok_base(_spbase, IndexMixin, dict):
    # 设定格式为 'dok'
    _format = 'dok'

    # 初始化函数，接受不同类型的输入创建稀疏矩阵
    def __init__(self, arg1, shape=None, dtype=None, copy=False, *, maxprint=None):
        # 调用 _spbase 的初始化方法，处理输入参数
        _spbase.__init__(self, arg1, maxprint=maxprint)

        # 检查是否为数组
        is_array = isinstance(self, sparray)
        
        # 如果输入是元组且形状合法，则初始化为形状为 arg1 的 DOK 矩阵
        if isinstance(arg1, tuple) and isshape(arg1, allow_1d=is_array):
            self._shape = check_shape(arg1, allow_1d=is_array)
            self._dict = {}
            self.dtype = getdtype(dtype, default=float)
        
        # 如果输入是稀疏矩阵，则转换为 DOK 格式
        elif issparse(arg1):
            if arg1.format == self.format:
                arg1 = arg1.copy() if copy else arg1
            else:
                arg1 = arg1.todok()

            if dtype is not None:
                arg1 = arg1.astype(dtype, copy=False)

            self._dict = arg1._dict
            self._shape = check_shape(arg1.shape, allow_1d=is_array)
            self.dtype = getdtype(arg1.dtype)
        
        # 如果输入是密集矩阵，则根据参数转换为 DOK 格式
        else:
            try:
                arg1 = np.asarray(arg1)
            except Exception as e:
                raise TypeError('Invalid input format.') from e

            if arg1.ndim > 2:
                raise TypeError('Expected rank <=2 dense array or matrix.')

            # 如果是一维数组，根据 dtype 创建对应的非零元素字典
            if arg1.ndim == 1:
                if dtype is not None:
                    arg1 = arg1.astype(dtype)
                self._dict = {i: v for i, v in enumerate(arg1) if v != 0}
                self.dtype = getdtype(arg1.dtype)
            # 如果是二维数组，先转换为 COO 格式再转换为 DOK 格式
            else:
                d = self._coo_container(arg1, shape=shape, dtype=dtype).todok()
                self._dict = d._dict
                self.dtype = getdtype(d.dtype)
            self._shape = check_shape(arg1.shape, allow_1d=is_array)

    # 禁止直接使用 update 方法
    def update(self, val):
        raise NotImplementedError("Direct update to DOK sparse format is not allowed.")

    # 获取非零元素个数，如果有轴参数则抛出异常
    def _getnnz(self, axis=None):
        if axis is not None:
            raise NotImplementedError(
                "_getnnz over an axis is not implemented for DOK format."
            )
        return len(self._dict)

    # 统计非零元素的数量，如果有轴参数则抛出异常
    def count_nonzero(self, axis=None):
        if axis is not None:
            raise NotImplementedError(
                "count_nonzero over an axis is not implemented for DOK format."
            )
        return sum(x != 0 for x in self.values())

    # 将 _getnnz 方法的文档字符串设置为与 _spbase 中相同的文档字符串
    _getnnz.__doc__ = _spbase._getnnz.__doc__
    # 将 count_nonzero 方法的文档字符串设置为与 _spbase 中相同的文档字符串
    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    # 返回 _dict 字典的长度
    def __len__(self):
        return len(self._dict)

    # 检查是否包含指定的键
    def __contains__(self, key):
        return key in self._dict
    # 使用 setdefault 方法设置指定键的默认值，如果键不存在则插入到字典中
    def setdefault(self, key, default=None, /):
        return self._dict.setdefault(key, default)

    # 使用 __delitem__ 方法删除指定键对应的条目
    def __delitem__(self, key, /):
        del self._dict[key]

    # 清空字典中的所有条目
    def clear(self):
        return self._dict.clear()

    # 使用 pop 方法删除并返回指定键对应的值
    def pop(self, /, *args):
        return self._dict.pop(*args)

    # 抛出 TypeError 异常，因为不支持 dok_array 类型的反向迭代
    def __reversed__(self):
        raise TypeError("reversed is not defined for dok_array type")

    # 抛出 TypeError 异常，因为不支持 dok_array 类型的按位或操作
    def __or__(self, other):
        type_names = f"{type(self).__name__} and {type(other).__name__}"
        raise TypeError(f"unsupported operand type for |: {type_names}")

    # 抛出 TypeError 异常，因为不支持 dok_array 类型的按位或反转操作
    def __ror__(self, other):
        type_names = f"{type(self).__name__} and {type(other).__name__}"
        raise TypeError(f"unsupported operand type for |: {type_names}")

    # 抛出 TypeError 异常，因为不支持 dok_array 类型的按位或赋值操作
    def __ior__(self, other):
        type_names = f"{type(self).__name__} and {type(other).__name__}"
        raise TypeError(f"unsupported operand type for |: {type_names}")

    # 使用 popitem 方法删除并返回字典中的一对键值对
    def popitem(self):
        return self._dict.popitem()

    # 返回字典中所有键值对的视图
    def items(self):
        return self._dict.items()

    # 返回字典中所有键的视图
    def keys(self):
        return self._dict.keys()

    # 返回字典中所有值的视图
    def values(self):
        return self._dict.values()

    # 获取指定键的值，如果键不存在则返回默认值 0.0
    def get(self, key, default=0.0):
        """This provides dict.get method functionality with type checking"""
        if key in self._dict:
            return self._dict[key]
        # 如果键不存在于字典中，进行类型检查并根据条件抛出异常
        if isintlike(key) and self.ndim == 1:
            key = (key,)
        if self.ndim != len(key):
            raise IndexError(f'Index {key} length needs to match self.shape')
        try:
            for i in key:
                assert isintlike(i)
        except (AssertionError, TypeError, ValueError) as e:
            raise IndexError('Index must be or consist of integers.') from e
        # 根据索引和数组维度计算新的键值
        key = tuple(i + M if i < 0 else i for i, M in zip(key, self.shape))
        if any(i < 0 or i >= M for i, M in zip(key, self.shape)):
            raise IndexError('Index out of bounds.')
        if self.ndim == 1:
            key = key[0]
        return self._dict.get(key, default)

    # 用于获取一维索引的值的方法
    def _get_int(self, idx):
        return self._dict.get(idx, self.dtype.type(0))

    # 用于获取一维切片的值的方法
    def _get_slice(self, idx):
        i_range = range(*idx.indices(self.shape[0]))
        return self._get_array(list(i_range))
    # 获取数组中指定位置的元素或子集
    def _get_array(self, idx):
        # 将索引转换为 NumPy 数组
        idx = np.asarray(idx)
        # 如果索引是0维，返回索引对应的值，否则返回默认值0
        if idx.ndim == 0:
            val = self._dict.get(int(idx), self.dtype.type(0))
            return np.array(val, stype=self.dtype)
        
        # 创建一个新的 DOK 容器，用于存储新的数据
        new_dok = self._dok_container(idx.shape, dtype=self.dtype)
        # 获取所有索引位置的值
        dok_vals = [self._dict.get(i, 0) for i in idx.ravel()]
        
        # 如果有值存在
        if dok_vals:
            # 如果索引是1维，将值添加到新的 DOK 容器中
            if len(idx.shape) == 1:
                for i, v in enumerate(dok_vals):
                    if v:
                        new_dok._dict[i] = v
            else:
                # 如果索引是多维，解开索引并将对应的值添加到新的 DOK 容器中
                new_idx = np.unravel_index(np.arange(len(dok_vals)), idx.shape)
                new_idx = new_idx[0] if len(new_idx) == 1 else zip(*new_idx)
                for i, v in zip(new_idx, dok_vals):  # 此处原文有错误，已更正
                    if v:
                        new_dok._dict[i] = v
        return new_dok

    # 获取二维矩阵中指定整数索引的元素
    def _get_intXint(self, row, col):
        return self._dict.get((row, col), self.dtype.type(0))

    # 获取二维矩阵中指定整数行和切片列的元素
    def _get_intXslice(self, row, col):
        return self._get_sliceXslice(slice(row, row + 1), col)

    # 获取二维矩阵中指定切片行和整数列的元素
    def _get_sliceXint(self, row, col):
        return self._get_sliceXslice(row, slice(col, col + 1))

    # 获取二维矩阵中指定切片行和切片列的元素
    def _get_sliceXslice(self, row, col):
        # 获取切片行和列的起始、停止和步长
        row_start, row_stop, row_step = row.indices(self.shape[0])
        col_start, col_stop, col_step = col.indices(self.shape[1])
        # 根据行和列的起始、停止和步长生成行和列的范围
        row_range = range(row_start, row_stop, row_step)
        col_range = range(col_start, col_stop, col_step)
        shape = (len(row_range), len(col_range))
        
        # 根据条件选择不同的执行路径
        # 如果非零元素数量大于等于2倍的行列数，选择O(nr*nc)路径
        if len(self) >= 2 * shape[0] * shape[1]:
            # 循环遍历行列范围，获取对应元素
            return self._get_columnXarray(row_range, col_range)
        
        # 否则选择O(nnz)路径，遍历矩阵中的元素
        newdok = self._dok_container(shape, dtype=self.dtype)
        for key in self.keys():
            i, ri = divmod(int(key[0]) - row_start, row_step)
            if ri != 0 or i < 0 or i >= shape[0]:
                continue
            j, rj = divmod(int(key[1]) - col_start, col_step)
            if rj != 0 or j < 0 or j >= shape[1]:
                continue
            newdok._dict[i, j] = self._dict[key]
        return newdok

    # 获取二维矩阵中指定整数行和数组列的元素
    def _get_intXarray(self, row, col):
        col = col.squeeze()
        return self._get_columnXarray([row], col)

    # 获取二维矩阵中指定数组行和整数列的元素
    def _get_arrayXint(self, row, col):
        row = row.squeeze()
        return self._get_columnXarray(row, [col])

    # 获取二维矩阵中指定切片行和数组列的元素
    def _get_sliceXarray(self, row, col):
        row = list(range(*row.indices(self.shape[0])))
        return self._get_columnXarray(row, col)

    # 获取二维矩阵中指定数组行和切片列的元素
    def _get_arrayXslice(self, row, col):
        col = list(range(*col.indices(self.shape[1])))
        return self._get_columnXarray(row, col)
    # 根据给定的行和列索引，从字典中获取对应的值，生成新的稀疏矩阵对象
    def _get_columnXarray(self, row, col):
        # 创建一个新的稀疏矩阵容器，根据行和列的长度确定其形状和数据类型
        newdok = self._dok_container((len(row), len(col)), dtype=self.dtype)

        # 遍历行索引和列索引的组合
        for i, r in enumerate(row):
            for j, c in enumerate(col):
                # 从内部字典获取键为(r, c)的值，如果不存在则默认为0
                v = self._dict.get((r, c), 0)
                # 如果值不为0，则将其添加到新的稀疏矩阵对象中的相应位置
                if v:
                    newdok._dict[i, j] = v
        return newdok

    # 根据给定的行和列数组索引，从字典中获取对应的值，生成新的稀疏矩阵对象
    def _get_arrayXarray(self, row, col):
        # 将行和列数组广播为至少二维的数组
        i, j = map(np.atleast_2d, np.broadcast_arrays(row, col))
        # 创建一个新的稀疏矩阵容器，形状与广播后的数组形状相同，使用指定的数据类型
        newdok = self._dok_container(i.shape, dtype=self.dtype)

        # 遍历广播后的行列索引的所有组合
        for key in itertools.product(range(i.shape[0]), range(i.shape[1])):
            # 从内部字典获取键为(i[key], j[key])的值，如果不存在则默认为0
            v = self._dict.get((i[key], j[key]), 0)
            # 如果值不为0，则将其添加到新的稀疏矩阵对象中的相应位置
            if v:
                newdok._dict[key] = v
        return newdok

    # 一维索引设置方法，根据索引和值更新内部字典
    def _set_int(self, idx, x):
        # 如果值不为0，则将其设置到内部字典中对应的索引位置
        if x:
            self._dict[idx] = x
        # 如果值为0且索引在字典中存在，则从字典中删除该索引
        elif idx in self._dict:
            del self._dict[idx]

    # 一维数组索引设置方法，根据索引数组和值数组更新内部字典
    def _set_array(self, idx, x):
        # 将索引数组展平为一维数组
        idx_set = idx.ravel()
        # 将值数组展平为一维数组
        x_set = x.ravel()
        # 如果索引数组和值数组长度不相等，则根据情况抛出异常或将值数组扩展至与索引数组相同长度
        if len(idx_set) != len(x_set):
            if len(x_set) == 1:
                x_set = np.full(len(idx_set), x_set[0], dtype=self.dtype)
            else:
                raise ValueError("Need len(index)==len(data) or len(data)==1")
        # 遍历展平后的索引数组和值数组
        for i, v in zip(idx_set, x_set):
            # 如果值不为0，则将其设置到内部字典中对应的索引位置
            if v:
                self._dict[i] = v
            # 如果值为0且索引在字典中存在，则从字典中删除该索引
            elif i in self._dict:
                del self._dict[i]

    # 二维整数索引设置方法，根据行和列索引以及值更新内部字典
    def _set_intXint(self, row, col, x):
        key = (row, col)
        # 如果值不为0，则将其设置到内部字典中对应的键位置
        if x:
            self._dict[key] = x
        # 如果值为0且键在字典中存在，则从字典中删除该键
        elif key in self._dict:
            del self._dict[key]

    # 二维数组索引设置方法，根据行列数组索引和值数组更新内部字典
    def _set_arrayXarray(self, row, col, x):
        # 将行和列数组展平为一维整数列表
        row = list(map(int, row.ravel()))
        col = list(map(int, col.ravel()))
        # 将值数组展平为一维数组
        x = x.ravel()
        # 使用zip函数更新内部字典，将行列对作为键，对应的值作为值
        self._dict.update(zip(zip(row, col), x))

        # 遍历值为0的位置
        for i in np.nonzero(x == 0)[0]:
            key = (row[i], col[i])
            # 如果内部字典中对应位置的值也为0，则可能已被后续更新替代，因此从字典中删除该键
            if self._dict[key] == 0:
                del self._dict[key]
    # 定义特殊方法 __add__，用于实现稀疏矩阵与标量的加法
    def __add__(self, other):
        # 检查是否为标量类型
        if isscalarlike(other):
            # 根据当前对象的数据类型和另一个标量的数据类型，确定结果的数据类型
            res_dtype = upcast_scalar(self.dtype, other)
            # 创建一个新的稀疏矩阵容器，使用指定的形状和数据类型
            new = self._dok_container(self.shape, dtype=res_dtype)
            # 将该标量加到每个元素上
            for key in itertools.product(*[range(d) for d in self.shape]):
                aij = self._dict.get(key, 0) + other
                # 如果结果不为零，则存储到新的稀疏矩阵容器中
                if aij:
                    new[key] = aij
        # 如果 other 是稀疏矩阵类型
        elif issparse(other):
            # 检查两个矩阵的形状是否相等
            if other.shape != self.shape:
                raise ValueError("Matrix dimensions are not equal.")
            # 根据两个矩阵的数据类型，确定结果的数据类型
            res_dtype = upcast(self.dtype, other.dtype)
            # 创建一个新的稀疏矩阵容器，使用指定的形状和数据类型
            new = self._dok_container(self.shape, dtype=res_dtype)
            # 复制当前对象的字典内容到新的稀疏矩阵容器中
            new._dict = self._dict.copy()
            # 如果 other 是以 "dok" 格式存储的稀疏矩阵
            if other.format == "dok":
                o_items = other.items()
            else:
                # 将 other 转换为 COO 格式的稀疏矩阵
                other = other.tocoo()
                # 如果当前对象是一维的稀疏矩阵
                if self.ndim == 1:
                    o_items = zip(other.coords[0], other.data)
                else:
                    o_items = zip(zip(*other.coords), other.data)
            # 使用忽略溢出的错误状态，更新新稀疏矩阵容器的字典
            with np.errstate(over='ignore'):
                new._dict.update((k, new[k] + v) for k, v in o_items)
        # 如果 other 是密集矩阵类型
        elif isdense(other):
            # 将当前稀疏矩阵转换为密集矩阵后与 other 相加
            new = self.todense() + other
        else:
            # 若无法处理，则返回 Not Implemented
            return NotImplemented
        # 返回计算后的结果稀疏矩阵
        return new

    # 定义特殊方法 __radd__，实现右加法，即 addition 是可交换的
    def __radd__(self, other):
        return self + other  # addition is comutative

    # 定义特殊方法 __neg__，实现取负操作
    def __neg__(self):
        # 如果当前稀疏矩阵的数据类型是布尔类型，抛出未实现错误
        if self.dtype.kind == 'b':
            raise NotImplementedError(
                'Negating a sparse boolean matrix is not supported.'
            )
        # 创建一个新的稀疏矩阵容器，使用指定的形状和数据类型
        new = self._dok_container(self.shape, dtype=self.dtype)
        # 对当前稀疏矩阵容器中的每个元素取负，并更新到新的稀疏矩阵容器中
        new._dict.update((k, -v) for k, v in self.items())
        # 返回计算后的结果稀疏矩阵
        return new

    # 定义内部方法 _mul_scalar，实现稀疏矩阵与标量的乘法
    def _mul_scalar(self, other):
        # 根据当前对象的数据类型和另一个标量的数据类型，确定结果的数据类型
        res_dtype = upcast_scalar(self.dtype, other)
        # 创建一个新的稀疏矩阵容器，使用指定的形状和数据类型
        new = self._dok_container(self.shape, dtype=res_dtype)
        # 对当前稀疏矩阵容器中的每个元素乘以指定的标量，并更新到新的稀疏矩阵容器中
        new._dict.update(((k, v * other) for k, v in self.items()))
        # 返回计算后的结果稀疏矩阵
        return new

    # 定义内部方法 _matmul_vector，实现稀疏矩阵与向量的矩阵乘法
    def _matmul_vector(self, other):
        # 根据当前对象的数据类型和另一个向量的数据类型，确定结果的数据类型
        res_dtype = upcast(self.dtype, other.dtype)

        # 如果是向量与向量的矩阵乘法
        if self.ndim == 1:
            # 如果 other 是稀疏矩阵类型
            if issparse(other):
                # 如果 other 是以 "dok" 格式存储的稀疏矩阵
                if other.format == "dok":
                    # 找到当前稀疏矩阵和 other 矩阵共同拥有的键
                    keys = self.keys() & other.keys()
                else:
                    # 将 other 转换为 COO 格式的稀疏矩阵
                    keys = self.keys() & other.tocoo().coords[0]
                # 返回两个向量的点积，并使用结果的数据类型进行类型转换
                return res_dtype(sum(self._dict[k] * other._dict[k] for k in keys))
            # 如果 other 是密集矩阵类型
            elif isdense(other):
                # 返回两个向量的点积，并使用结果的数据类型进行类型转换
                return res_dtype(sum(other[k] * v for k, v in self.items()))
            else:
                # 若无法处理，则返回 Not Implemented
                return NotImplemented

        # 如果是矩阵与向量的矩阵乘法
        result = np.zeros(self.shape[0], dtype=res_dtype)
        # 遍历当前稀疏矩阵容器中的每个键值对
        for (i, j), v in self.items():
            # 计算矩阵乘法的结果，并更新到结果矩阵中
            result[i] += v * other[j]
        # 返回计算后的结果矩阵
        return result
    # 矩阵或向量与多向量的乘法操作，返回结果的数据类型由输入矩阵和向量的数据类型决定
    def _matmul_multivector(self, other):
        result_dtype = upcast(self.dtype, other.dtype)
        
        # 如果是向量 @ 多向量的操作
        if self.ndim == 1:
            # 对于其它的一维或二维情况，计算内积
            return sum(v * other[j] for j, v in self._dict.items())

        # 如果是矩阵 @ 多向量的操作
        M = self.shape[0]
        new_shape = (M,) if other.ndim == 1 else (M, other.shape[1])
        result = np.zeros(new_shape, dtype=result_dtype)
        
        # 遍历矩阵的每个元素 (i, j)，并将结果累加到对应位置
        for (i, j), v in self.items():
            result[i] += v * other[j]
        
        return result

    # 原地乘法操作，如果 other 是标量，则将自身的每个元素乘以 other
    def __imul__(self, other):
        if isscalarlike(other):
            self._dict.update((k, v * other) for k, v in self.items())
            return self
        return NotImplemented

    # 除法操作，如果 other 是标量，则将自身的每个元素除以 other
    def __truediv__(self, other):
        if isscalarlike(other):
            # 确定结果的数据类型
            res_dtype = upcast_scalar(self.dtype, other)
            # 创建新的对象，每个元素除以 other
            new = self._dok_container(self.shape, dtype=res_dtype)
            new._dict.update(((k, v / other) for k, v in self.items()))
            return new
        # 如果 other 不是标量，则调用 tocsr 方法进行除法操作
        return self.tocsr() / other

    # 原地真除法操作，如果 other 是标量，则将自身的每个元素除以 other
    def __itruediv__(self, other):
        if isscalarlike(other):
            self._dict.update((k, v / other) for k, v in self.items())
            return self
        return NotImplemented

    # 序列化方法，用于将对象转换为可序列化的字典形式
    def __reduce__(self):
        # 使用 dict 类的 __reduce__ 方法进行序列化
        return dict.__reduce__(self)

    # 返回对角线元素，仅支持二维结构，否则引发 ValueError 异常
    def diagonal(self, k=0):
        if self.ndim == 2:
            return super().diagonal(k)
        raise ValueError("diagonal requires two dimensions")

    # 转置方法，如果是一维结构则返回副本，如果是二维则进行转置
    def transpose(self, axes=None, copy=False):
        if self.ndim == 1:
            return self.copy()

        # 检查 axes 参数，对于稀疏数组/矩阵不支持指定轴的调换
        if axes is not None and axes != (1, 0):
            raise ValueError(
                "Sparse arrays/matrices do not support "
                "an 'axes' parameter because swapping "
                "dimensions is the only logical permutation."
            )

        M, N = self.shape
        # 创建新的对象，进行转置操作
        new = self._dok_container((N, M), dtype=self.dtype, copy=copy)
        # 将当前对象的每个元素 ((left, right), val) 进行调换后更新到新对象
        new._dict.update((((right, left), val) for (left, right), val in self.items()))
        return new

    # 将 _spbase.transpose.__doc__ 的文档字符串赋给 transpose 方法的文档字符串
    transpose.__doc__ = _spbase.transpose.__doc__
    def conjtransp(self):
        """DEPRECATED: Return the conjugate transpose.

        .. deprecated:: 1.14.0

            `conjtransp` is deprecated and will be removed in v1.16.0.
            Use ``.T.conj()`` instead.
        """
        # 构建警告信息，指出该方法即将被移除
        msg = ("`conjtransp` is deprecated and will be removed in v1.16.0. "
                   "Use `.T.conj()` instead.")
        # 发出警告，指示调用方停止使用该方法
        warn(msg, DeprecationWarning, stacklevel=2)

        # 如果是一维数组，将其转换为 COO 格式，并对数据进行共轭处理
        if self.ndim == 1:
            new = self.tocoo()
            new.data = new.data.conjugate()
            return new

        # 获取数组的形状
        M, N = self.shape
        # 创建一个新的 DOK 格式容器，交换行列，并对值进行共轭处理
        new = self._dok_container((N, M), dtype=self.dtype)
        new._dict = {(right, left): np.conj(val) for (left, right), val in self.items()}
        return new

    def copy(self):
        # 创建当前对象的一个副本
        new = self._dok_container(self.shape, dtype=self.dtype)
        # 更新副本的字典内容
        new._dict.update(self._dict)
        return new

    # 将 copy 方法的文档字符串设置为 _spbase.copy 方法的文档字符串
    copy.__doc__ = _spbase.copy.__doc__

    @classmethod
    def fromkeys(cls, iterable, value=1, /):
        # 使用给定的可迭代对象和值创建一个临时字典
        tmp = dict.fromkeys(iterable, value)
        # 确定稀疏矩阵的形状
        if isinstance(next(iter(tmp)), tuple):
            shape = tuple(max(idx) + 1 for idx in zip(*tmp))
        else:
            shape = (max(tmp) + 1,)
        # 使用形状和值的类型创建一个新的稀疏矩阵对象
        result = cls(shape, dtype=type(value))
        result._dict = tmp
        return result

    def tocoo(self, copy=False):
        # 获取非零元素的数量
        nnz = self.nnz
        # 如果没有非零元素，返回一个形状为 self.shape 的 COO 格式容器
        if nnz == 0:
            return self._coo_container(self.shape, dtype=self.dtype)

        # 确定索引的数据类型
        idx_dtype = self._get_index_dtype(maxval=max(self.shape))
        # 从值中获取数据数组
        data = np.fromiter(self.values(), dtype=self.dtype, count=nnz)
        # 如果是二维数组，以元组的方式处理键
        inds = zip(*self.keys()) if self.ndim > 1 else (self.keys(),)
        # 对键进行类型转换
        coords = tuple(np.fromiter(ix, dtype=idx_dtype, count=nnz) for ix in inds)
        # 使用数据和坐标创建一个 COO 格式容器
        A = self._coo_container((data, coords), shape=self.shape, dtype=self.dtype)
        A.has_canonical_format = True
        return A

    # 将 tocoo 方法的文档字符串设置为 _spbase.tocoo 方法的文档字符串
    tocoo.__doc__ = _spbase.tocoo.__doc__

    def todok(self, copy=False):
        # 如果 copy 参数为 True，则返回当前对象的一个副本
        if copy:
            return self.copy()
        # 否则直接返回当前对象
        return self

    # 将 todok 方法的文档字符串设置为 _spbase.todok 方法的文档字符串
    todok.__doc__ = _spbase.todok.__doc__

    def tocsc(self, copy=False):
        # 如果是一维数组，抛出 NotImplementedError 异常
        if self.ndim == 1:
            raise NotImplementedError("tocsr() not valid for 1d sparse array")
        # 否则先转换为 COO 格式，再转换为 CSC 格式
        return self.tocoo(copy=False).tocsc(copy=copy)

    # 将 tocsc 方法的文档字符串设置为 _spbase.tocsc 方法的文档字符串
    tocsc.__doc__ = _spbase.tocsc.__doc__
    # 定义 resize 方法，用于调整数组的形状
    def resize(self, *shape):
        # 检查是否为 sparray 类型的实例
        is_array = isinstance(self, sparray)
        # 调用 check_shape 函数，验证并返回合法的形状参数
        shape = check_shape(shape, allow_1d=is_array)
        # 如果给定的形状维度与当前数组维度不同，抛出未实现错误
        if len(shape) != len(self.shape):
            raise NotImplementedError

        # 如果数组为一维
        if self.ndim == 1:
            # 获取新的数组长度
            newN = shape[-1]
            # 删除超出新长度的所有元素
            for i in list(self._dict):
                if i >= newN:
                    del self._dict[i]
            # 更新数组的形状
            self._shape = shape
            return

        # 如果数组为二维及以上
        newM, newN = shape
        M, N = self.shape
        # 如果新形状的行数或列数小于当前数组的行数或列数
        if newM < M or newN < N:
            # 删除超出新形状范围的所有元素
            for i, j in list(self.keys()):
                if i >= newM or j >= newN:
                    del self._dict[i, j]
        # 更新数组的形状
        self._shape = shape

    # 设置 resize 方法的文档字符串为 _spbase.resize 方法的文档字符串
    resize.__doc__ = _spbase.resize.__doc__

    # 为了避免从 _base.py 导入 `tocsr`，添加了这个方法用于一维数组
    def astype(self, dtype, casting='unsafe', copy=True):
        # 将 dtype 转换为 np.dtype 类型
        dtype = np.dtype(dtype)
        # 如果当前数组的数据类型与指定的 dtype 不同
        if self.dtype != dtype:
            # 创建一个新的 sparray 对象，数据类型为指定的 dtype
            result = self._dok_container(self.shape, dtype=dtype)
            # 将当前字典的值转换为指定 dtype 的 numpy 数组，并与键对应起来
            data = np.array(list(self._dict.values()), dtype=dtype)
            result._dict = dict(zip(self._dict, data))
            return result
        # 如果 copy 参数为 True，则返回当前数组的副本
        elif copy:
            return self.copy()
        # 否则返回当前数组本身
        return self
# 判断输入对象是否为 dok_matrix 类型
def isspmatrix_dok(x):
    """Is `x` of dok_array type?

    Parameters
    ----------
    x
        object to check for being a dok matrix

    Returns
    -------
    bool
        True if `x` is a dok matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import dok_array, dok_matrix, coo_matrix, isspmatrix_dok
    >>> isspmatrix_dok(dok_matrix([[5]]))
    True
    >>> isspmatrix_dok(dok_array([[5]]))
    False
    >>> isspmatrix_dok(coo_matrix([[5]]))
    False
    """
    # 使用 isinstance 判断 x 是否为 dok_matrix 类型
    return isinstance(x, dok_matrix)


# This namespace class separates array from matrix with isinstance
class dok_array(_dok_base, sparray):
    """
    Dictionary Of Keys based sparse array.

    This is an efficient structure for constructing sparse
    arrays incrementally.

    This can be instantiated in several ways:
        dok_array(D)
            where D is a 2-D ndarray

        dok_array(S)
            with another sparse array or matrix S (equivalent to S.todok())

        dok_array((M,N), [dtype])
            create the array with initial shape (M,N)
            dtype is optional, defaulting to dtype='d'

    Attributes
    ----------
    dtype : dtype
        Data type of the array
    shape : 2-tuple
        Shape of the array
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    size
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    - Allows for efficient O(1) access of individual elements.
    - Duplicates are not allowed.
    - Can be efficiently converted to a coo_array once constructed.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import dok_array
    >>> S = dok_array((5, 5), dtype=np.float32)
    >>> for i in range(5):
    ...     for j in range(5):
    ...         S[i, j] = i + j    # Update element

    """


class dok_matrix(spmatrix, _dok_base):
    """
    Dictionary Of Keys based sparse matrix.

    This is an efficient structure for constructing sparse
    matrices incrementally.

    This can be instantiated in several ways:
        dok_matrix(D)
            where D is a 2-D ndarray

        dok_matrix(S)
            with another sparse array or matrix S (equivalent to S.todok())

        dok_matrix((M,N), [dtype])
            create the matrix with initial shape (M,N)
            dtype is optional, defaulting to dtype='d'

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    size
    T

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    - Allows for efficient O(1) access of individual elements.
    """
    def set_shape(self, shape):
        """
        Set the shape of the sparse matrix to a new shape.

        Parameters
        ----------
        shape : tuple
            New shape for the sparse matrix.

        Returns
        -------
        None

        Notes
        -----
        This method reshapes the matrix to the specified shape without copying data.
        """
        # Reshape the matrix to the new shape and convert it to the current format
        new_matrix = self.reshape(shape, copy=False).asformat(self.format)
        # Update the current instance's dictionary to that of the new_matrix
        self.__dict__ = new_matrix.__dict__

    def get_shape(self):
        """
        Get the shape of the sparse matrix.

        Returns
        -------
        tuple
            Shape of the sparse matrix.
        """
        return self._shape

    shape = property(fget=get_shape, fset=set_shape)

    def __reversed__(self):
        """
        Return a reverse iterator over the rows of the matrix.

        Returns
        -------
        iterator
            Iterator over the rows of the matrix in reverse order.
        """
        return self._dict.__reversed__()

    def __or__(self, other):
        """
        Override the bitwise OR operator for the sparse matrix.

        Parameters
        ----------
        other : _dok_base or object
            Another sparse matrix or an object to perform OR operation with.

        Returns
        -------
        dict
            Dictionary representing the OR operation result between the matrices.
            If `other` is not of type _dok_base, returns OR result with `other`.

        Notes
        -----
        This method performs element-wise OR operation between two sparse matrices
        or between the sparse matrix and another object.
        """
        if isinstance(other, _dok_base):
            return self._dict | other._dict
        return self._dict | other

    def __ror__(self, other):
        """
        Override the bitwise OR operator when the sparse matrix is on the right side.

        Parameters
        ----------
        other : _dok_base or object
            Another sparse matrix or an object.

        Returns
        -------
        dict
            Dictionary representing the OR operation result between the matrices
            or between the sparse matrix and `other`.

        Notes
        -----
        This method performs element-wise OR operation when the sparse matrix is on
        the right side of the OR operator.
        """
        if isinstance(other, _dok_base):
            return self._dict | other._dict
        return self._dict | other

    def __ior__(self, other):
        """
        Override the in-place OR operator for the sparse matrix.

        Parameters
        ----------
        other : _dok_base or object
            Another sparse matrix or an object to perform in-place OR operation with.

        Returns
        -------
        _dok_base
            Reference to the updated sparse matrix after in-place OR operation.

        Notes
        -----
        This method performs in-place element-wise OR operation between two sparse matrices
        or between the sparse matrix and another object.
        """
        if isinstance(other, _dok_base):
            self._dict |= other._dict
        else:
            self._dict |= other
        return self
```