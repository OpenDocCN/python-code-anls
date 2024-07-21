# `.\pytorch\functorch\dim\reference.py`

```
# 导入torch模块，用于张量操作
import torch

# 导入functorch._C模块中的dim别名_C
from functorch._C import dim as _C

# 导入op_properties模块中的全部内容
from . import op_properties

# 导入batch_tensor模块中的_enable_layers函数
from .batch_tensor import _enable_layers

# 导入tree_map模块中的tree_flatten和tree_map函数
from .tree_map import tree_flatten, tree_map

# 将_C.DimList赋值给DimList变量
DimList = _C.DimList

# 导入operator模块
import operator

# 导入functools模块中的reduce函数
from functools import reduce

# 使用集合pointwise存储op_properties.pointwise的内容，避免为集合编写C++绑定
pointwise = set(op_properties.pointwise)


# 定义函数prod，计算可迭代对象x中所有元素的乘积
def prod(x):
    return reduce(operator.mul, x, 1)


# 定义函数_wrap_dim，根据条件_wrap_dim封装维度
def _wrap_dim(d, N, keepdim):
    from . import Dim

    # 如果d是Dim类型，则保持维度或抛出异常
    if isinstance(d, Dim):
        assert not keepdim, "cannot preserve first-class dimensions with keepdim=True"
        return d
    # 如果d大于等于0，则减去N
    elif d >= 0:
        return d - N
    # 否则返回d
    else:
        return d


# 定义函数_dims，根据条件_dims确定维度
def _dims(d, N, keepdim, single_dim):
    from . import Dim

    # 如果d是Dim或int类型，则返回包含_wrap_dim处理后结果的ltuple
    if isinstance(d, (Dim, int)):
        return ltuple((_wrap_dim(d, N, keepdim),))
    # 否则抛出异常
    assert not single_dim, f"expected a single dimension or int but found: {d}"
    return ltuple(_wrap_dim(x, N, keepdim) for x in d)


# 定义函数_bind_dims_to_size，将维度绑定到尺寸
def _bind_dims_to_size(lhs_size, rhs, lhs_debug):
    from . import DimensionMismatchError

    # 找出未绑定的维度
    not_bound = tuple((i, r) for i, r in enumerate(rhs) if not r.is_bound)

    # 如果只有一个未绑定维度
    if len(not_bound) == 1:
        idx, d = not_bound[0]
        rhs_so_far = prod(r.size for r in rhs if r.is_bound)
        # 如果lhs_size不能被rhs_so_far整除，则抛出异常
        if lhs_size % rhs_so_far != 0:
            rhs_s = tuple("?" if not r.is_bound else str(r.size) for r in rhs)
            raise DimensionMismatchError(
                f"inferred dimension does not evenly fit into larger dimension: {lhs_size} vs {rhs_s}"
            )
        # 计算新的尺寸并赋值给d.size
        new_size = lhs_size // rhs_so_far
        d.size = new_size
    # 如果有多个未绑定维度，则抛出异常
    elif len(not_bound) > 1:
        rhs_s = tuple("?" if not r.is_bound else str(r.size) for r in rhs)
        raise DimensionMismatchError(
            f"cannot infer the size of two dimensions at once: {rhs} with sizes {rhs_s}"
        )
    else:
        # 计算rhs的总尺寸并检查是否与lhs_size相等，不等则抛出异常
        rhs_size = prod(r.size for r in rhs)
        if lhs_size != rhs_size:
            raise DimensionMismatchError(
                f"Dimension sizes to do not match ({lhs_size} != {rhs_size}) when matching {lhs_debug} to {rhs}"
            )


# 定义函数_tensor_levels，根据输入类型返回相应的值或元组
def _tensor_levels(inp):
    from . import _Tensor

    # 如果inp是_Tensor类型，则返回inp的属性值构成的元组
    if isinstance(inp, _Tensor):
        return inp._tensor, llist(inp._levels), inp._has_device
    # 否则返回inp、范围内的列表和True
    else:
        return inp, llist(range(-inp.ndim, 0)), True


# 定义函数_match_levels，匹配维度等级
def _match_levels(v, from_levels, to_levels):
    view = []
    permute = []
    requires_view = False
    size = v.size()
    # 遍历to_levels中的每个维度
    for t in to_levels:
        try:
            # 尝试在from_levels中找到t的索引
            idx = from_levels.index(t)
            permute.append(idx)
            view.append(size[idx])
        # 如果找不到t，则添加1到view，并设置requires_view为True
        except ValueError:
            view.append(1)
            requires_view = True
    # 如果permute不是[0, 1, ..., len(permute)-1]，则对v进行维度置换
    if permute != list(range(len(permute))):
        v = v.permute(*permute)
    # 如果requires_view为True，则对v进行视图调整
    if requires_view:
        v = v.view(*view)
    # 返回变量 v 的值作为函数的结果
    return v
# 定义一个方法，用于将一个单维度位置化，但不进行置换，
# 主要用于多张量操作中，希望操作的维度在物理上尽可能不移动
def _positional_no_permute(self, dim, expand_dim=False):
    # 导入必要的模块和类
    from . import Tensor
    
    # 获取当前对象的张量和级别列表
    ptensor, levels = self._tensor, llist(self._levels)
    
    # 尝试获取维度在级别列表中的索引
    try:
        idx = levels.index(dim)
    except ValueError:
        # 如果未找到维度并且允许扩展维度，则在索引 0 处插入新的维度
        if not expand_dim:
            raise
        idx = 0
        ptensor = ptensor.expand(dim.size, *ptensor.size())
        levels.insert(0, 0)
    
    # 计算在当前维度之前的整数级别数量
    idx_batched = 0
    for i in range(idx):
        if isinstance(levels[i], int):
            levels[i] -= 1
            idx_batched += 1
    
    # 将当前维度的级别设置为负数，表示已处理
    levels[idx] = -idx_batched - 1
    
    # 使用 Tensor 类的方法从位置化的张量和级别创建新的 Tensor 对象，并保留批处理信息
    return Tensor.from_positional(ptensor, levels, self._has_device), idx_batched


# 定义一个函数，用于检查两个对象是否相同类型的 Dim 对象
def seq(a, b):
    # 导入 Dim 类
    from . import Dim
    
    # 如果 a 和 b 都是 Dim 类的实例，则直接比较它们的引用
    if isinstance(a, Dim) != isinstance(b, Dim):
        return False
    if isinstance(a, Dim):
        return a is b
    else:
        return a == b


# 定义一个类 isin，实现了 __contains__ 和 index 方法，用于检查对象是否在其中
class isin:
    def __contains__(self, item):
        # 遍历实例中的对象，如果找到与 item 相同的对象，则返回 True
        for x in self:
            if seq(item, x):
                return True
        # 如果没有找到匹配的对象，则返回 False
        return False

    def index(self, item):
        # 遍历实例中的对象，如果找到与 item 相同的对象，则返回其索引值
        for i, x in enumerate(self):
            if seq(item, x):
                return i
        # 如果没有找到匹配的对象，则抛出 ValueError 异常
        raise ValueError


# 定义 llist 类，继承自 isin 和 list，表示具有顺序和包含检查功能的列表
class llist(isin, list):
    pass


# 定义 ltuple 类，继承自 isin 和 tuple，表示具有顺序和包含检查功能的元组
class ltuple(isin, tuple):
    pass


# 定义一个空字典 empty_dict
empty_dict = {}


# 定义一个类方法 __torch_function__，用于处理 Torch 函数调用
@classmethod
def __torch_function__(self, orig, cls, args, kwargs=empty_dict):
    # 导入必要的类和函数
    from . import _Tensor, Tensor, TensorLike
    from .delayed_mul_tensor import DelayedMulTensor
    
    # 如果调用的是 torch.Tensor 的乘法方法
    if orig is torch.Tensor.__mul__:
        lhs, rhs = args
        # 如果 lhs 和 rhs 都是 _Tensor 类的实例，且维度均为 0
        if (
            isinstance(lhs, _Tensor)
            and isinstance(rhs, _Tensor)
            and lhs.ndim == 0
            and rhs.ndim == 0
        ):
            # 返回 DelayedMulTensor 对象
            return DelayedMulTensor(lhs, rhs)
    
    # 创建一个空的级别列表 all_dims
    all_dims = llist()
    
    # 将参数 args 和 kwargs 扁平化，同时获取用于重构的函数 unflatten
    flat_args, unflatten = tree_flatten((args, kwargs))
    
    # 初始化一个变量用于保存包含张量的设备信息
    device_holding_tensor = None
    
    # 遍历扁平化后的参数列表
    for f in flat_args:
        if isinstance(f, _Tensor):
            # 如果当前参数是 _Tensor 类的实例，并且具有设备信息，则记录该设备信息
            if f._has_device:
                device_holding_tensor = f._batchtensor
            # 遍历张量的维度，并将未出现过的维度添加到 all_dims 中
            for d in f.dims:
                if d not in all_dims:
                    all_dims.append(d)
    
    # 定义一个内部函数 unwrap，用于处理 Tensor 对象的包装
    def unwrap(t):
        if isinstance(t, _Tensor):
            r = t._batchtensor
            # 如果存在包含张量的设备信息且当前张量没有设备信息，则将其移至相同设备上
            if device_holding_tensor is not None and not t._has_device:
                r = r.to(device=device_holding_tensor.device)
            return r
        return t
    
    # 如果原始函数在pointwise中存在，则执行以下操作
    if orig in pointwise:
        # 创建一个新的链表来存储结果级别和参数级别
        result_levels = llist()
        arg_levels = llist()
        # 初始化一个空列表用于存储需要扩展的参数索引和级别
        to_expand = []
        # 遍历扁平化的参数列表及其索引
        for i, f in enumerate(flat_args):
            # 如果参数f是TensorLike类型
            if isinstance(f, TensorLike):
                # 获取张量、级别和设备信息
                ptensor, levels, _ = _tensor_levels(f)
                # 如果f是_Tensor类型且没有设备信息，并且有指定的持有设备信息
                if (
                    isinstance(f, _Tensor)
                    and not f._has_device
                    and device_holding_tensor is not None
                ):
                    # 将张量移动到指定的设备
                    ptensor = ptensor.to(device=device_holding_tensor.device)
                # 更新flat_args中的对应位置的参数为处理后的张量
                flat_args[i] = ptensor
                # 将levels中的级别添加到result_levels中（去重）
                for l in levels:
                    if l not in result_levels:
                        result_levels.append(l)
                # 将需要扩展的参数索引及其级别添加到to_expand列表中
                to_expand.append((i, levels))

        # 对需要扩展的参数进行级别匹配
        for i, levels in to_expand:
            flat_args[i] = _match_levels(flat_args[i], levels, result_levels)
        
        # 将flat_args解扁平化为args和kwargs
        args, kwargs = unflatten(flat_args)
        # 调用原始函数并获取结果
        result = orig(*args, **kwargs)

        # 定义一个函数wrap，用于处理结果中的张量
        def wrap(t):
            # 如果t是TensorLike类型，则根据结果级别创建张量
            if isinstance(t, TensorLike):
                return Tensor.from_positional(
                    t, result_levels, device_holding_tensor is not None
                )
            # 否则直接返回t
            return t

        # 对结果应用wrap函数并返回处理后的结果
        return tree_map(wrap, result)
    
    # 如果原始函数不在pointwise中
    else:
        # 定义一个函数wrap，用于处理结果中的张量
        def wrap(t):
            # 如果t是TensorLike类型，则根据批处理创建张量
            if isinstance(t, TensorLike):
                return Tensor.from_batched(t, device_holding_tensor is not None)
            # 否则直接返回t
            return t

        # 启用所有维度的图层操作
        with _enable_layers(all_dims):
            # 打印调试信息，指示对原始函数进行批处理
            print(f"batch_tensor for {orig}")
            # 解包扁平化参数并生成args和kwargs
            args, kwargs = unflatten(unwrap(f) for f in flat_args)
            # 调用原始函数并获取结果
            result = orig(*args, **kwargs)
            # 对结果应用wrap函数并返回处理后的结果
            return tree_map(wrap, result)
# 定义一个方法，接受可变数量的位置参数dims，表示张量的维度
def positional(self, *dims):
    # 从当前目录导入Dim、DimensionBindError、Tensor
    from . import Dim, DimensionBindError, Tensor
    
    # 获取当前对象的_tensor和_levels属性，并分别赋给ptensor和levels变量
    ptensor, levels = self._tensor, llist(self._levels)
    
    # 创建一个空的链表flat_dims
    flat_dims = llist()
    
    # 创建一个空列表view
    view = []
    
    # 初始化标志变量needs_view为False
    needs_view = False
    
    # 获取当前对象的维度数，并赋值给变量ndim
    ndim = self.ndim
    
    # 遍历dims中的每一个维度d
    for d in dims:
        # 如果d是DimList类型
        if isinstance(d, DimList):
            # 将d中的每个元素扩展到flat_dims
            flat_dims.extend(d)
            # 将d中每个元素的size添加到view中
            view.extend(e.size for e in d)
        # 如果d是Dim类型
        elif isinstance(d, Dim):
            # 将d添加到flat_dims
            flat_dims.append(d)
            # 将d的size添加到view中
            view.append(d.size)
        # 如果d是整数类型
        elif isinstance(d, int):
            # 将d用_wrap_dim函数处理后添加到flat_dims
            d = _wrap_dim(d, ndim, False)
            flat_dims.append(d)
            # 将ptensor在维度d上的size添加到view中
            view.append(ptensor.size(d))
        # 如果d是其他类型
        else:
            # 将d中的每个元素扩展到flat_dims
            flat_dims.extend(d)
            # 计算d中每个元素size的乘积，并添加到view中
            view.append(prod(e.size for e in d))
            # 设置needs_view标志为True
            needs_view = True
    
    # 创建一个排列列表permute，初始化为levels的索引列表
    permute = list(range(len(levels)))
    
    # 获取flat_dims的长度，赋值给nflat
    nflat = len(flat_dims)
    
    # 遍历flat_dims中的每一个元素d及其索引i
    for i, d in enumerate(flat_dims):
        # 尝试在levels中查找d的索引，并赋值给idx
        try:
            idx = levels.index(d)
        # 如果找不到，抛出DimensionBindError异常
        except ValueError as e:
            raise DimensionBindError(
                f"tensor of dimensions {self.dims} does not contain dim {d}"
            ) from e
        # 获取permute中idx处的值，并赋给p
        p = permute[idx]
        # 从levels和permute中删除索引idx处的元素
        del levels[idx]
        del permute[idx]
        # 将0插入到levels和permute的索引i处
        levels.insert(i, 0)
        permute.insert(i, p)
    
    # 对ptensor进行排列，排列顺序为permute中的元素顺序
    ptensor = ptensor.permute(*permute)
    
    # 初始化变量seen为0
    seen = 0
    
    # 从levels的最后一个元素开始遍历到第一个元素
    for i in range(len(levels) - 1, -1, -1):
        # 如果levels中的元素是整数
        if isinstance(levels[i], int):
            # seen加1
            seen += 1
            # 将levels中索引i处的元素设置为负数seen
            levels[i] = -seen
    
    # 使用Tensor.from_positional方法创建一个新的Tensor对象result
    result = Tensor.from_positional(ptensor, levels, self._has_device)
    
    # 如果needs_view为True
    if needs_view:
        # 使用result对象的reshape方法进行视图变换，保留前len(flat_dims)个维度
        result = result.reshape(*view, *result.size()[len(flat_dims):])
    
    # 返回result对象
    return result
    # 定义一个方法 fn，接受任意数量的位置参数和关键字参数
    def fn(self, *args, **kwargs):
        # 调用 _getarg 函数获取维度信息，处理可能的未提供参数情况
        dim = _getarg(dim_name, dim_offset, args, kwargs, _not_present)
        # 检查维度是否为 _not_present 或者 (single_dim 为 True 且维度不是 Dim 类型)
        if dim is _not_present or (single_dim and not isinstance(dim, Dim)):
            # 在当前维度上启用层次，并输出调试信息
            with _enable_layers(self.dims):
                print(f"dim fallback batch_tensor for {orig}")
                # 返回通过 Tensor.from_batched 构造的张量
                return Tensor.from_batched(
                    orig(self._batchtensor, *args, **kwargs), self._has_device
                )
        
        # 获取是否保持维度信息，根据 reduce 变量的值决定
        keepdim = (
            _getarg("keepdim", keepdim_offset, args, kwargs, False) if reduce else False
        )
        # 获取当前对象的张量和级别列表
        t, levels = self._tensor, llist(self._levels)
        # 根据给定的维度信息计算维度列表
        dims = _dims(dim, self._batchtensor.ndim, keepdim, single_dim)
        # 获取维度索引元组
        dim_indices = tuple(levels.index(d) for d in dims)
        
        # 如果进行了降维操作且不保持维度，则创建新级别列表
        if reduce and not keepdim:
            new_levels = [l for i, l in enumerate(levels) if i not in dim_indices]
        else:
            new_levels = levels
        
        # 如果维度索引元组长度为 1，将其转换为单一的索引值，确保对于只接受单一参数的维度能正常工作
        if len(dim_indices) == 1:
            dim_indices = dim_indices[0]
        
        # 将 dim_name、dim_offset 更新为 dim_indices，以备后续使用
        args = list(args)
        _patcharg(dim_name, dim_offset, args, kwargs, dim_indices)
        
        # 定义一个内部函数 wrap，根据输入的类型包装为 TensorLike 或直接返回
        def wrap(t):
            if isinstance(t, TensorLike):
                return Tensor.from_positional(t, new_levels, self._has_device)
            return t
        
        # 在新级别列表上启用层次，并输出调试信息
        with _enable_layers(new_levels):
            print(f"dim used batch_tensor for {orig}")
            # 调用原始函数 orig，并对其结果应用 tree_map(wrap) 处理后返回
            r = orig(t, *args, **kwargs)
            return tree_map(wrap, r)
    
    # 返回定义的函数 fn
    return fn
def _def(name, *args, **kwargs):
    # 导入自定义的 _Tensor 模块
    from . import _Tensor
    
    # 获取 torch.Tensor 中指定名称的原始方法
    orig = getattr(torch.Tensor, name)
    
    # 将原始方法替换为经过包装的方法，并设置到 _Tensor 模块中
    setattr(_Tensor, name, _wrap(orig, *args, **kwargs))


# 定义一个常量 no_slice，表示 slice(None)
no_slice = slice(None)

# 保存 torch.Tensor 的原始 __getitem__ 方法
_orig_getitem = torch.Tensor.__getitem__


# 定义一个维度跟踪器类 dim_tracker
class dim_tracker:
    def __init__(self):
        # 初始化维度列表和计数列表
        self.dims = llist()  # llist 是什么类型，需要进一步补充说明
        self.count = []

    # 记录维度的使用情况
    def record(self, d):
        # 如果维度 d 不在维度列表中，则添加到列表，并初始化计数
        if d not in self.dims:
            self.dims.append(d)
            self.count.append(1)

    # 根据维度 d 获取其使用次数
    def __getitem__(self, d):
        return self.count[self.dims.index(d)]


# 定义自定义的 __getitem__ 方法 t__getitem__
def t__getitem__(self, input):
    # 导入必要的模块和类
    from . import _Tensor, Dim, DimensionBindError, DimList, Tensor, TensorLike

    # 处理简单的索引情况
    is_simple = (
        not isinstance(input, Dim)  # input 不是 Dim 类型
        and not isinstance(input, (tuple, list))  # input 不是 tuple 或 list
        and not (  # 且不是 functorch bug 的特殊情况
            isinstance(input, TensorLike) and input.ndim == 0
        )
    )

    if is_simple:
        if isinstance(self, _Tensor):
            # 如果 self 是 _Tensor 类型，则调用 _Tensor 的 __torch_function__
            return _Tensor.__torch_function__(_orig_getitem, None, (self, input))
        else:
            # 否则调用原始的 __getitem__ 方法
            return _orig_getitem(self, input)

    # 进一步优化索引情况
    if not isinstance(input, tuple):
        input = [input]
    else:
        input = list(input)

    dims_indexed = 0  # 初始化索引的维度计数
    expanding_object = None  # 用于标记 ... 或未绑定的维度列表的索引
    dimlists = []  # 存储维度列表的索引位置

    # 遍历输入索引
    for i, s in enumerate(input):
        if s is ... or isinstance(s, DimList) and not s.is_bound:
            # 处理 ... 或未绑定的维度列表的情况
            if expanding_object is not None:
                # 如果已经有 expanding_object，则抛出异常
                msg = (
                    "at most one ... or unbound dimension list can exist in indexing list but"
                    f" found 2 at offsets {i} and {expanding_object}"
                )
                raise DimensionBindError(msg)
            expanding_object = i

        # 处理维度列表的情况
        if isinstance(s, DimList):
            dims_indexed += len(s) if s.is_bound else 0
            dimlists.append(i)
        elif s is not None and s is not ...:
            dims_indexed += 1

    ndim = self.ndim  # 获取 tensor 的维度数
    if dims_indexed > ndim:
        # 如果索引的维度数大于 tensor 的实际维度数，则抛出 IndexError
        raise IndexError(
            f"at least {dims_indexed} indices were supplied but the tensor only has {ndim} dimensions."
        )
    # 检查是否有需要扩展的对象
    if expanding_object is not None:
        # 计算需要扩展的维度数
        expanding_ndims = ndim - dims_indexed
        # 获取要索引的对象
        obj = input[expanding_object]
        # 如果对象是省略符号...
        if obj is ...:
            # 将输入中的扩展对象位置替换为 no_slice，复制 expanding_ndims 次
            input[expanding_object : expanding_object + 1] = [
                no_slice
            ] * expanding_ndims
        else:
            # 绑定对象的长度为 expanding_ndims
            obj.bind_len(expanding_ndims)

    # 将维度列表展平到索引中
    for i in reversed(dimlists):
        # 将输入中的索引 i 替换为 input[i]
        input[i : i + 1] = input[i]

    # 重置已索引的维度计数
    dims_indexed = 0
    # 标记不需要视图
    requires_view = False
    # 获取自身大小
    size = self.size()
    # 视图大小列表
    view_sizes = []
    # 维度跟踪器
    dims_seen = dim_tracker()

    # 定义一个函数用于添加维度信息
    def add_dims(t):
        # 如果 t 不是 _Tensor 类型，则返回
        if not isinstance(t, _Tensor):
            return
        # 遍历 t 的维度，记录到 dims_seen
        for d in t.dims:
            dims_seen.record(d)

    # 添加当前对象的维度信息
    add_dims(self)
    # 维度包列表
    dim_packs = []

    # 遍历输入中的索引
    for i, idx in enumerate(input):
        # 如果索引为 None
        if idx is None:
            # 将输入中的索引 i 替换为 no_slice
            input[i] = no_slice
            # 添加视图大小为 1
            view_sizes.append(1)
            # 标记需要视图
            requires_view = True
        else:
            # 获取当前维度的大小
            sz = size[dims_indexed]
            # 如果 idx 是 Dim 类型
            if isinstance(idx, Dim):
                # 设置 idx 的大小为 sz
                idx.size = sz
                # 记录 idx 到 dims_seen
                dims_seen.record(idx)
                # 添加视图大小为 sz
                view_sizes.append(sz)
            # 如果 idx 是 tuple 或 list 类型，并且不为空且第一个元素是 Dim 类型
            elif isinstance(idx, (tuple, list)) and idx and isinstance(idx[0], Dim):
                # 遍历 idx 中的每个维度
                for d in idx:
                    dims_seen.record(idx)
                # 将当前维度的大小绑定到 idx
                _bind_dims_to_size(sz, idx, f"offset {i}")
                # 添加视图大小为 idx 中每个维度的大小
                view_sizes.extend(d.size for d in idx)
                # 标记需要视图
                requires_view = True
                # 将当前索引 i 加入到维度包列表
                dim_packs.append(i)
            else:
                # 添加 idx 的维度信息
                add_dims(idx)
                # 添加视图大小为 sz
                view_sizes.append(sz)
            # 已索引的维度数加一
            dims_indexed += 1

    # 如果需要视图
    if requires_view:
        # 创建一个新的视图，大小为 view_sizes
        self = self.view(*view_sizes)

    # 逆序处理维度包列表
    for i in reversed(dim_packs):
        # 将输入中的索引 i 替换为 input[i]
        input[i : i + 1] = input[i]

    # 当前状态：
    # input 是扁平化的，包含 Dim、Tensor 或标准索引的有效内容
    # self 可能有一级维度信息

    # 要索引：
    # 从 self 中去掉一级维度，它们只成为它们位置的直接索引

    # 确定索引张量的维度：所有索引张量中的维度的并集。
    # 这些维度将出现并需要绑定到第一个张量出现的地方

    # 如果 self 是 _Tensor 类型
    if isinstance(self, _Tensor):
        # 获取 ptensor_self 和 levels
        ptensor_self, levels = self._tensor, list(self._levels)
        # 从 levels 中获取平坦化的输入
        input_it = iter(input)
        flat_inputs = [next(input_it) if isinstance(l, int) else l for l in levels]
        # 是否具有设备信息
        has_device = self._has_device
        # 需要填充的数量
        to_pad = 0
    else:
        # 获取 ptensor_self 和平坦化的输入 flat_inputs
        ptensor_self, flat_inputs = self, input
        # 需要填充的数量
        to_pad = ptensor_self.ndim - len(flat_inputs)
        # 标记具有设备信息
        has_device = True

    # 结果级别列表
    result_levels = []
    # 索引级别列表
    index_levels = []
    # 张量插入点
    tensor_insert_point = None
    # 要扩展的内容
    to_expand = {}
    # 是否需要获取索引
    requires_getindex = False
    # 对于输入列表中的每个元素及其索引进行遍历
    for i, inp in enumerate(flat_inputs):
        # 检查当前元素是否为维度对象且在dims_seen中只出现过一次
        if isinstance(inp, Dim) and dims_seen[inp] == 1:
            # 将当前元素替换为no_slice，并将其添加到result_levels列表中
            flat_inputs[i] = no_slice
            result_levels.append(inp)
        # 如果当前元素是类TensorLike的对象
        elif isinstance(inp, TensorLike):
            # 设置标志requires_getindex为True
            requires_getindex = True
            # 如果tensor_insert_point尚未设置，则设置为当前result_levels的长度
            if tensor_insert_point is None:
                tensor_insert_point = len(result_levels)
            # 获取inp对象的张量、层级和未使用的值
            ptensor, levels, _ = _tensor_levels(inp)
            # 将当前元素对应的levels存储在to_expand字典中
            to_expand[i] = levels
            # 更新flat_inputs中的当前元素为ptensor
            flat_inputs[i] = ptensor
            # 将levels中未出现在index_levels中的层级添加到index_levels中
            for l in levels:
                if l not in index_levels:
                    index_levels.append(l)
        # 如果当前元素不是Dim或TensorLike对象
        else:
            # 设置标志requires_getindex为True
            requires_getindex = True
            # 将0添加到result_levels中表示当前元素
            result_levels.append(0)

    # 如果tensor_insert_point已经设置
    if tensor_insert_point is not None:
        # 在result_levels的tensor_insert_point处插入index_levels中的层级
        result_levels[tensor_insert_point:tensor_insert_point] = index_levels

    # 遍历to_expand字典中的每个元素及其levels
    for i, levels in to_expand.items():
        # 使用_match_levels函数匹配flat_inputs中当前元素和其levels与index_levels的匹配
        flat_inputs[i] = _match_levels(flat_inputs[i], levels, index_levels)

    # 如果requires_getindex标志为True
    if requires_getindex:
        # 调用_orig_getitem函数获取ptensor_self和flat_inputs的结果
        result = _orig_getitem(ptensor_self, flat_inputs)
    else:
        # 否则将结果设置为ptensor_self
        result = ptensor_self

    # 初始化next_positional为-1
    next_positional = -1
    # 如果需要填充to_pad
    if to_pad > 0:
        # 将to_pad个0添加到result_levels末尾
        result_levels.extend([0] * to_pad)
    # 反向遍历result_levels中的每个元素及其索引
    for i, r in enumerate(reversed(result_levels)):
        # 如果当前元素是整数
        if isinstance(r, int):
            # 将result_levels中的倒数第i个元素设置为next_positional的值，然后next_positional减1
            result_levels[-1 - i] = next_positional
            next_positional -= 1

    # 返回使用result、result_levels和has_device参数构造的Tensor对象
    return Tensor.from_positional(result, result_levels, has_device)
# XXX - dim is optional and can be the outer-most dimension...
def stack(tensors, new_dim, dim=0, out=None):
    # 如果 dim 是整数类型，则使用 torch.stack 函数将 tensors 列表中的张量在 dim 维度上堆叠，并进行索引替换
    if isinstance(dim, int):
        return torch.stack(tensors, dim, out).index(dim, new_dim)
    index = None
    # 如果指定了输出张量 out，则调用 _positional_no_permute 函数处理 out，并进行扩展维度操作
    if out is not None:
        out, index = _positional_no_permute(out, dim, expand_dim=True)
    ptensors = []
    # 遍历 tensors 列表中的每个张量 t
    for t in tensors:
        # 使用 _positional_no_permute 函数处理张量 t，并进行扩展维度操作
        pt, pi = _positional_no_permute(t, dim, expand_dim=True)
        # 如果 index 不为空且 pi 不等于 index，则调用 move_dim 方法将 pt 在 pi 维度上移动到 index 维度上
        if index is not None and pi != index:
            pt = pt.move_dim(pi, index)
        else:
            index = pi
        ptensors.append(pt)
    # 使用 torch.stack 函数将处理后的 ptensors 列表中的张量在 index 维度上堆叠，结果存入 out 中
    pr = torch.stack(ptensors, index, out=out)
    # 使用 index 维度上的索引范围 (index, index + 1)，将结果张量 pr 在 new_dim 和 dim 维度上进行索引
    return pr.index((index, index + 1), (new_dim, dim))


_orig_split = torch.Tensor.split


def split(self, split_size_or_sections, dim=0):
    from . import _Tensor, Dim

    # 如果 split_size_or_sections 是整数类型，或者其中任何一个元素是整数类型
    if isinstance(split_size_or_sections, int) or any(
        isinstance(t, int) for t in split_size_or_sections
    ):
        # 如果 dim 是 Dim 对象，则抛出 ValueError 异常
        if isinstance(dim, Dim):
            raise ValueError(
                "when dim is specified as a Dim object, split sizes must also be dimensions."
            )
        # 调用 _orig_split 函数进行张量分割操作，返回结果
        return _orig_split(self, split_size_or_sections, dim=dim)

    # 如果 dim 是 Dim 对象，则确保 self 是 _Tensor 类型的张量，否则抛出异常
    if isinstance(dim, Dim):
        assert isinstance(self, _Tensor), f"Tensor does not have dimension {dim}"
        self, dim = _positional_no_permute(self, dim)

    # 获取 self 在 dim 维度上的大小
    size = self.size(dim)
    total_bound_size = 0
    unbound = []
    sizes = []
    # 遍历 split_size_or_sections 列表中的每个元素 d
    for i, d in enumerate(split_size_or_sections):
        # 如果 d 是已绑定的维度
        if d.is_bound:
            sizes.append(d.size)
            total_bound_size += d.size
        else:
            sizes.append(0)
            unbound.append(i)

    # 如果存在未绑定的维度
    if unbound:
        # 确保结果维度不大于原始维度 size
        assert (
            total_bound_size <= size
        ), f"result dimensions are larger than original: {total_bound_size} vs {size} ({split_size_or_sections})"
        remaining_size = size - total_bound_size
        # 计算每个未绑定维度的 chunk_size
        chunk_size = -(-remaining_size // len(unbound))
        for u in unbound:
            sz = min(chunk_size, remaining_size)
            # 将 split_size_or_sections[u] 的 size 设置为 sz
            split_size_or_sections[u].size = sz
            sizes[u] = sz
            remaining_size -= sz
    else:
        # 确保结果维度与原始维度 size 相等
        assert (
            total_bound_size == size
        ), f"result dimensions do not match original: {total_bound_size} vs {size} ({split_size_or_sections})"
    # 使用 _orig_split 函数将 self 按 sizes 列表在 dim 维度上进行分割，并返回分割后的张量元组
    return tuple(
        t.index(dim, d)
        for d, t in zip(split_size_or_sections, _orig_split(self, sizes, dim=dim))
    )
```