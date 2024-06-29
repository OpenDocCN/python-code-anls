# `D:\src\scipysrc\numpy\numpy\ma\__init__.pyi`

```py
# 导入 numpy._pytesttester 模块中的 PytestTester 类
from numpy._pytesttester import PytestTester

# 导入 numpy.ma 模块中的 extras 子模块
from numpy.ma import extras as extras

# 导入 numpy.ma.core 模块中的以下类和函数
from numpy.ma.core import (
    # 定义异常类 MAError 别名为 MAError
    MAError as MAError,
    # 定义异常类 MaskError 别名为 MaskError
    MaskError as MaskError,
    # 定义 MaskType 类别名为 MaskType
    MaskType as MaskType,
    # 定义 MaskedArray 类别名为 MaskedArray
    MaskedArray as MaskedArray,
    # 导入函数 abs 别名为 abs
    abs as abs,
    # 导入函数 absolute 别名为 absolute
    absolute as absolute,
    # 导入函数 add 别名为 add
    add as add,
    # 导入函数 all 别名为 all
    all as all,
    # 导入函数 allclose 别名为 allclose
    allclose as allclose,
    # 导入函数 allequal 别名为 allequal
    allequal as allequal,
    # 导入函数 alltrue 别名为 alltrue
    alltrue as alltrue,
    # 导入函数 amax 别名为 amax
    amax as amax,
    # 导入函数 amin 别名为 amin
    amin as amin,
    # 导入函数 angle 别名为 angle
    angle as angle,
    # 导入函数 anom 别名为 anom
    anom as anom,
    # 导入函数 anomalies 别名为 anomalies
    anomalies as anomalies,
    # 导入函数 any 别名为 any
    any as any,
    # 导入函数 append 别名为 append
    append as append,
    # 导入函数 arange 别名为 arange
    arange as arange,
    # 导入函数 arccos 别名为 arccos
    arccos as arccos,
    # 导入函数 arccosh 别名为 arccosh
    arccosh as arccosh,
    # 导入函数 arcsin 别名为 arcsin
    arcsin as arcsin,
    # 导入函数 arcsinh 别名为 arcsinh
    arcsinh as arcsinh,
    # 导入函数 arctan 别名为 arctan
    arctan as arctan,
    # 导入函数 arctan2 别名为 arctan2
    arctan2 as arctan2,
    # 导入函数 arctanh 别名为 arctanh
    arctanh as arctanh,
    # 导入函数 argmax 别名为 argmax
    argmax as argmax,
    # 导入函数 argmin 别名为 argmin
    argmin as argmin,
    # 导入函数 argsort 别名为 argsort
    argsort as argsort,
    # 导入函数 around 别名为 around
    around as around,
    # 导入函数 array 别名为 array
    array as array,
    # 导入函数 asanyarray 别名为 asanyarray
    asanyarray as asanyarray,
    # 导入函数 asarray 别名为 asarray
    asarray as asarray,
    # 导入函数 bitwise_and 别名为 bitwise_and
    bitwise_and as bitwise_and,
    # 导入函数 bitwise_or 别名为 bitwise_or
    bitwise_or as bitwise_or,
    # 导入函数 bitwise_xor 别名为 bitwise_xor
    bitwise_xor as bitwise_xor,
    # 导入 bool 类型别名为 bool
    bool as bool,
    # 导入函数 ceil 别名为 ceil
    ceil as ceil,
    # 导入函数 choose 别名为 choose
    choose as choose,
    # 导入函数 clip 别名为 clip
    clip as clip,
    # 导入函数 common_fill_value 别名为 common_fill_value
    common_fill_value as common_fill_value,
    # 导入函数 compress 别名为 compress
    compress as compress,
    # 导入函数 compressed 别名为 compressed
    compressed as compressed,
    # 导入函数 concatenate 别名为 concatenate
    concatenate as concatenate,
    # 导入函数 conjugate 别名为 conjugate
    conjugate as conjugate,
    # 导入函数 convolve 别名为 convolve
    convolve as convolve,
    # 导入函数 copy 别名为 copy
    copy as copy,
    # 导入函数 correlate 别名为 correlate
    correlate as correlate,
    # 导入函数 cos 别名为 cos
    cos as cos,
    # 导入函数 cosh 别名为 cosh
    cosh as cosh,
    # 导入函数 count 别名为 count
    count as count,
    # 导入函数 cumprod 别名为 cumprod
    cumprod as cumprod,
    # 导入函数 cumsum 别名为 cumsum
    cumsum as cumsum,
    # 导入函数 default_fill_value 别名为 default_fill_value
    default_fill_value as default_fill_value,
    # 导入函数 diag 别名为 diag
    diag as diag,
    # 导入函数 diagonal 别名为 diagonal
    diagonal as diagonal,
    # 导入函数 diff 别名为 diff
    diff as diff,
    # 导入函数 divide 别名为 divide
    divide as divide,
    # 导入函数 empty 别名为 empty
    empty as empty,
    # 导入函数 empty_like 别名为 empty_like
    empty_like as empty_like,
    # 导入函数 equal 别名为 equal
    equal as equal,
    # 导入函数 exp 别名为 exp
    exp as exp,
    # 导入函数 expand_dims 别名为 expand_dims
    expand_dims as expand_dims,
    # 导入函数 fabs 别名为 fabs
    fabs as fabs,
    # 导入函数 filled 别名为 filled
    filled as filled,
    # 导入函数 fix_invalid 别名为 fix_invalid
    fix_invalid as fix_invalid,
    # 导入函数 flatten_mask 别名为 flatten_mask
    flatten_mask as flatten_mask,
    # 导入函数 flatten_structured_array 别名为 flatten_structured_array
    flatten_structured_array as flatten_structured_array,
    # 导入函数 floor 别名为 floor
    floor as floor,
    # 导入函数 floor_divide 别名为 floor_divide
    floor_divide as floor_divide,
    # 导入函数 fmod 别名为 fmod
    fmod as fmod,
    # 导入函数 frombuffer 别名为 frombuffer
    frombuffer as frombuffer,
    # 导入函数 fromflex 别名为 fromflex
    fromflex as fromflex,
    # 导入函数 fromfunction 别名为 fromfunction
    fromfunction as fromfunction,
    # 导入函数 getdata 别名为 getdata
    getdata as getdata,
    # 导入函数 getmask 别名为 getmask
    getmask as getmask,
    # 导入函数 getmaskarray 别名为 getmaskarray
    getmaskarray as getmaskarray,
    # 导入函数 greater 别名为 greater
    greater as greater,
    # 导入函数 greater_equal 别名为 greater_equal
    greater_equal as greater_equal,
    # 导入函数 harden_mask 别名为 harden_mask
    harden_mask as harden_mask,
    # 导入函数 hypot 别名为 hypot
    hypot as hypot,
    # 导入函数 identity 别名为 identity
    identity as identity,
    # 导入函数 ids 别名为 ids
    ids as ids,
    # 导入函数 indices 别名为 indices
    indices as indices,
    # 导入函数 inner 别名为 inner
    inner as inner,
    # 导入函数 innerproduct 别名为 innerproduct
    innerproduct as innerproduct,
    # 导入函数 isMA 别名为 isMA
    isMA as isMA,
    # 导入函数 isMaskedArray 别名为 isMaskedArray
    isMaskedArray as isMaskedArray,
    # 导入函数 is_mask 别名为 is_mask
    is_mask as is_mask,
    # 导入函数 is_masked 别名为 is_masked
    is_masked as is_masked,
    # 导入函数 isarray 别名为 isarray
    isarray as isarray,
    # 导入函数 left_shift 别名为 left_shift
    left_shift as left_shift,
    # 导入函数 less 别名为 less
    less as
    # 导入所需的函数和方法，用于处理掩码数组的操作和数学运算
    masked_invalid as masked_invalid,        # 导入函数 masked_invalid，用于处理无效掩码
    masked_less as masked_less,              # 导入函数 masked_less，用于处理小于比较掩码
    masked_less_equal as masked_less_equal,  # 导入函数 masked_less_equal，用于处理小于等于比较掩码
    masked_not_equal as masked_not_equal,    # 导入函数 masked_not_equal，用于处理不等于比较掩码
    masked_object as masked_object,          # 导入函数 masked_object，用于处理对象掩码
    masked_outside as masked_outside,        # 导入函数 masked_outside，用于处理外部掩码
    masked_print_option as masked_print_option,  # 导入函数 masked_print_option，用于掩码打印选项
    masked_singleton as masked_singleton,    # 导入函数 masked_singleton，用于处理单例掩码
    masked_values as masked_values,          # 导入函数 masked_values，用于处理数值掩码
    masked_where as masked_where,            # 导入函数 masked_where，用于条件掩码
    max as max,                              # 导入 max 函数，用于求最大值
    maximum as maximum,                      # 导入 maximum 函数，用于求最大值
    maximum_fill_value as maximum_fill_value,  # 导入 maximum_fill_value 函数，用于最大填充值
    mean as mean,                            # 导入 mean 函数，用于求均值
    min as min,                              # 导入 min 函数，用于求最小值
    minimum as minimum,                      # 导入 minimum 函数，用于求最小值
    minimum_fill_value as minimum_fill_value,  # 导入 minimum_fill_value 函数，用于最小填充值
    mod as mod,                              # 导入 mod 函数，用于求余数
    multiply as multiply,                    # 导入 multiply 函数，用于求乘积
    mvoid as mvoid,                          # 导入 mvoid 函数，用于处理虚类型
    ndim as ndim,                            # 导入 ndim 函数，用于返回数组的维度数
    negative as negative,                    # 导入 negative 函数，用于求负数
    nomask as nomask,                        # 导入 nomask 函数，用于不使用掩码
    nonzero as nonzero,                      # 导入 nonzero 函数，用于返回非零元素的索引
    not_equal as not_equal,                  # 导入 not_equal 函数，用于比较不等于
    ones as ones,                            # 导入 ones 函数，用于创建全为1的数组
    outer as outer,                          # 导入 outer 函数，用于外积计算
    outerproduct as outerproduct,            # 导入 outerproduct 函数，用于外积计算
    power as power,                          # 导入 power 函数，用于幂运算
    prod as prod,                            # 导入 prod 函数，用于计算数组元素的乘积
    product as product,                      # 导入 product 函数，用于计算数组元素的乘积
    ptp as ptp,                              # 导入 ptp 函数，用于计算数组的峰值到峰值范围
    put as put,                              # 导入 put 函数，用于按照索引赋值
    putmask as putmask,                      # 导入 putmask 函数，用于按照掩码条件赋值
    ravel as ravel,                          # 导入 ravel 函数，用于将多维数组展平为一维数组
    remainder as remainder,                  # 导入 remainder 函数，用于求余数
    repeat as repeat,                        # 导入 repeat 函数，用于重复数组元素
    reshape as reshape,                      # 导入 reshape 函数，用于改变数组形状
    resize as resize,                        # 导入 resize 函数，用于改变数组大小
    right_shift as right_shift,              # 导入 right_shift 函数，用于右移位运算
    round as round,                          # 导入 round 函数，用于四舍五入
    set_fill_value as set_fill_value,        # 导入 set_fill_value 函数，用于设置填充值
    shape as shape,                          # 导入 shape 函数，用于返回数组的形状
    sin as sin,                              # 导入 sin 函数，用于求正弦值
    sinh as sinh,                            # 导入 sinh 函数，用于求双曲正弦值
    size as size,                            # 导入 size 函数，用于返回数组元素的个数
    soften_mask as soften_mask,              # 导入 soften_mask 函数，用于软化掩码
    sometrue as sometrue,                    # 导入 sometrue 函数，用于判断是否有真值
    sort as sort,                            # 导入 sort 函数，用于数组排序
    sqrt as sqrt,                            # 导入 sqrt 函数，用于求平方根
    squeeze as squeeze,                      # 导入 squeeze 函数，用于去除维度为1的轴
    std as std,                              # 导入 std 函数，用于求标准差
    subtract as subtract,                    # 导入 subtract 函数，用于求差值
    sum as sum,                              # 导入 sum 函数，用于求和
    swapaxes as swapaxes,                    # 导入 swapaxes 函数，用于交换数组的轴
    take as take,                            # 导入 take 函数，用于按索引获取数组元素
    tan as tan,                              # 导入 tan 函数，用于求正切值
    tanh as tanh,                            # 导入 tanh 函数，用于求双曲正切值
    trace as trace,                          # 导入 trace 函数，用于计算对角线元素的和
    transpose as transpose,                  # 导入 transpose 函数，用于矩阵转置
    true_divide as true_divide,              # 导入 true_divide 函数，用于真实除法
    var as var,                              # 导入 var 函数，用于求方差
    where as where,                          # 导入 where 函数，用于条件选择
    zeros as zeros,                          # 导入 zeros 函数，用于创建全为0的数组
# 导入 numpy.ma.extras 模块中的多个函数，并使用简化的别名
from numpy.ma.extras import (
    apply_along_axis as apply_along_axis,           # 别名 apply_along_axis，沿指定轴应用函数
    apply_over_axes as apply_over_axes,             # 别名 apply_over_axes，沿指定一组轴应用函数
    atleast_1d as atleast_1d,                       # 别名 atleast_1d，将输入至少转换为 1 维数组
    atleast_2d as atleast_2d,                       # 别名 atleast_2d，将输入至少转换为 2 维数组
    atleast_3d as atleast_3d,                       # 别名 atleast_3d，将输入至少转换为 3 维数组
    average as average,                             # 别名 average，计算数组的加权平均值
    clump_masked as clump_masked,                   # 别名 clump_masked，将掩码数组的相邻条目组合成序列
    clump_unmasked as clump_unmasked,               # 别名 clump_unmasked，将未掩码数组的相邻条目组合成序列
    column_stack as column_stack,                   # 别名 column_stack，按列堆叠序列或数组
    compress_cols as compress_cols,                 # 别名 compress_cols，压缩列中的掩码数组
    compress_nd as compress_nd,                     # 别名 compress_nd，压缩多维数组的掩码数组
    compress_rowcols as compress_rowcols,           # 别名 compress_rowcols，压缩行列中的掩码数组
    compress_rows as compress_rows,                 # 别名 compress_rows，压缩行中的掩码数组
    count_masked as count_masked,                   # 别名 count_masked，计算数组中掩码条目的数量
    corrcoef as corrcoef,                           # 别名 corrcoef，计算相关系数矩阵
    cov as cov,                                     # 别名 cov，计算协方差矩阵
    diagflat as diagflat,                           # 别名 diagflat，创建对角扁平数组
    dot as dot,                                     # 别名 dot，计算两个数组的点积
    dstack as dstack,                               # 别名 dstack，按深度堆叠序列或数组
    ediff1d as ediff1d,                             # 别名 ediff1d，计算 1 维数组的首差值
    flatnotmasked_contiguous as flatnotmasked_contiguous,  # 别名 flatnotmasked_contiguous，返回连续未掩码条目的迭代器
    flatnotmasked_edges as flatnotmasked_edges,     # 别名 flatnotmasked_edges，返回边界未掩码条目的迭代器
    hsplit as hsplit,                               # 别名 hsplit，水平分割数组成子数组
    hstack as hstack,                               # 别名 hstack，按水平堆叠序列或数组
    isin as isin,                                   # 别名 isin，测试数组的成员资格
    in1d as in1d,                                   # 别名 in1d，测试数组的成员资格，返回布尔数组
    intersect1d as intersect1d,                     # 别名 intersect1d，计算两个数组的交集
    mask_cols as mask_cols,                         # 别名 mask_cols，返回列中的掩码数组
    mask_rowcols as mask_rowcols,                   # 别名 mask_rowcols，返回行列中的掩码数组
    mask_rows as mask_rows,                         # 别名 mask_rows，返回行中的掩码数组
    masked_all as masked_all,                       # 别名 masked_all，创建所有掩码数组
    masked_all_like as masked_all_like,             # 别名 masked_all_like，创建与输入形状相同的所有掩码数组
    median as median,                               # 别名 median，计算数组的中位数
    mr_ as mr_,                                     # 别名 mr_，返回平滑移动均值的序列
    ndenumerate as ndenumerate,                     # 别名 ndenumerate，返回多维数组的索引和值的迭代器
    notmasked_contiguous as notmasked_contiguous,   # 别名 notmasked_contiguous，返回连续未掩码条目的迭代器
    notmasked_edges as notmasked_edges,             # 别名 notmasked_edges，返回边界未掩码条目的迭代器
    polyfit as polyfit,                             # 别名 polyfit，计算多项式拟合
    row_stack as row_stack,                         # 别名 row_stack，按行堆叠序列或数组
    setdiff1d as setdiff1d,                         # 别名 setdiff1d，计算两个数组的差集
    setxor1d as setxor1d,                           # 别名 setxor1d，计算两个数组的异或集
    stack as stack,                                 # 别名 stack，按堆叠序列或数组
    unique as unique,                               # 别名 unique，查找数组的唯一元素
    union1d as union1d,                             # 别名 union1d，计算两个数组的并集
    vander as vander,                               # 别名 vander，生成范德蒙矩阵
    vstack as vstack                                # 别名 vstack，按垂直堆叠序列或数组
)

__all__: list[str]   # 定义一个列表，包含模块中应导出的所有公共名称
test: PytestTester    # 定义一个变量 test，其类型为 PytestTester，可能用于测试相关功能
```