# `D:\src\scipysrc\scipy\scipy\special\_cdflib_wrappers.pxd`

```
# 导入 sf_error 模块中的 sf_error 对象
from . cimport sf_error

# 导入 libc.math 模块中的 NAN, isnan, isinf, isfinite
from libc.math cimport NAN, isnan, isinf, isfinite

# 从特定的头文件 "special_wrappers.h" 中导入以下函数声明
cdef extern from "special_wrappers.h" nogil:
    double cephes_ndtr_wrap(double a)
    double cephes_ndtri_wrap(double y0)

# 从特定的头文件 "cdflib.h" 中导入以下结构体和函数声明
cdef extern from "cdflib.h" nogil:
    cdef struct TupleDDI:
        double d1
        double d2
        int i1

    cdef struct TupleDID:
        double d1
        int i1
        double d2

    cdef struct TupleDDID:
        double d1
        double d2
        int i1
        double d3

    TupleDID cdfbet_which3(double, double, double, double, double)
    TupleDID cdfbet_which4(double, double, double, double, double)
    TupleDID cdfbin_which2(double, double, double, double, double)
    TupleDID cdfbin_which3(double, double, double, double, double)
    TupleDID cdfchi_which3(double, double, double)
    TupleDDID cdfchn_which1(double, double, double)
    TupleDID cdfchn_which2(double, double, double)
    TupleDID cdfchn_which3(double, double, double)
    TupleDID cdfchn_which4(double, double, double)
    TupleDID cdff_which4(double, double, double, double);
    TupleDDID cdffnc_which1(double, double, double, double);
    TupleDID cdffnc_which2(double, double, double, double, double);
    TupleDID cdffnc_which3(double, double, double, double, double);
    TupleDID cdffnc_which4(double, double, double, double, double);
    TupleDID cdffnc_which5(double, double, double, double, double);
    TupleDID cdfgam_which2(double, double, double, double);
    TupleDID cdfgam_which3(double, double, double, double);
    TupleDID cdfgam_which4(double, double, double, double);
    TupleDID cdfnbn_which2(double, double, double, double, double);
    TupleDID cdfnbn_which3(double, double, double, double, double);
    TupleDID cdfnor_which3(double, double, double, double);
    TupleDID cdfnor_which4(double, double, double, double);
    TupleDID cdfpoi_which2(double, double, double);
    TupleDDID cdft_which1(double, double);
    TupleDID cdft_which2(double, double, double);
    TupleDID cdft_which3(double, double, double);
    TupleDDID cdftnc_which1(double, double, double);
    TupleDID cdftnc_which2(double, double, double, double);
    TupleDID cdftnc_which3(double, double, double, double);
    TupleDID cdftnc_which4(double, double, double, double);


# 定义内联函数 get_result，接收多个参数，返回一个 double 类型的值，无异常，无 GIL
cdef inline double get_result(
        char *name,
        char **argnames,
        double result,
        int status,
        double bound,
        int return_bound
) noexcept nogil:
    cdef char *arg
    """Get result and perform error handling from cdflib output."""
    # 如果状态值小于 0，表示出现错误
    if status < 0:
        # 获取对应参数名
        arg = argnames[-(status + 1)]
        # 调用 sf_error 模块中的 error 函数，报错参数范围错误
        sf_error.error(name, sf_error.ARG,
                       "Input parameter %s is out of range", arg)
        return NAN  # 返回 NaN 表示无效数据
    # 如果状态值等于 0，表示正常返回结果
    if status == 0:
        return result  # 直接返回结果值
    # 如果状态值等于 1，表示结果可能低于搜索边界值
    if status == 1:
        # 调用 sf_error 模块中的 error 函数，报错结果低于搜索边界
        sf_error.error(name, sf_error.OTHER,
                       "Answer appears to be lower than lowest search bound (%g)", bound)
        return bound if return_bound else NAN  # 如果需要返回边界值则返回，否则返回 NaN
    # 如果状态码为2：
    if status == 2:
        # 调用错误处理函数，报告其他类型错误，提示超出最大搜索范围（%g为格式化字符串，表示bound）
        sf_error.error(name, sf_error.OTHER,
                       "Answer appears to be higher than highest search bound (%g)", bound)
        # 如果需要返回bound，则返回bound；否则返回NaN
        return bound if return_bound else NAN
    
    # 如果状态码为3或4：
    if status == 3 or status == 4:
        # 调用错误处理函数，报告其他类型错误，指示两个内部参数的总和不等于1.0
        sf_error.error(name, sf_error.OTHER,
                       "Two internal parameters that should sum to 1.0 do not.")
        # 返回NaN
        return NAN
    
    # 如果状态码为10：
    if status == 10:
        # 调用错误处理函数，报告其他类型错误，指示计算错误
        sf_error.error(name, sf_error.OTHER, "Computational error")
        # 返回NaN
        return NAN
    
    # 如果状态码不在已知范围内：
    # 调用错误处理函数，报告未知错误
    sf_error.error(name, sf_error.OTHER, "Unknown error.")
    # 返回NaN
    return NAN
# 计算不完全贝塔函数的值，使用三个参数 p, b, x
cdef inline double btdtria(double p, double b, double x) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算补码 q = 1 - p
        double y = 1.0 - x  # 计算补码 y = 1 - x
        double result, bound  # 定义结果和边界变量
        int status = 10  # 初始化状态为 10
        char *argnames[5]  # 定义包含参数名的字符串数组
        TupleDID ret  # 定义一个元组，包含 double, int, double 类型的返回值

    argnames[0] = "p"  # 参数名称列表
    argnames[1] = "q"
    argnames[2] = "x"
    argnames[3] = "y"
    argnames[4] = "b"

    # 如果 p, b, x 中有任何一个是 NaN，则返回 NaN
    if isnan(p) or isnan(b) or isnan(x):
        return NAN

    # 调用 cdfbet_which3 函数计算结果
    ret = cdfbet_which3(p, q, x, y, b)
    result, status, bound = ret.d1, ret.i1, ret.d2  # 提取函数返回的结果值
    # 调用 get_result 函数并返回结果
    return get_result("btdtria", argnames, result, status, bound, 1)


# 计算不完全贝塔函数的值，使用三个参数 a, p, x
cdef inline double btdtrib(double a, double p, double x) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算补码 q = 1 - p
        double y = 1.0 - x  # 计算补码 y = 1 - x
        double result, bound  # 定义结果和边界变量
        int status = 10  # 初始化状态为 10
        char *argnames[5]  # 定义包含参数名的字符串数组
        TupleDID ret  # 定义一个元组，包含 double, int, double 类型的返回值

    # 如果 a, p, x 中有任何一个是 NaN，则返回 NaN
    if isnan(a) or isnan(p) or isnan(x):
        return NAN

    argnames[0] = "p"  # 参数名称列表
    argnames[1] = "q"
    argnames[2] = "x"
    argnames[3] = "y"
    argnames[4] = "a"

    # 调用 cdfbet_which4 函数计算结果
    ret = cdfbet_which4(p, q, x, y, a)
    result, status, bound = ret.d1, ret.i1, ret.d2  # 提取函数返回的结果值
    # 调用 get_result 函数并返回结果
    return get_result("btdtrib", argnames, result, status, bound, 1)


# 计算二项分布的累积分布函数的反函数值，使用三个参数 p, xn, pr
cdef inline double bdtrik(double p, double xn, double pr) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算补码 q = 1 - p
        double ompr = 1.0 - pr  # 计算补码 ompr = 1 - pr
        double result, bound  # 定义结果和边界变量
        int status  # 定义状态变量
        char *argnames[5]  # 定义包含参数名的字符串数组
        TupleDID ret  # 定义一个元组，包含 double, int, double 类型的返回值

    # 如果 p, xn, pr 中有任何一个是 NaN 或 xn 不是有限值，则返回 NaN
    if isnan(p) or not isfinite(xn) or isnan(pr):
        return NAN

    argnames[0] = "p"  # 参数名称列表
    argnames[1] = "q"
    argnames[2] = "xn"
    argnames[3] = "pr"
    argnames[4] = "ompr"

    # 调用 cdfbin_which2 函数计算结果
    ret = cdfbin_which2(p, q, xn, pr, ompr)
    result, status, bound = ret.d1, ret.i1, ret.d2  # 提取函数返回的结果值
    # 调用 get_result 函数并返回结果
    return get_result("btdtrik", argnames, result, status, bound, 1)


# 计算二项分布的累积分布函数的逆函数值，使用三个参数 s, p, pr
cdef inline double bdtrin(double s, double p, double pr) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算补码 q = 1 - p
        double ompr = 1.0 - pr  # 计算补码 ompr = 1 - pr
        double result, bound  # 定义结果和边界变量
        int status = 10  # 初始化状态为 10
        char *argnames[5]  # 定义包含参数名的字符串数组
        TupleDID ret  # 定义一个元组，包含 double, int, double 类型的返回值

    # 如果 s, p, pr 中有任何一个是 NaN，则返回 NaN
    if isnan(s) or isnan(p) or isnan(pr):
        return NAN

    argnames[0] = "p"  # 参数名称列表
    argnames[1] = "q"
    argnames[2] = "s"
    argnames[3] = "pr"
    argnames[4] = "ompr"

    # 调用 cdfbin_which3 函数计算结果
    ret = cdfbin_which3(p, q, s, pr, ompr)
    result, status, bound = ret.d1, ret.i1, ret.d2  # 提取函数返回的结果值
    # 调用 get_result 函数并返回结果
    return get_result("btdtrin", argnames, result, status, bound, 1)


# 计算卡方分布的累积分布函数的逆函数值，使用两个参数 p, x
cdef inline double chdtriv(double p, double x) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算补码 q = 1 - p
        double result, bound  # 定义结果和边界变量
        int status = 10  # 初始化状态为 10
        char *argnames[3]  # 定义包含参数名的字符串数组

    # 如果 p, x 中有任何一个是 NaN，则返回 NaN
    if isnan(p) or isnan(x):
        return NAN

    argnames[0] = "p"  # 参数名称列表
    argnames[1] = "q"
    argnames[2] = "x"

    # 调用 cdfchi_which3 函数计算结果
    ret = cdfchi_which3(p, q, x)
    result, status, bound = ret.d1, ret.i1, ret.d2  # 提取函数返回的结果值
    # 调用 get_result 函数并返回结果
    return get_result("chdtriv", argnames, result, status, bound, 1)


# 计算非中心卡方分布的累积分布函数的值，使用三个参数 x, df, nc
cdef inline double chndtr(double x, double df, double nc) noexcept nogil:
    cdef:
        double result, _, bound  # 定义结果和边界变量
        int status = 10  # 初始化状态为 10
        char *argnames[3]  # 定义包含参数名的字符串数组
        TupleDDID ret  # 定义一个元组，包含 double, double, int, double 类型的返回值
    # 如果 x、df 或 nc 中有任何一个是 NaN，则返回 NAN
    if isnan(x) or isnan(df) or isnan(nc):
      return NAN

    # 将参数列表中的第一个、第二个和第三个参数分别设置为 "x"、"df"、"nc"
    argnames[0] = "x"
    argnames[1] = "df"
    argnames[2] = "nc"

    # 调用 cdfchn_which1 函数计算结果
    ret = cdfchn_which1(x, df, nc)
    # 将返回值中的结果、状态和边界分别赋值给 result、status 和 bound
    result, status, bound = ret.d1, ret.i1, ret.d3
    # 调用 get_result 函数获取最终结果，返回结果值
    return get_result("chndtr", argnames, result, status, bound, 1)
# 定义一个内联函数，计算非中心卡方分布的累积分布函数的反函数值。
# 函数参数包括 x：要计算的反函数的值，p：概率，nc：非中心参数。
cdef inline double chndtridf(double x, double p, double nc) noexcept nogil:
    cdef:
        double result, bound  # 定义结果值和边界值
        int status = 10  # 初始化状态为10，用于返回状态信息
        char *argnames[3]  # 参数名称数组
        TupleDID ret  # 定义返回元组类型

    # 如果任一参数 x、p 或 nc 是 NaN，则返回 NaN
    if isnan(x) or isnan(p) or isnan(nc):
        return NAN

    # 设置参数名称数组的值
    argnames[0] = "p"
    argnames[1] = "x"
    argnames[2] = "nc"

    # 调用 cdfchn_which3 函数，返回结果元组
    ret = cdfchn_which3(p, x, nc)
    # 解包结果元组中的值：d1 是结果值，i1 是状态值，d2 是边界值
    result, status, bound = ret.d1, ret.i1, ret.d2
    # 调用 get_result 函数，返回结果值作为反函数的计算结果
    return get_result("chndtridf", argnames, result, status, bound, 1)


# 定义一个内联函数，计算非中心卡方分布的累积分布函数的值。
# 函数参数包括 x：要计算的值，df：自由度，p：概率。
cdef inline double chndtrinc(double x, double df, double p) noexcept nogil:
    cdef:
        double result, bound  # 定义结果值和边界值
        int status = 10  # 初始化状态为10，用于返回状态信息
        char *argnames[3]  # 参数名称数组
        TupleDID ret  # 定义返回元组类型

    # 如果任一参数 x、df 或 p 是 NaN，则返回 NaN
    if isnan(x) or isnan(df) or isnan(p):
        return NAN

    # 设置参数名称数组的值
    argnames[0] = "p"
    argnames[1] = "x"
    argnames[2] = "df"

    # 调用 cdfchn_which4 函数，返回结果元组
    ret = cdfchn_which4(p, x, df)
    # 解包结果元组中的值：d1 是结果值，i1 是状态值，d2 是边界值
    result, status, bound = ret.d1, ret.i1, ret.d2
    # 调用 get_result 函数，返回结果值作为累积分布函数的计算结果
    return get_result("chndtrinc", argnames, result, status, bound, 1)


# 定义一个内联函数，计算非中心卡方分布的累积分布函数的值。
# 函数参数包括 p：概率，df：自由度，nc：非中心参数。
cdef inline double chndtrix(double p, double df, double nc) noexcept nogil:
    cdef:
        double result, bound  # 定义结果值和边界值
        int status = 10  # 初始化状态为10，用于返回状态信息
        char *argnames[3]  # 参数名称数组
        TupleDID ret  # 定义返回元组类型

    # 如果任一参数 p、df 或 nc 是 NaN，则返回 NaN
    if isnan(p) or isnan(df) or isnan(nc):
        return NAN

    # 设置参数名称数组的值
    argnames[0] = "p"
    argnames[1] = "df"
    argnames[2] = "nc"

    # 调用 cdfchn_which2 函数，返回结果元组
    ret = cdfchn_which2(p, df, nc)
    # 解包结果元组中的值：d1 是结果值，i1 是状态值，d2 是边界值
    result, status, bound = ret.d1, ret.i1, ret.d2
    # 调用 get_result 函数，返回结果值作为累积分布函数的计算结果
    return get_result("chndtrix", argnames, result, status, bound, 0)


# 定义一个内联函数，计算 F 分布的累积分布函数的反函数值。
# 函数参数包括 dfn：自由度，p：概率，f：F 统计量。
cdef inline double fdtridfd(double dfn, double p, double f) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算概率的补值 q
        double result, bound  # 定义结果值和边界值
        int status = 10  # 初始化状态为10，用于返回状态信息
        char *argnames[4]  # 参数名称数组
        TupleDID ret  # 定义返回元组类型

    # 如果任一参数 dfn、p 或 f 是 NaN，则返回 NaN
    if isnan(dfn) or isnan(p) or isnan(f):
        return NAN

    # 设置参数名称数组的值
    argnames[0] = "p"
    argnames[1] = "q"
    argnames[2] = "f"
    argnames[3] = "dfn"

    # 调用 cdff_which4 函数，返回结果元组
    ret = cdff_which4(p, q, f, dfn)
    # 解包结果元组中的值：d1 是结果值，i1 是状态值，d2 是边界值
    result, status, bound = ret.d1, ret.i1, ret.d2
    # 调用 get_result 函数，返回结果值作为反函数的计算结果
    return get_result("fdtridfd", argnames, result, status, bound, 1)


# 定义一个内联函数，计算 Gamma 分布的累积分布函数的值。
# 函数参数包括 p：概率，shp：形状参数，x：Gamma 变量。
cdef inline double gdtria(double p, double shp, double x) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算概率的补值 q
        double result, bound  # 定义结果值和边界值
        int status = 10  # 初始化状态为10，用于返回状态信息
        char *argnames[4]  # 参数名称数组
        TupleDID ret  # 定义返回元组类型

    # 如果任一参数 p、shp 或 x 是 NaN，则返回 NaN
    if isnan(p) or isnan(shp) or isnan(x):
        return NAN

    # 设置参数名称数组的值
    argnames[0] = "p"
    argnames[1] = "q"
    argnames[2] = "x"
    argnames[3] = "shp"

    # 调用 cdfgam_which4 函数，返回结果元组
    ret = cdfgam_which4(p, q, x, shp)
    # 解包结果元组中的值：d1 是结果值，i1 是状态值，d2 是边界值
    result, status, bound = ret.d1, ret.i1, ret.d2
    # 调用 get_result 函数，返回结果值作为累积分布函数的计算结果
    return get_result("gdtria", argnames, result, status, bound, 1)


# 定义一个内联函数，计算 Gamma 分布的累积分布函数的反函数值。
# 函数参数包括 scl：尺度参数，p：概率，x：Gamma 变量。
cdef inline double gdtrib(double scl, double p, double x) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算概率的补值 q
        double result, bound  # 定义结果值和边界值
        int status = 10  # 初始化状态为10，用于返回状态信息
        char *argnames[4]  # 参数名称数组
        TupleDID ret  # 定义返回元组类型

    # 如果任一参数 scl、p 或 x 是 NaN，则返回
# 计算 generalized gamma distribution 中的累积分布函数值
cdef inline double gdtrix(double scl, double shp, double p) noexcept nogil:
    cdef:
        double q = 1.0 - p                # 计算概率的补数 q
        double result, bound              # 定义结果和边界变量
        int status = 10                   # 设定初始状态值
        char *argnames[4]                 

    # 检查输入参数是否包含 NaN，如果是则返回 NaN
    if isnan(scl) or isnan(shp) or isnan(p):
        return NAN

    # 设置参数名列表
    argnames[0] = "p"
    argnames[1] = "q"
    argnames[2] = "shp"
    argnames[3] = "scl"

    # 调用 C 库函数计算 generalized gamma 分布的累积分布函数
    ret = cdfgam_which2(p, q, shp, scl)
    result, status, bound = ret.d1, ret.i1, ret.d2

    # 调用函数获取结果
    return get_result("gdtrix", argnames, result, status, bound, 1)


# 计算负二项分布中的累积分布函数值
cdef inline double nbdtrik(double p, double xn, double pr) noexcept nogil:
    cdef:
        double q = 1.0 - p                # 计算概率的补数 q
        double ompr = 1.0 - pr            # 计算概率的补数 ompr
        double result, bound              # 定义结果和边界变量
        int status = 10                   # 设定初始状态值
        char *argnames[5]

    # 检查输入参数是否包含 NaN 或无穷大，如果是则返回 NaN
    if isnan(p) or not isfinite(xn) or isnan(pr):
        return NAN

    # 设置参数名列表
    argnames[0] = "p"
    argnames[1] = "q"
    argnames[2] = "xn"
    argnames[3] = "pr"
    argnames[4] = "ompr"

    # 调用 C 库函数计算负二项分布的累积分布函数
    ret = cdfnbn_which2(p, q, xn, pr, ompr)
    result, status, bound = ret.d1, ret.i1, ret.d2

    # 调用函数获取结果
    return get_result("nbdtrik", argnames, result, status, bound, 1)


# 计算负二项分布中的逆累积分布函数值
cdef inline double nbdtrin(double s, double p, double pr) noexcept nogil:
    cdef:
        double q = 1.0 - p                # 计算概率的补数 q
        double ompr = 1.0 - pr            # 计算概率的补数 ompr
        double result, bound              # 定义结果和边界变量
        int status = 10                   # 设定初始状态值
        char *argnames[5]

    # 检查输入参数是否包含 NaN，如果是则返回 NaN
    if isnan(s) or isnan(p) or isnan(pr):
        return NAN

    # 设置参数名列表
    argnames[0] = "p"
    argnames[1] = "q"
    argnames[2] = "s"
    argnames[3] = "pr"
    argnames[4] = "ompr"

    # 调用 C 库函数计算负二项分布的逆累积分布函数
    ret = cdfnbn_which3(p, q, s, pr, ompr)
    result, status, bound = ret.d1, ret.i1, ret.d2

    # 调用函数获取结果
    return get_result("nbdtrin", argnames, result, status, bound, 1)


# 计算非中心 F 分布中的累积分布函数值
cdef inline double ncfdtr(double dfn, double dfd, double nc, double f) noexcept nogil:
    cdef:
        double result, _, bound           # 定义结果和边界变量
        int status = 10                   # 设定初始状态值
        char *argnames[4]

    # 检查输入参数是否包含 NaN，如果是则返回 NaN
    if isnan(dfn) or isnan(dfd) or isnan(nc) or isnan(f):
        return NAN

    # 设置参数名列表
    argnames[0] = "f"
    argnames[1] = "dfn"
    argnames[2] = "dfd"
    argnames[3] = "nc"

    # 调用 C 库函数计算非中心 F 分布的累积分布函数
    ret = cdffnc_which1(f, dfn, dfd, nc)
    result, status, bound = ret.d1, ret.i1, ret.d3

    # 调用函数获取结果
    return get_result("ncfdtr", argnames, result, status, bound, 0)


# 计算非中心 F 分布中的逆累积分布函数值
cdef inline double ncfdtri(double dfn, double dfd, double nc, double p) noexcept nogil:
    cdef:
        double q = 1.0 - p                # 计算概率的补数 q
        double result, bound              # 定义结果和边界变量
        int status = 10                   # 设定初始状态值
        char *argnames[5]

    # 检查输入参数是否包含 NaN，如果是则返回 NaN
    if isnan(dfn) or isnan(dfd) or isnan(nc) or isnan(p):
        return NAN

    # 设置参数名列表
    argnames[0] = "p"
    argnames[1] = "q"
    argnames[2] = "dfn"
    argnames[3] = "dfd"
    argnames[4] = "nc"

    # 调用 C 库函数计算非中心 F 分布的逆累积分布函数
    ret = cdffnc_which2(p, q, dfn, dfd, nc)
    result, status, bound = ret.d1, ret.i1, ret.d2

    # 调用函数获取结果
    return get_result("ncfdtri", argnames, result, status, bound, 1)


# 计算非中心 F 分布中的逆累积分布函数值的参数 dfd 版本
cdef inline double ncfdtridfd(double dfn, double p, double nc, double f) noexcept nogil:
    # 声明并初始化变量 q，表示概率 1 - p
    cdef:
        double q = 1.0 - p  # q 的计算公式
        double result, bound  # 定义结果和边界变量
        int status = 10  # 初始化状态为 10
        char *argnames[5]  # 声明一个包含 5 个 char* 元素的数组，用于存储参数名称
        TupleDID ret  # 声明一个 TupleDID 类型的变量 ret，用于存储函数返回值

    # 如果输入的 dfn、p、nc 或 f 中有任何一个是 NaN，则返回 NaN
    if isnan(dfn) or isnan(p) or isnan(nc) or isnan(f):
      return NAN

    # 设置参数名称数组中的元素
    argnames[0] = "p"
    argnames[1] = "q"
    argnames[2] = "f"
    argnames[3] = "dfn"
    argnames[4] = "nc"

    # 调用 cdffnc_which4 函数，传入 p, q, f, dfn, nc 作为参数
    ret = cdffnc_which4(p, q, f, dfn, nc)
    # 从返回的 TupleDID 对象中获取结果、状态和边界值
    result, status, bound = ret.d1, ret.i1, ret.d2
    # 调用 get_result 函数，传入函数名 "ncfdtridfd"、参数名称数组、结果、状态、边界和 1 作为参数，并返回结果
    return get_result("ncfdtridfd", argnames, result, status, bound, 1)
# 计算非中心 F 分布的累积分布函数的值
cdef inline double ncfdtridfn(double p, double dfd, double nc, double f) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算概率的补数
        double result, bound  # 定义函数返回值和边界值
        int status = 10  # 定义状态变量，默认为10
        char *argnames[5]  # 声明字符串数组，用于存储参数名
        TupleDID ret  # 声明 TupleDID 类型的变量 ret

    if isnan(p) or isnan(dfd) or isnan(nc) or isnan(f):
      return NAN  # 如果任何参数是 NaN，则返回 NaN

    argnames[0] = "p"  # 设置参数名数组的各个位置
    argnames[1] = "q"
    argnames[2] = "f"
    argnames[3] = "dfd"
    argnames[4] = "nc"

    ret = cdffnc_which3(p, q, f, dfd, nc)  # 调用特定函数计算结果
    result, status, bound = ret.d1, ret.i1, ret.d2  # 从返回的 TupleDID 中获取结果、状态和边界值
    return get_result("ncfdtridfn", argnames, result, status, bound, 1)  # 调用 get_result 函数返回结果


# 计算非中心 F 分布的累积分布函数的值
cdef inline double ncfdtrinc(double dfn, double dfd, double p, double f) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算概率的补数
        double result, bound  # 定义函数返回值和边界值
        int status = 10  # 定义状态变量，默认为10
        char *argnames[5]  # 声明字符串数组，用于存储参数名
        TupleDID ret  # 声明 TupleDID 类型的变量 ret

    if isnan(dfn) or isnan(dfd) or isnan(p) or isnan(f):
      return NAN  # 如果任何参数是 NaN，则返回 NaN

    argnames[0] = "p"  # 设置参数名数组的各个位置
    argnames[1] = "q"
    argnames[2] = "f"
    argnames[3] = "dfn"
    argnames[4] = "dfd"

    ret = cdffnc_which5(p, q, f, dfn, dfd)  # 调用特定函数计算结果
    result, status, bound = ret.d1, ret.i1, ret.d2  # 从返回的 TupleDID 中获取结果、状态和边界值
    return get_result("ncfdtrinc", argnames, result, status, bound, 1)  # 调用 get_result 函数返回结果


# 计算非中心 t 分布的累积分布函数的值
cdef inline double nctdtr(double df, double nc, double t) noexcept nogil:
    cdef:
        double result, _, bound  # 定义函数返回值、不使用的变量和边界值
        int status = 10  # 定义状态变量，默认为10
        char *argnames[3]  # 声明字符串数组，用于存储参数名
        TupleDDID ret  # 声明 TupleDDID 类型的变量 ret

    if isnan(df) or isnan(nc) or isnan(t):
      return NAN  # 如果任何参数是 NaN，则返回 NaN

    argnames[0] = "t"  # 设置参数名数组的各个位置
    argnames[1] = "df"
    argnames[2] = "nc"

    ret = cdftnc_which1(t, df, nc)  # 调用特定函数计算结果
    result, status, bound = ret.d1, ret.i1, ret.d3  # 从返回的 TupleDDID 中获取结果、状态和边界值
    return get_result("nctdtr", argnames, result, status, bound, 1)  # 调用 get_result 函数返回结果


# 计算非中心 t 分布的累积分布函数的值
cdef inline double nctdtridf(double p, double nc, double t) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算概率的补数
        double result, bound  # 定义函数返回值和边界值
        int status = 10  # 定义状态变量，默认为10
        char *argnames[4]  # 声明字符串数组，用于存储参数名
        TupleDID ret  # 声明 TupleDID 类型的变量 ret

    if isnan(p) or isnan(nc) or isnan(t):
      return NAN  # 如果任何参数是 NaN，则返回 NaN

    argnames[0] = "p"  # 设置参数名数组的各个位置
    argnames[1] = "q"
    argnames[2] = "t"
    argnames[3] = "nc"

    ret = cdftnc_which3(p, q, t, nc)  # 调用特定函数计算结果
    result, status, bound = ret.d1, ret.i1, ret.d2  # 从返回的 TupleDID 中获取结果、状态和边界值
    return get_result("nctdtridf", argnames, result, status, bound, 1)  # 调用 get_result 函数返回结果


# 计算非中心 t 分布的累积分布函数的值
cdef inline double nctdtrinc(double df, double p, double t) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算概率的补数
        double result, bound  # 定义函数返回值和边界值
        int status = 10  # 定义状态变量，默认为10
        char *argnames[4]  # 声明字符串数组，用于存储参数名
        TupleDID ret  # 声明 TupleDID 类型的变量 ret

    if isnan(df) or isnan(p) or isnan(t):
      return NAN  # 如果任何参数是 NaN，则返回 NaN

    argnames[0] = "p"  # 设置参数名数组的各个位置
    argnames[1] = "q"
    argnames[2] = "t"
    argnames[3] = "df"

    ret = cdftnc_which4(p, q, t, df)  # 调用特定函数计算结果
    result, status, bound = ret.d1, ret.i1, ret.d2  # 从返回的 TupleDID 中获取结果、状态和边界值
    return get_result("nctdtrinc", argnames, result, status, bound, 1)  # 调用 get_result 函数返回结果


# 计算非中心 t 分布的累积分布函数的值
cdef inline double nctdtrit(double df, double nc, double p) noexcept nogil:
    cdef:
        double q = 1.0 - p  # 计算概率的补数
        double result, bound  # 定义函数返回值和边界值
        int status = 10  # 定义状态变量，默认为10
        char *argnames[4]  # 声明字符串数组，用于存储参数名
        TupleDID ret  # 声明 TupleDID 类型的变量 ret

    if isnan(df) or isnan(nc) or isnan(p):
      return NAN  # 如果任何参数是 NaN，则返回 NaN

    argnames[0] = "p"  # 设置参数名数组的各个位置
    argnames[1] = "q"
    argnames[2] = "t"
    argnames[3] = "nc"
    # 将参数名与索引关联起来，设置第二个参数名为"q"，第三个参数名为"df"，第四个参数名为"nc"
    argnames[1] = "q"
    argnames[2] = "df"
    argnames[3] = "nc"

    # 调用 cdftnc_which2 函数计算并返回结果
    ret = cdftnc_which2(p, q, df, nc)
    # 从返回的结果对象 ret 中获取分布的参数 result、状态 status 和边界 bound
    result, status, bound = ret.d1, ret.i1, ret.d2
    # 调用 get_result 函数以获取最终结果，参数包括分布类型 "nctdtrit"、参数名列表 argnames、计算结果 result、状态 status、边界 bound，最后一个参数为 1
    return get_result("nctdtrit", argnames, result, status, bound, 1)
cdef inline double nrdtrimn(double p, double std, double x) noexcept nogil:
    cdef:
        double q = 1.0 - p
        double result, bound  # 定义结果和边界变量
        int status = 10  # 设置默认状态值为10
        char *argnames[4]  # 定义字符指针数组，用于存储参数名
        TupleDID ret  # 定义元组类型 ret，包含一个 double 和一个 int

    if isnan(p) or isnan(std) or isnan(x):  # 检查输入参数是否为 NaN，若是则返回 NaN
      return NAN

    argnames[0] = "p"  # 设置参数名数组的第一个元素
    argnames[1] = "q"  # 设置参数名数组的第二个元素
    argnames[2] = "x"  # 设置参数名数组的第三个元素
    argnames[3] = "std"  # 设置参数名数组的第四个元素

    ret = cdfnor_which3(p, q, x, std)  # 调用 cdfnor_which3 函数，返回元组结果
    result, status, bound = ret.d1, ret.i1, ret.d2  # 解包元组，获取结果、状态和边界
    return get_result("nrdtrimn", argnames, result, status, bound, 1)  # 返回使用 get_result 处理后的结果


cdef inline double nrdtrisd(double mn, double p, double x) noexcept nogil:
    cdef:
        double q = 1.0 - p
        double result, bound  # 定义结果和边界变量
        int status = 10  # 设置默认状态值为10
        char *argnames[4]  # 定义字符指针数组，用于存储参数名
        TupleDID ret  # 定义元组类型 ret，包含一个 double 和一个 int

    if isnan(mn) or isnan(p) or isnan(x):  # 检查输入参数是否为 NaN，若是则返回 NaN
      return NAN

    argnames[0] = "p"  # 设置参数名数组的第一个元素
    argnames[1] = "q"  # 设置参数名数组的第二个元素
    argnames[2] = "x"  # 设置参数名数组的第三个元素
    argnames[3] = "mn"  # 设置参数名数组的第四个元素

    ret = cdfnor_which4(p, q, x, mn)  # 调用 cdfnor_which4 函数，返回元组结果
    result, status, bound = ret.d1, ret.i1, ret.d2  # 解包元组，获取结果、状态和边界
    return get_result("nrdtrisd", argnames, result, status, bound, 1)  # 返回使用 get_result 处理后的结果


cdef inline double pdtrik(double p, double xlam) noexcept nogil:
    cdef:
        double q = 1.0 - p
        double result, bound  # 定义结果和边界变量
        int status = 10  # 设置默认状态值为10
        char *argnames[3]  # 定义字符指针数组，用于存储参数名
        TupleDID ret  # 定义元组类型 ret，包含一个 double 和一个 int

    if isnan(p) or isnan(xlam):  # 检查输入参数是否为 NaN，若是则返回 NaN
      return NAN

    argnames[0] = "p"  # 设置参数名数组的第一个元素
    argnames[1] = "q"  # 设置参数名数组的第二个元素
    argnames[2] = "xlam"  # 设置参数名数组的第三个元素

    ret = cdfpoi_which2(p, q, xlam)  # 调用 cdfpoi_which2 函数，返回元组结果
    result, status, bound = ret.d1, ret.i1, ret.d2  # 解包元组，获取结果、状态和边界
    return get_result("pdtrik", argnames, result, status, bound, 1)  # 返回使用 get_result 处理后的结果


cdef inline double stdtr(double df, double t) noexcept nogil:
    cdef:
        double result, _, bound  # 定义结果和边界变量
        int status = 10  # 设置默认状态值为10
        char *argnames[2]  # 定义字符指针数组，用于存储参数名
        TupleDDID ret  # 定义元组类型 ret，包含两个 double 和一个 int

    argnames[0] = "t"  # 设置参数名数组的第一个元素
    argnames[1] = "df"  # 设置参数名数组的第二个元素

    if isinf(df) and df > 0:  # 如果 df 是正无穷大，则处理 t 的情况
        return NAN if isnan(t) else cephes_ndtr_wrap(t)  # 若 t 是 NaN 则返回 NaN，否则调用 cephes_ndtr_wrap 处理 t

    if isnan(df) or isnan(t):  # 检查输入参数是否为 NaN，若是则返回 NaN
      return NAN

    ret = cdft_which1(t, df)  # 调用 cdft_which1 函数，返回元组结果
    result, status, bound = ret.d1, ret.i1, ret.d3  # 解包元组，获取结果、状态和边界
    return get_result("stdtr", argnames, result, status, bound, 1)  # 返回使用 get_result 处理后的结果


cdef inline double stdtridf(double p, double t) noexcept nogil:
    cdef:
        double q = 1.0 - p
        double result, bound  # 定义结果和边界变量
        int status = 10  # 设置默认状态值为10
        char *argnames[3]  # 定义字符指针数组，用于存储参数名
        TupleDID ret  # 定义元组类型 ret，包含一个 double 和一个 int

    if isnan(p) or isnan(q) or isnan(t):  # 检查输入参数是否为 NaN，若是则返回 NaN
        return NAN

    argnames[0] = "p"  # 设置参数名数组的第一个元素
    argnames[1] = "q"  # 设置参数名数组的第二个元素
    argnames[2] = "t"  # 设置参数名数组的第三个元素

    ret = cdft_which3(p, q, t)  # 调用 cdft_which3 函数，返回元组结果
    result, status, bound = ret.d1, ret.i1, ret.d2  # 解包元组，获取结果、状态和边界
    return get_result("stdtridf", argnames, result, status, bound, 1)  # 返回使用 get_result 处理后的结果


cdef inline double stdtrit(double df, double p) noexcept nogil:
    cdef:
        double q = 1.0 - p
        double result, bound  # 定义结果和边界变量
        int status = 10  # 设置默认状态值为10
        char *argnames[3]  # 定义字符指针数组，用于存储参数名
        TupleDID ret  # 定义元组类型 ret，包含一个 double 和一个 int

    if isinf(df) and df > 0:  # 如果 df 是正无穷大，则处理 p 的情况
        return NAN if isnan(p) else cephes_ndtri_wrap(p)  # 若 p 是 NaN 则返回 NaN，否则调用 cephes_ndtri_wrap 处理 p

    if isnan(p) or isnan(df):  # 检查输入参数是否为 NaN，若是则返回 NaN
      return NAN

    argnames[0] = "p"  # 设置参数名数组的第一个元素
    argnames[1] = "q"  # 设置参数名数组的第二个元素
    argnames[2] = "df"  # 设置参数名数组的第三个元素

    ret = cdft_which2(p, q, df)  # 调用 cdft_which2 函数，返回元组结果
    result, status, bound = ret.d1, ret.i
    # 从返回的对象 `ret` 中提取三个值，分别赋给 `result`, `status`, `bound`
    result, status, bound = ret.d1, ret.i1, ret.d2
    # 调用函数 `get_result`，传入参数 "stdtrit", argnames, result, status, bound, 1，并返回其结果
    return get_result("stdtrit", argnames, result, status, bound, 1)
```