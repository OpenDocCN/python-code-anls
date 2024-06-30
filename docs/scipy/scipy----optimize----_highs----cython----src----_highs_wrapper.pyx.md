# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\_highs_wrapper.pyx`

```
# 设置 Cython 的语言级别为3
# 导入必要的库和模块
import numpy as np
cimport numpy as np  # 导入 Cython 版本的 numpy 模块
from scipy.optimize import OptimizeWarning  # 导入优化警告相关模块
from warnings import warn  # 导入警告模块中的 warn 函数
import numbers  # 导入 numbers 模块，用于处理数字类型

# 导入 C++ 标准库中的一些数据结构和类型定义
from libcpp.string cimport string
from libcpp.map cimport map as cppmap
from libcpp.cast cimport reinterpret_cast

# 从自定义模块中导入常量和枚举类型
from .HConst cimport (
    HIGHS_CONST_INF,

    HighsModelStatus,
    HighsModelStatusNOTSET,
    HighsModelStatusMODEL_ERROR,
    HighsModelStatusOPTIMAL,
    HighsModelStatusREACHED_TIME_LIMIT,
    HighsModelStatusREACHED_ITERATION_LIMIT,

    HighsOptionTypeBOOL,
    HighsOptionTypeINT,
    HighsOptionTypeDOUBLE,
    HighsOptionTypeSTRING,

    HighsBasisStatus,
    HighsBasisStatusLOWER,
    HighsBasisStatusUPPER,

    MatrixFormatkColwise,
    HighsVarType,
)

# 从自定义模块中导入 Highs 类和与其相关的状态定义
from .Highs cimport Highs
from .HighsStatus cimport (
    HighsStatus,
    highsStatusToString,
    HighsStatusError,
    HighsStatusWarning,
    HighsStatusOK,
)

# 从自定义模块中导入线性规划相关的类和结构体定义
from .HighsLp cimport (
    HighsLp,
    HighsSolution,
    HighsBasis,
)

# 从自定义模块中导入 HighsInfo 类型
from .HighsInfo cimport HighsInfo

# 从自定义模块中导入 HighsOptions 类型和相关选项记录定义
from .HighsOptions cimport (
    HighsOptions,
    OptionRecord,
    OptionRecordBool,
    OptionRecordInt,
    OptionRecordDouble,
    OptionRecordString,
)

# 导入自定义模块中的函数，用于将基变量状态转换为字符串表示
from .HighsModelUtils cimport utilBasisStatusToString

# 初始化 numpy 数组接口
np.import_array()

# 用于引用默认值和边界的选项；
# 创建一个映射以便快速查找
cdef HighsOptions _ref_opts
cdef cppmap[string, OptionRecord*] _ref_opt_lookup
cdef OptionRecord * _r = NULL
# 遍历 _ref_opts 中的记录，并建立名称到记录指针的映射
for _r in _ref_opts.records:
    _ref_opt_lookup[_r.name] = _r


# 定义一个函数，生成关于选项警告的字符串
cdef str _opt_warning(string name, val, valid_set=None):
    cdef OptionRecord * r = _ref_opt_lookup[name]

    # 如果选项类型为 BOOL
    if r.type == HighsOptionTypeBOOL:
        # 获取默认值
        default_value = (<OptionRecordBool*> r).default_value
        # 返回警告信息，说明输入的值不合法，使用默认值替代
        return ('Option "%s" is "%s", but only True or False is allowed. '
                'Using default: %s.' % (name.decode(), str(val), default_value))

    # 如果选项类型为 INT
    if r.type == HighsOptionTypeINT:
        # 获取上下界和默认值
        lower_bound = int((<OptionRecordInt*> r).lower_bound)
        upper_bound = int((<OptionRecordInt*> r).upper_bound)
        default_value = int((<OptionRecordInt*> r).default_value)
        # 如果范围比较小，使用集合的形式表示
        if upper_bound - lower_bound < 10:
            int_range = str(set(range(lower_bound, upper_bound + 1)))
        else:
            int_range = '[%d, %d]' % (lower_bound, upper_bound)
        # 返回警告信息，说明输入的值不合法，使用默认值替代
        return ('Option "%s" is "%s", but only values in %s are allowed. '
                'Using default: %d.' % (name.decode(), str(val), int_range, default_value))

    # 如果选项类型为 DOUBLE
    if r.type == HighsOptionTypeDOUBLE:
        # 获取上下界和默认值
        lower_bound = (<OptionRecordDouble*> r).lower_bound
        upper_bound = (<OptionRecordDouble*> r).upper_bound
        default_value = (<OptionRecordDouble*> r).default_value
        # 返回警告信息，说明输入的值不合法，使用默认值替代
        return ('Option "%s" is "%s", but only values in (%g, %g) are allowed. '
                'Using default: %g.' % (name.decode(), str(val), lower_bound, upper_bound, default_value))

    # 如果选项类型为 STRING
    if r.type == HighsOptionTypeSTRING:
        # TODO: 此处应添加字符串类型选项的处理逻辑
        pass
    # 如果选项类型为字符串类型
    if r.type == HighsOptionTypeSTRING:
        # 如果有有效的值集合
        if valid_set is not None:
            # 描述信息，指示只有特定集合中的值是允许的
            descr = 'but only values in %s are allowed. ' % str(set(valid_set))
        else:
            # 描述信息，指示这是一个无效的值
            descr = 'but this is an invalid value. %s. ' % r.description.decode()
        # 默认值为字符串类型选项的默认值
        default_value = (<OptionRecordString*> r).default_value.decode()
        # 返回错误消息，指出选项名称、当前值、描述和默认值的情况
        return ('Option "%s" is "%s", '
                '%s'
                'Using default: %s.' % (name.decode(), str(val), descr, default_value))

    # 如果代码执行到这里，表示选项类型不是字符串类型，理论上不应该到达这里
    return('Option "%s" is "%s", but this is not a valid value. '
           'See documentation for valid options. '
           'Using default.' % (name.decode(), str(val)))
# 将给定的选项从字典应用到 Highs 对象中
cdef void apply_options(dict options, Highs & highs):
    '''Take options from dictionary and apply to HiGHS object.'''

    # 初始化错误检查状态
    cdef HighsStatus opt_status = HighsStatusOK

    # 处理所有整数类型的选项
    for opt in set([
            'allowed_simplex_cost_scale_factor',                # 允许的单纯形成本比例因子
            'allowed_simplex_matrix_scale_factor',              # 允许的单纯形矩阵比例因子
            'dual_simplex_cleanup_strategy',                    # 对偶单纯形清理策略
            'ipm_iteration_limit',                              # 内点法迭代限制
            'keep_n_rows',                                      # 保留行数
            'threads',                                          # 线程数
            'mip_max_nodes',                                    # MIP 最大节点数
            'highs_debug_level',                                # HiGHS 调试级别
            'simplex_crash_strategy',                           # 单纯形崩溃策略
            'simplex_dual_edge_weight_strategy',                # 单纯形对偶边权重策略
            'simplex_dualise_strategy',                         # 单纯形对偶化策略
            'simplex_iteration_limit',                          # 单纯形迭代限制
            'simplex_permute_strategy',                         # 单纯形置换策略
            'simplex_price_strategy',                           # 单纯形价格策略
            'simplex_primal_edge_weight_strategy',              # 单纯形原始边权重策略
            'simplex_scale_strategy',                           # 单纯形缩放策略
            'simplex_strategy',                                 # 单纯形策略
            'simplex_update_limit',                             # 单纯形更新限制
            'small_matrix_value',                               # 小矩阵值
    ]):
        val = options.get(opt, None)
        if val is not None:
            if not isinstance(val, int):
                # 如果值不是整数，发出警告
                warn(_opt_warning(opt.encode(), val), OptimizeWarning)
            else:
                # 设置整数型选项的值到 Highs 对象中
                opt_status = highs.setHighsOptionValueInt(opt.encode(), val)
                if opt_status != HighsStatusOK:
                    # 如果设置失败，发出警告
                    warn(_opt_warning(opt.encode(), val), OptimizeWarning)
                else:
                    # 如果设置成功且选项是 "threads"，则重置全局调度器
                    if opt == "threads":
                        highs.resetGlobalScheduler(blocking=True)

    # 处理所有浮点数类型的选项
    for opt in set([
            'dual_feasibility_tolerance',                       # 对偶可行性容忍度
            'dual_objective_value_upper_bound',                 # 对偶目标值上界
            'dual_simplex_cost_perturbation_multiplier',       # 对偶单纯形成本扰动乘子
            'dual_steepest_edge_weight_log_error_threshhold',   # 对偶最陡边权重对数误差阈值
            'infinite_bound',                                   # 无限边界
            'infinite_cost',                                    # 无限成本
            'ipm_optimality_tolerance',                         # 内点法最优性容忍度
            'large_matrix_value',                               # 大矩阵值
            'primal_feasibility_tolerance',                     # 原始可行性容忍度
            'simplex_initial_condition_tolerance',              # 单纯形初始条件容忍度
            'small_matrix_value',                               # 小矩阵值
            'start_crossover_tolerance',                        # 启动交叉容忍度
            'time_limit',                                       # 时间限制
            'mip_rel_gap'                                       # MIP 相对间隙
    ]):
        val = options.get(opt, None)
        if val is not None:
            if not isinstance(val, numbers.Number):
                # 如果值不是数字类型，发出警告
                warn(_opt_warning(opt.encode(), val), OptimizeWarning)
            else:
                # 设置浮点数型选项的值到 Highs 对象中
                opt_status = highs.setHighsOptionValueDbl(opt.encode(), val)
                if opt_status != HighsStatusOK:
                    # 如果设置失败，发出警告
                    warn(_opt_warning(opt.encode(), val), OptimizeWarning)

    # 处理所有字符串类型的选项
    for opt in set(['solver']):
        val = options.get(opt, None)
        if val is not None:
            if not isinstance(val, str):
                # 如果值不是字符串类型，发出警告
                warn(_opt_warning(opt.encode(), val), OptimizeWarning)
            else:
                # 设置字符串型选项的值到 Highs 对象中
                opt_status = highs.setHighsOptionValueStr(opt.encode(), val.encode())
                if opt_status != HighsStatusOK:
                    # 如果设置失败，发出警告
                    warn(_opt_warning(opt.encode(), val), OptimizeWarning)
    # 处理布尔型选项转换为字符串形式
    for opt in set([
            'parallel',
            'presolve',
    ]):
        # 获取选项对应的值
        val = options.get(opt, None)
        # 如果值不为 None
        if val is not None:
            # 如果值是布尔型
            if isinstance(val, bool):
                # 根据布尔值设置相应的字符串值
                if val:
                    val0 = b'on'
                else:
                    val0 = b'off'
                # 调用高级优化库的函数设置选项值，并检查返回状态
                opt_status = highs.setHighsOptionValueStr(opt.encode(), val0)
                # 如果设置失败，发出警告
                if opt_status != HighsStatusOK:
                    warn(_opt_warning(opt.encode(), val, valid_set=[True, False]), OptimizeWarning)
            else:
                # 如果值不是布尔型，发出相应的警告
                warn(_opt_warning(opt.encode(), val, valid_set=[True, False]), OptimizeWarning)


    # 处理直接的布尔型选项
    for opt in set([
            'less_infeasible_DSE_check',
            'less_infeasible_DSE_choose_row',
            'log_to_console',
            'mps_parser_type_free',
            'output_flag',
            'run_as_hsol',
            'run_crossover',
            'simplex_initial_condition_check',
            'use_original_HFactor_logic',
    ]):
        # 获取选项对应的值
        val = options.get(opt, None)
        # 如果值不为 None
        if val is not None:
            # 如果值是 True 或者 False
            if val in [True, False]:
                # 调用高级优化库的函数设置布尔型选项值，并检查返回状态
                opt_status = highs.setHighsOptionValueBool(opt.encode(), val)
                # 如果设置失败，发出警告
                if opt_status != HighsStatusOK:
                    warn(_opt_warning(opt.encode(), val), OptimizeWarning)
            else:
                # 如果值既不是 True 也不是 False，发出相应的警告
                warn(_opt_warning(opt.encode(), val), OptimizeWarning)
# 定义一个指针类型别名，指向 HighsVarType 类型的指针
ctypedef HighsVarType* HighsVarType_ptr

# 定义一个函数 _highs_wrapper，用于使用 HiGHS 解决线性规划问题
def _highs_wrapper(
        double[::1] c,                 # 目标函数的系数数组
        int[::1] astart,               # CSC 格式中列偏移索引数组
        int[::1] aindex,               # CSC 格式中行索引数组
        double[::1] avalue,            # 稀疏矩阵中的数值数组
        double[::1] lhs,               # 不等式约束的左边界数组
        double[::1] rhs,               # 不等式约束的右边界数组
        double[::1] lb,                # 变量下界数组
        double[::1] ub,                # 变量上界数组
        np.uint8_t[::1] integrality,   # 变量整数性数组
        dict options):                 # 选项字典，控制求解行为和输出

    '''Solve linear programs using HiGHS [1]_.

    Assume problems of the form:

        MIN c.T @ x
        s.t. lhs <= A @ x <= rhs
             lb <= x <= ub

    Parameters
    ----------
    c : 1-D array, (n,)
        目标函数系数数组。
    astart : 1-D array
        CSC 格式中列偏移索引数组。
    aindex : 1-D array
        CSC 格式中行索引数组。
    avalue : 1-D array
        稀疏矩阵中的数值数组。
    lhs : 1-D array (or None), (m,)
        不等式约束的左边界值数组。
        如果 ``lhs=None``，则假定为 ``-inf`` 数组。
    rhs : 1-D array, (m,)
        不等式约束的右边界值数组。
    lb : 1-D array (or None), (n,)
        变量 x 的下界数组。
        如果 ``lb=None``，则假定为全为 `0` 的数组。
    ub : 1-D array (or None), (n,)
        变量 x 的上界数组。
        如果 ``ub=None``，则假定为 ``inf`` 数组。
    Returns
    -------
    res : dict
        返回求解结果的字典

        如果模型状态是 OPTIMAL、REACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND、
        REACHED_TIME_LIMIT、REACHED_ITERATION_LIMIT 中的一种：

            - ``status`` : int
                模型状态码。

            - ``message`` : str
                与模型状态码相关的消息。

            - ``x`` : list
                解变量列表。

            - ``slack`` : list
                松弛变量列表。

            - ``lambda`` : list
                与约束 Ax = b 相关的拉格朗日乘子。

            - ``s`` : list
                与约束 x >= 0 相关的拉格朗日乘子。

            - ``fun``
                最终的目标函数值。

            - ``simplex_nit`` : int
                简单形法求解器完成的迭代次数。

            - ``ipm_nit`` : int
                内点法求解器完成的迭代次数。

        如果模型状态不是上述状态之一：

            - ``status`` : int
                模型状态码。

            - ``message`` : str
                与模型状态码相关的消息。

    Notes
    -----
    如果 ``options['write_solution_to_file']`` 是 ``True``，
    但 ``options['solution_file']`` 未设置或为 ``''``，
    则解将打印到标准输出。

    如果达到任何迭代限制，将无法获得解。

    如果用户设置的任何选项值不正确，将引发 ``OptimizeWarning``。

    References
    ----------
    .. [1] https://highs.dev/
    '''
    # 导入 HiGHS 的选项说明文档链接，用于参考
    .. [2] https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.html
    '''

    # 使用 Cython 的 cdef 定义整型变量，分别存储列数、行数、非零元素数和整数性变量数
    cdef int numcol = c.size
    cdef int numrow = rhs.size
    cdef int numnz = avalue.size
    cdef int numintegrality = integrality.size

    # 创建 HighsLp 对象 lp，并设置其基本属性
    cdef HighsLp lp
    lp.num_col_ = numcol
    lp.num_row_ = numrow
    lp.a_matrix_.num_col_ = numcol
    lp.a_matrix_.num_row_ = numrow
    lp.a_matrix_.format_ = MatrixFormatkColwise

    # 根据列数分配空间给列相关属性
    lp.col_cost_.resize(numcol)
    lp.col_lower_.resize(numcol)
    lp.col_upper_.resize(numcol)

    # 根据行数分配空间给行相关属性
    lp.row_lower_.resize(numrow)
    lp.row_upper_.resize(numrow)
    lp.a_matrix_.start_.resize(numcol + 1)
    lp.a_matrix_.index_.resize(numnz)
    lp.a_matrix_.value_.resize(numnz)

    # 只有在整数性变量数大于零时，设置 lp 对象的整数性变量属性
    cdef HighsVarType * integrality_ptr = NULL
    if numintegrality > 0:
        lp.integrality_.resize(numintegrality)
        integrality_ptr = reinterpret_cast[HighsVarType_ptr](&integrality[0])
        lp.integrality_.assign(integrality_ptr, integrality_ptr + numcol)

    # 显式创建指针以传递给 HiGHS C++ API；确保不访问空内存视图
    cdef:
        double * colcost_ptr = NULL
        double * collower_ptr = NULL
        double * colupper_ptr = NULL
        double * rowlower_ptr = NULL
        double * rowupper_ptr = NULL
        int * astart_ptr = NULL
        int * aindex_ptr = NULL
        double * avalue_ptr = NULL
        
    # 如果行数大于零，设置行下界和上界，并将其分配给 lp 对象
    if numrow > 0:
        rowlower_ptr = &lhs[0]
        rowupper_ptr = &rhs[0]
        lp.row_lower_.assign(rowlower_ptr, rowlower_ptr + numrow)
        lp.row_upper_.assign(rowupper_ptr, rowupper_ptr + numrow)
    else:
        # 如果行数为零，清空 lp 对象的行下界和上界
        lp.row_lower_.empty()
        lp.row_upper_.empty()
        
    # 如果列数大于零，设置列成本、下界和上界，并将其分配给 lp 对象
    if numcol > 0:
        colcost_ptr = &c[0]
        collower_ptr = &lb[0]
        colupper_ptr = &ub[0]
        lp.col_cost_.assign(colcost_ptr, colcost_ptr + numcol)
        lp.col_lower_.assign(collower_ptr, collower_ptr + numcol)
        lp.col_upper_.assign(colupper_ptr, colupper_ptr + numcol)
    else:
        # 如果列数为零，清空 lp 对象的列成本、下界、上界和整数性变量属性
        lp.col_cost_.empty()
        lp.col_lower_.empty()
        lp.col_upper_.empty()
        lp.integrality_.empty()
        
    # 如果非零元素数大于零，设置 A 矩阵的起始位置、索引和值，并将其分配给 lp 对象
    if numnz > 0:
        astart_ptr = &astart[0]
        aindex_ptr = &aindex[0]
        avalue_ptr = &avalue[0]
        lp.a_matrix_.start_.assign(astart_ptr, astart_ptr + numcol + 1)
        lp.a_matrix_.index_.assign(aindex_ptr, aindex_ptr + numnz)
        lp.a_matrix_.value_.assign(avalue_ptr, avalue_ptr + numnz)
    else:
        # 如果非零元素数为零，清空 lp 对象的 A 矩阵的起始位置、索引和值
        lp.a_matrix_.start_.empty()
        lp.a_matrix_.index_.empty()
        lp.a_matrix_.value_.empty()

    # 创建 Highs 对象 highs，并应用 options 中的设置
    cdef Highs highs
    apply_options(options, highs)

    # 创建 HighsModelStatus 和 HighsStatus 对象以进行模型传递初始化
    cdef HighsModelStatus err_model_status = HighsModelStatusNOTSET
    cdef HighsStatus init_status = highs.passModel(lp)
    # 如果初始化状态不是 HighsStatusOK
    if init_status != HighsStatusOK:
        # 并且初始化状态不是 HighsStatusWarning
        if init_status != HighsStatusWarning:
            # 将错误模型状态设置为 HighsModelStatusMODEL_ERROR
            err_model_status = HighsModelStatusMODEL_ERROR
            # 返回包含错误状态和消息的字典
            return {
                'status': <int> err_model_status,
                'message': highs.modelStatusToString(err_model_status).decode(),
            }

    # 解决线性规划问题
    cdef HighsStatus run_status = highs.run()
    # 如果运行状态是 HighsStatusError
    if run_status == HighsStatusError:
        # 返回包含模型状态和消息的字典
        return {
            'status': <int> highs.getModelStatus(),
            'message': highsStatusToString(run_status).decode(),
        }

    # 从解决方案中提取所需的信息
    cdef HighsModelStatus model_status = highs.getModelStatus()

    # 如果可能需要一个信息对象以及放置解决方案的位置
    cdef HighsInfo info = highs.getHighsInfo()  # 获取信息对象，通常是安全的
    cdef HighsSolution solution
    cdef HighsBasis basis
    cdef double[:, ::1] marg_bnds = np.zeros((2, numcol))  # marg_bnds[0, :]: lower

    # 失败模式：
    #     LP：如果模型状态不是 Optimal，那么读取任何结果都是不安全的（也没有帮助的）
    #    MIP：具有非 Optimal 状态或已超时/达到最大迭代次数
    #             1) 如果不是 Optimal/TimedOut/MaxIter 状态，则没有解决方案
    #             2) 如果是 TimedOut/MaxIter 状态，则可能有一个可行解。
    #                如果目标函数值不是无穷大，则当前解决方案是可行的，并且可以返回。否则，没有解决方案。
    mipFailCondition = model_status not in {
        HighsModelStatusOPTIMAL,
        HighsModelStatusREACHED_TIME_LIMIT,
        HighsModelStatusREACHED_ITERATION_LIMIT,
    } or (model_status in {
        HighsModelStatusREACHED_TIME_LIMIT,
        HighsModelStatusREACHED_ITERATION_LIMIT,
    } and (info.objective_function_value == HIGHS_CONST_INF))
    lpFailCondition = model_status != HighsModelStatusOPTIMAL
    # 如果是 MIP 并且满足 MIP 失败条件，或者是 LP 并且满足 LP 失败条件
    if (highs.getLp().isMip() and mipFailCondition) or (not highs.getLp().isMip() and lpFailCondition):
        # 返回包含模型状态、消息和相关信息的字典
        return {
            'status': <int> model_status,
            'message': f'model_status is {highs.modelStatusToString(model_status).decode()}; '
                       f'primal_status is {utilBasisStatusToString(<HighsBasisStatus> info.primal_solution_status).decode()}',
            'simplex_nit': info.simplex_iteration_count,
            'ipm_nit': info.ipm_iteration_count,
            'fun': None,
            'crossover_nit': info.crossover_iteration_count,
        }
    # 如果模型状态使得可以读取解决方案
    else:
        # Should be safe to read the solution:
        # 获取求解器的解
        solution = highs.getSolution()
        # 获取基
        basis = highs.getBasis()

        # lagrangians for bounds based on column statuses
        # 根据列的状态获取边界的拉格朗日乘子
        for ii in range(numcol):
            if HighsBasisStatusLOWER == basis.col_status[ii]:
                marg_bnds[0, ii] = solution.col_dual[ii]
            elif HighsBasisStatusUPPER == basis.col_status[ii]:
                marg_bnds[1, ii] = solution.col_dual[ii]

        # 构建结果字典
        res = {
            'status': <int> model_status,
            'message': highs.modelStatusToString(model_status).decode(),

            # Primal solution
            # 原始解
            'x': [solution.col_value[ii] for ii in range(numcol)],

            # Ax + s = b => Ax = b - s
            # 注意：这适用于所有约束条件（A_ub 和 A_eq）
            'slack': [rhs[ii] - solution.row_value[ii] for ii in range(numrow)],

            # lambda are the lagrange multipliers associated with Ax=b
            # lambda 是与 Ax=b 相关的拉格朗日乘子
            'lambda': [solution.row_dual[ii] for ii in range(numrow)],
            'marg_bnds': marg_bnds,

            'fun': info.objective_function_value,
            'simplex_nit': info.simplex_iteration_count,
            'ipm_nit': info.ipm_iteration_count,
            'crossover_nit': info.crossover_iteration_count,
        }

        # 如果是整数规划，添加额外信息
        if highs.getLp().isMip():
            res.update({
                'mip_node_count': info.mip_node_count,
                'mip_dual_bound': info.mip_dual_bound,
                'mip_gap': info.mip_gap,
            })

        # 返回结果字典
        return res
```