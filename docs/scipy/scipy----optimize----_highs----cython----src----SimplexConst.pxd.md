# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\SimplexConst.pxd`

```
# cython: language_level=3

# 从 libcpp 库中导入 bool 类型
from libcpp cimport bool

# 从 SimplexConst.h 头文件中声明外部 C 函数或变量，使用 nogil 模式
cdef extern from "SimplexConst.h" nogil:

    # 定义 SimplexAlgorithm 枚举类型，包含 PRIMAL 和 DUAL 两个枚举值
    cdef enum SimplexAlgorithm:
        PRIMAL "SimplexAlgorithm::kPrimal" = 0
        DUAL "SimplexAlgorithm::kDual"

    # 定义 SimplexStrategy 枚举类型，包含多个策略值
    cdef enum SimplexStrategy:
        SIMPLEX_STRATEGY_MIN "SimplexStrategy::kSimplexStrategyMin" = 0
        SIMPLEX_STRATEGY_CHOOSE "SimplexStrategy::kSimplexStrategyChoose" = SIMPLEX_STRATEGY_MIN
        SIMPLEX_STRATEGY_DUAL "SimplexStrategy::kSimplexStrategyDual"
        SIMPLEX_STRATEGY_DUAL_PLAIN "SimplexStrategy::kSimplexStrategyDualPlain" = SIMPLEX_STRATEGY_DUAL
        SIMPLEX_STRATEGY_DUAL_TASKS "SimplexStrategy::kSimplexStrategyDualTasks"
        SIMPLEX_STRATEGY_DUAL_MULTI "SimplexStrategy::kSimplexStrategyDualMulti"
        SIMPLEX_STRATEGY_PRIMAL "SimplexStrategy::kSimplexStrategyPrimal"
        SIMPLEX_STRATEGY_MAX "SimplexStrategy::kSimplexStrategyMax" = SIMPLEX_STRATEGY_PRIMAL
        SIMPLEX_STRATEGY_NUM "SimplexStrategy::kSimplexStrategyNum"

    # 定义 SimplexCrashStrategy 枚举类型，包含多个崩溃处理策略值
    cdef enum SimplexCrashStrategy:
        SIMPLEX_CRASH_STRATEGY_MIN "SimplexCrashStrategy::kSimplexCrashStrategyMin" = 0
        SIMPLEX_CRASH_STRATEGY_OFF "SimplexCrashStrategy::kSimplexCrashStrategyOff" = SIMPLEX_CRASH_STRATEGY_MIN
        SIMPLEX_CRASH_STRATEGY_LTSSF_K "SimplexCrashStrategy::kSimplexCrashStrategyLtssfK"
        SIMPLEX_CRASH_STRATEGY_LTSSF "SimplexCrashStrategy::kSimplexCrashStrategyLtssf" = SIMPLEX_CRASH_STRATEGY_LTSSF_K
        SIMPLEX_CRASH_STRATEGY_BIXBY "SimplexCrashStrategy::kSimplexCrashStrategyBixby"
        SIMPLEX_CRASH_STRATEGY_LTSSF_PRI "SimplexCrashStrategy::kSimplexCrashStrategyLtssfPri"
        SIMPLEX_CRASH_STRATEGY_LTSF_K "SimplexCrashStrategy::kSimplexCrashStrategyLtsfK"
        SIMPLEX_CRASH_STRATEGY_LTSF_PRI "SimplexCrashStrategy::kSimplexCrashStrategyLtsfPri"
        SIMPLEX_CRASH_STRATEGY_LTSF "SimplexCrashStrategy::kSimplexCrashStrategyLtsf"
        SIMPLEX_CRASH_STRATEGY_BIXBY_NO_NONZERO_COL_COSTS "SimplexCrashStrategy::kSimplexCrashStrategyBixbyNoNonzeroColCosts"
        SIMPLEX_CRASH_STRATEGY_BASIC "SimplexCrashStrategy::kSimplexCrashStrategyBasic"
        SIMPLEX_CRASH_STRATEGY_TEST_SING "SimplexCrashStrategy::kSimplexCrashStrategyTestSing"
        SIMPLEX_CRASH_STRATEGY_MAX "SimplexCrashStrategy::kSimplexCrashStrategyMax" = SIMPLEX_CRASH_STRATEGY_TEST_SING
    # 定义枚举 SimplexEdgeWeightStrategy，表示单纯形法边权重策略
    cdef enum SimplexEdgeWeightStrategy:
        # 使用枚举成员名和数值表示，与 C++ 代码中的枚举名称和数值相对应
        SIMPLEX_EDGE_WEIGHT_STRATEGY_MIN "SimplexEdgeWeightStrategy::kSimplexEdgeWeightStrategyMin" = -1
        SIMPLEX_EDGE_WEIGHT_STRATEGY_CHOOSE "SimplexEdgeWeightStrategy::kSimplexEdgeWeightStrategyChoose" = SIMPLEX_EDGE_WEIGHT_STRATEGY_MIN
        SIMPLEX_EDGE_WEIGHT_STRATEGY_DANTZIG "SimplexEdgeWeightStrategy::kSimplexEdgeWeightStrategyDantzig"
        SIMPLEX_EDGE_WEIGHT_STRATEGY_DEVEX "SimplexEdgeWeightStrategy::kSimplexEdgeWeightStrategyDevex"
        SIMPLEX_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE "SimplexEdgeWeightStrategy::kSimplexEdgeWeightStrategySteepestEdge"
        SIMPLEX_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE_UNIT_INITIAL "SimplexEdgeWeightStrategy::kSimplexEdgeWeightStrategySteepestEdgeUnitInitial"
        SIMPLEX_EDGE_WEIGHT_STRATEGY_MAX "SimplexEdgeWeightStrategy::kSimplexEdgeWeightStrategyMax" = SIMPLEX_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE_UNIT_INITIAL

    # 定义枚举 SimplexPriceStrategy，表示单纯形法价格策略
    cdef enum SimplexPriceStrategy:
        # 使用枚举成员名和数值表示，枚举成员之间的数值依次递增
        SIMPLEX_PRICE_STRATEGY_MIN = 0
        SIMPLEX_PRICE_STRATEGY_COL = SIMPLEX_PRICE_STRATEGY_MIN
        SIMPLEX_PRICE_STRATEGY_ROW
        SIMPLEX_PRICE_STRATEGY_ROW_SWITCH
        SIMPLEX_PRICE_STRATEGY_ROW_SWITCH_COL_SWITCH
        SIMPLEX_PRICE_STRATEGY_MAX = SIMPLEX_PRICE_STRATEGY_ROW_SWITCH_COL_SWITCH

    # 定义枚举 SimplexDualChuzcStrategy，表示单纯形法双对偶乔兹基策略
    cdef enum SimplexDualChuzcStrategy:
        # 使用枚举成员名和数值表示，枚举成员之间的数值依次递增
        SIMPLEX_DUAL_CHUZC_STRATEGY_MIN = 0
        SIMPLEX_DUAL_CHUZC_STRATEGY_CHOOSE = SIMPLEX_DUAL_CHUZC_STRATEGY_MIN
        SIMPLEX_DUAL_CHUZC_STRATEGY_QUAD
        SIMPLEX_DUAL_CHUZC_STRATEGY_HEAP
        SIMPLEX_DUAL_CHUZC_STRATEGY_BOTH
        SIMPLEX_DUAL_CHUZC_STRATEGY_MAX = SIMPLEX_DUAL_CHUZC_STRATEGY_BOTH

    # 定义枚举 InvertHint，表示单纯形法求逆提示
    cdef enum InvertHint:
        # 使用枚举成员名和数值表示，枚举成员之间的数值依次递增
        INVERT_HINT_NO = 0
        INVERT_HINT_UPDATE_LIMIT_REACHED
        INVERT_HINT_SYNTHETIC_CLOCK_SAYS_INVERT
        INVERT_HINT_POSSIBLY_OPTIMAL
        INVERT_HINT_POSSIBLY_PRIMAL_UNBOUNDED
        INVERT_HINT_POSSIBLY_DUAL_UNBOUNDED
        INVERT_HINT_POSSIBLY_SINGULAR_BASIS
        INVERT_HINT_PRIMAL_INFEASIBLE_IN_PRIMAL_SIMPLEX
        INVERT_HINT_CHOOSE_COLUMN_FAIL
        INVERT_HINT_Count

    # 定义枚举 DualEdgeWeightMode，表示对偶单纯形法边权重模式
    cdef enum DualEdgeWeightMode:
        # 使用枚举成员名和数值表示，与 C++ 代码中的枚举名称和数值相对应
        DANTZIG "DualEdgeWeightMode::DANTZIG" = 0
        DEVEX "DualEdgeWeightMode::DEVEX"
        STEEPEST_EDGE "DualEdgeWeightMode::STEEPEST_EDGE"
        Count "DualEdgeWeightMode::Count"

    # 定义枚举 PriceMode，表示价格模式
    cdef enum PriceMode:
        # 使用枚举成员名和数值表示，与 C++ 代码中的枚举名称和数值相对应
        ROW "PriceMode::ROW" = 0
        COL "PriceMode::COL"

    # 定义常量 PARALLEL_THREADS_DEFAULT
    const int PARALLEL_THREADS_DEFAULT

    # 定义常量 DUAL_TASKS_MIN_THREADS
    const int DUAL_TASKS_MIN_THREADS

    # 定义常量 DUAL_MULTI_MIN_THREADS
    const int DUAL_MULTI_MIN_THREADS

    # 定义常量 invert_if_row_out_negative
    const bool invert_if_row_out_negative

    # 定义常量 NONBASIC_FLAG_TRUE
    const int NONBASIC_FLAG_TRUE

    # 定义常量 NONBASIC_FLAG_FALSE
    const int NONBASIC_FLAG_FALSE

    # 定义常量 NONBASIC_MOVE_UP
    const int NONBASIC_MOVE_UP

    # 定义常量 NONBASIC_MOVE_DN
    const int NONBASIC_MOVE_DN

    # 定义常量 NONBASIC_MOVE_ZE
    const int NONBASIC_MOVE_ZE
```