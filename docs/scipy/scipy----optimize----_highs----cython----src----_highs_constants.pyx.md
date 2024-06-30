# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\_highs_constants.pyx`

```
# 设置 Cython 的语言级别为3
# 导入 HiGHS 中的枚举值和常量

from .HConst cimport (
    HIGHS_CONST_I_INF,  # 无穷大的整数值常量
    HIGHS_CONST_INF,  # 无穷大的浮点数值常量

    HighsDebugLevel_kHighsDebugLevelNone,  # 调试级别：无调试信息
    HighsDebugLevel_kHighsDebugLevelCheap,  # 调试级别：少量调试信息

    HighsModelStatusNOTSET,  # 模型状态：未设置
    HighsModelStatusLOAD_ERROR,  # 模型状态：加载错误
    HighsModelStatusMODEL_ERROR,  # 模型状态：模型错误
    HighsModelStatusMODEL_EMPTY,  # 模型状态：模型为空
    HighsModelStatusPRESOLVE_ERROR,  # 模型状态：预处理错误
    HighsModelStatusSOLVE_ERROR,  # 模型状态：求解错误
    HighsModelStatusPOSTSOLVE_ERROR,  # 模型状态：后处理错误
    HighsModelStatusINFEASIBLE,  # 模型状态：不可行
    HighsModelStatus_UNBOUNDED_OR_INFEASIBLE,  # 模型状态：无界或不可行
    HighsModelStatusUNBOUNDED,  # 模型状态：无界
    HighsModelStatusOPTIMAL,  # 模型状态：最优
    HighsModelStatusREACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND,  # 模型状态：达到对偶目标值上界
    HighsModelStatusREACHED_OBJECTIVE_TARGET,  # 模型状态：达到目标值目标
    HighsModelStatusREACHED_TIME_LIMIT,  # 模型状态：达到时间限制
    HighsModelStatusREACHED_ITERATION_LIMIT,  # 模型状态：达到迭代次数限制

    ObjSenseMINIMIZE,  # 目标函数方向：最小化
    kContinuous,  # 变量类型：连续型
    kInteger,  # 变量类型：整数型
    kSemiContinuous,  # 变量类型：半连续型
    kSemiInteger,  # 变量类型：半整数型
    kImplicitInteger,  # 变量类型：隐式整数型
)
from .HighsIO cimport (
    kInfo,  # 日志类型：信息
    kDetailed,  # 日志类型：详细
    kVerbose,  # 日志类型：详细至极
    kWarning,  # 日志类型：警告
    kError,  # 日志类型：错误
)
from .SimplexConst cimport (
    # 单纯形策略
    SIMPLEX_STRATEGY_CHOOSE,  # 单纯形策略：自动选择
    SIMPLEX_STRATEGY_DUAL,  # 单纯形策略：对偶法
    SIMPLEX_STRATEGY_PRIMAL,  # 单纯形策略：原始法

    # 崩溃策略
    SIMPLEX_CRASH_STRATEGY_OFF,  # 崩溃策略：关闭
    SIMPLEX_CRASH_STRATEGY_BIXBY,  # 崩溃策略：Bixby
    SIMPLEX_CRASH_STRATEGY_LTSF,  # 崩溃策略：LTSF

    # 边权重策略
    SIMPLEX_EDGE_WEIGHT_STRATEGY_CHOOSE,  # 边权重策略：自动选择
    SIMPLEX_EDGE_WEIGHT_STRATEGY_DANTZIG,  # 边权重策略：Dantzig
    SIMPLEX_EDGE_WEIGHT_STRATEGY_DEVEX,  # 边权重策略：Devex
    SIMPLEX_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE,  # 边权重策略：最陡边
)

# 设置常量的别名
CONST_I_INF = HIGHS_CONST_I_INF
CONST_INF = HIGHS_CONST_INF

# 设置调试级别的别名
MESSAGE_LEVEL_NONE = HighsDebugLevel_kHighsDebugLevelNone
MESSAGE_LEVEL_MINIMAL = HighsDebugLevel_kHighsDebugLevelCheap

# 设置日志类型的别名
LOG_TYPE_INFO = <int> kInfo
LOG_TYPE_DETAILED = <int> kDetailed
LOG_TYPE_VERBOSE = <int> kVerbose
LOG_TYPE_WARNING = <int> kWarning
LOG_TYPE_ERROR = <int> kError

# 设置模型状态的别名
MODEL_STATUS_NOTSET = <int> HighsModelStatusNOTSET
MODEL_STATUS_LOAD_ERROR = <int> HighsModelStatusLOAD_ERROR
MODEL_STATUS_MODEL_ERROR = <int> HighsModelStatusMODEL_ERROR
MODEL_STATUS_PRESOLVE_ERROR = <int> HighsModelStatusPRESOLVE_ERROR
MODEL_STATUS_SOLVE_ERROR = <int> HighsModelStatusSOLVE_ERROR
MODEL_STATUS_POSTSOLVE_ERROR = <int> HighsModelStatusPOSTSOLVE_ERROR
MODEL_STATUS_MODEL_EMPTY = <int> HighsModelStatusMODEL_EMPTY
MODEL_STATUS_INFEASIBLE = <int> HighsModelStatusINFEASIBLE
MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE = <int> HighsModelStatus_UNBOUNDED_OR_INFEASIBLE
MODEL_STATUS_UNBOUNDED = <int> HighsModelStatusUNBOUNDED
MODEL_STATUS_OPTIMAL = <int> HighsModelStatusOPTIMAL
MODEL_STATUS_REACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND = <int> HighsModelStatusREACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND
MODEL_STATUS_REACHED_OBJECTIVE_TARGET = <int> HighsModelStatusREACHED_OBJECTIVE_TARGET
MODEL_STATUS_REACHED_TIME_LIMIT = <int> HighsModelStatusREACHED_TIME_LIMIT
MODEL_STATUS_REACHED_ITERATION_LIMIT = <int> HighsModelStatusREACHED_ITERATION_LIMIT

# 设置单纯形策略的别名
HIGHS_SIMPLEX_STRATEGY_CHOOSE = <int> SIMPLEX_STRATEGY_CHOOSE
# 高斯消元法的双重策略
HIGHS_SIMPLEX_STRATEGY_DUAL = <int> SIMPLEX_STRATEGY_DUAL

# 高斯消元法的原始策略
HIGHS_SIMPLEX_STRATEGY_PRIMAL = <int> SIMPLEX_STRATEGY_PRIMAL

# 高斯消元法的崩溃策略：关闭
HIGHS_SIMPLEX_CRASH_STRATEGY_OFF = <int> SIMPLEX_CRASH_STRATEGY_OFF

# 高斯消元法的崩溃策略：Bixby策略
HIGHS_SIMPLEX_CRASH_STRATEGY_BIXBY = <int> SIMPLEX_CRASH_STRATEGY_BIXBY

# 高斯消元法的崩溃策略：LTSF策略
HIGHS_SIMPLEX_CRASH_STRATEGY_LTSF = <int> SIMPLEX_CRASH_STRATEGY_LTSF

# 高斯消元法的边权重策略：自动选择
HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_CHOOSE = <int> SIMPLEX_EDGE_WEIGHT_STRATEGY_CHOOSE

# 高斯消元法的边权重策略：Dantzig策略
HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_DANTZIG = <int> SIMPLEX_EDGE_WEIGHT_STRATEGY_DANTZIG

# 高斯消元法的边权重策略：Devex策略
HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_DEVEX = <int> SIMPLEX_EDGE_WEIGHT_STRATEGY_DEVEX

# 高斯消元法的边权重策略：最陡边策略
HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE = <int> SIMPLEX_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE

# 目标函数的方向：最小化
HIGHS_OBJECTIVE_SENSE_MINIMIZE = <int> ObjSenseMINIMIZE

# 变量类型：连续型变量
HIGHS_VAR_TYPE_CONTINUOUS = <int> kContinuous

# 变量类型：整数型变量
HIGHS_VAR_TYPE_INTEGER = <int> kInteger

# 变量类型：半连续型变量
HIGHS_VAR_TYPE_SEMI_CONTINUOUS = <int> kSemiContinuous

# 变量类型：半整数型变量
HIGHS_VAR_TYPE_SEMI_INTEGER = <int> kSemiInteger

# 变量类型：隐式整数型变量
HIGHS_VAR_TYPE_IMPLICIT_INTEGER = <int> kImplicitInteger
```