# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\HConst.pxd`

```
# 设置Cython编译器语言级别为3
# 导入需要的C++库类型和字符串类型
from libcpp cimport bool
from libcpp.string cimport string

# 从外部头文件"HConst.h"中导入常量，使用`nogil`来声明无GIL的函数
cdef extern from "HConst.h" nogil:

    # 定义常量整数`HIGHS_CONST_I_INF`，表示Highs库中的无限大
    const int HIGHS_CONST_I_INF "kHighsIInf"

    # 定义常量浮点数`HIGHS_CONST_INF`，表示Highs库中的无穷大
    const double HIGHS_CONST_INF "kHighsInf"

    # 定义常量浮点数`kHighsTiny`，表示Highs库中的极小值
    const double kHighsTiny

    # 定义常量浮点数`kHighsZero`，表示Highs库中的零值
    const double kHighsZero

    # 定义常量整数`kHighsThreadLimit`，表示Highs库中的线程限制数
    const int kHighsThreadLimit

    # 定义枚举类型`HighsDebugLevel`，表示Highs调试级别
    cdef enum HighsDebugLevel:
        HighsDebugLevel_kHighsDebugLevelNone "kHighsDebugLevelNone" = 0
        HighsDebugLevel_kHighsDebugLevelCheap
        HighsDebugLevel_kHighsDebugLevelCostly
        HighsDebugLevel_kHighsDebugLevelExpensive
        HighsDebugLevel_kHighsDebugLevelMin "kHighsDebugLevelMin" = HighsDebugLevel_kHighsDebugLevelNone
        HighsDebugLevel_kHighsDebugLevelMax "kHighsDebugLevelMax" = HighsDebugLevel_kHighsDebugLevelExpensive

    # 定义枚举类型`HighsModelStatus`，表示Highs模型状态
    ctypedef enum HighsModelStatus:
        HighsModelStatusNOTSET "HighsModelStatus::kNotset" = 0
        HighsModelStatusLOAD_ERROR "HighsModelStatus::kLoadError"
        HighsModelStatusMODEL_ERROR "HighsModelStatus::kModelError"
        HighsModelStatusPRESOLVE_ERROR "HighsModelStatus::kPresolveError"
        HighsModelStatusSOLVE_ERROR "HighsModelStatus::kSolveError"
        HighsModelStatusPOSTSOLVE_ERROR "HighsModelStatus::kPostsolveError"
        HighsModelStatusMODEL_EMPTY "HighsModelStatus::kModelEmpty"
        HighsModelStatusOPTIMAL "HighsModelStatus::kOptimal"
        HighsModelStatusINFEASIBLE "HighsModelStatus::kInfeasible"
        HighsModelStatus_UNBOUNDED_OR_INFEASIBLE "HighsModelStatus::kUnboundedOrInfeasible"
        HighsModelStatusUNBOUNDED "HighsModelStatus::kUnbounded"
        HighsModelStatusREACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND "HighsModelStatus::kObjectiveBound"
        HighsModelStatusREACHED_OBJECTIVE_TARGET "HighsModelStatus::kObjectiveTarget"
        HighsModelStatusREACHED_TIME_LIMIT "HighsModelStatus::kTimeLimit"
        HighsModelStatusREACHED_ITERATION_LIMIT "HighsModelStatus::kIterationLimit"
        HighsModelStatusUNKNOWN "HighsModelStatus::kUnknown"
        HighsModelStatusHIGHS_MODEL_STATUS_MIN "HighsModelStatus::kMin" = HighsModelStatusNOTSET
        HighsModelStatusHIGHS_MODEL_STATUS_MAX "HighsModelStatus::kMax" = HighsModelStatusUNKNOWN

    # 定义枚举类型`HighsBasisStatus`，表示Highs基态状态
    cdef enum HighsBasisStatus:
        HighsBasisStatusLOWER "HighsBasisStatus::kLower" = 0, # (slack) variable is at its lower bound [including fixed variables]
        HighsBasisStatusBASIC "HighsBasisStatus::kBasic" # (slack) variable is basic
        HighsBasisStatusUPPER "HighsBasisStatus::kUpper" # (slack) variable is at its upper bound
        HighsBasisStatusZERO "HighsBasisStatus::kZero" # free variable is non-basic and set to zero
        HighsBasisStatusNONBASIC "HighsBasisStatus::kNonbasic" # nonbasic with no specific bound information - useful for users and postsolve
    # 定义 SolverOption 枚举，包含三个选项：SOLVER_OPTION_SIMPLEX、SOLVER_OPTION_CHOOSE、SOLVER_OPTION_IPM
    cdef enum SolverOption:
        SOLVER_OPTION_SIMPLEX "SolverOption::SOLVER_OPTION_SIMPLEX" = -1
        SOLVER_OPTION_CHOOSE "SolverOption::SOLVER_OPTION_CHOOSE"
        SOLVER_OPTION_IPM "SolverOption::SOLVER_OPTION_IPM"

    # 定义 PrimalDualStatus 枚举，包含多个状态：STATUS_NOT_SET、STATUS_MIN、STATUS_NO_SOLUTION、STATUS_UNKNOWN、
    # STATUS_INFEASIBLE_POINT、STATUS_FEASIBLE_POINT、STATUS_MAX
    cdef enum PrimalDualStatus:
        PrimalDualStatusSTATUS_NOT_SET "PrimalDualStatus::STATUS_NOT_SET" = -1
        PrimalDualStatusSTATUS_MIN "PrimalDualStatus::STATUS_MIN" = PrimalDualStatusSTATUS_NOT_SET
        PrimalDualStatusSTATUS_NO_SOLUTION "PrimalDualStatus::STATUS_NO_SOLUTION"
        PrimalDualStatusSTATUS_UNKNOWN "PrimalDualStatus::STATUS_UNKNOWN"
        PrimalDualStatusSTATUS_INFEASIBLE_POINT "PrimalDualStatus::STATUS_INFEASIBLE_POINT"
        PrimalDualStatusSTATUS_FEASIBLE_POINT "PrimalDualStatus::STATUS_FEASIBLE_POINT"
        PrimalDualStatusSTATUS_MAX "PrimalDualStatus::STATUS_MAX" = PrimalDualStatusSTATUS_FEASIBLE_POINT

    # 定义 HighsOptionType 枚举，包含四种类型：kBool、kInt、kDouble、kString
    cdef enum HighsOptionType:
        HighsOptionTypeBOOL "HighsOptionType::kBool" = 0
        HighsOptionTypeINT "HighsOptionType::kInt"
        HighsOptionTypeDOUBLE "HighsOptionType::kDouble"
        HighsOptionTypeSTRING "HighsOptionType::kString"

    # workaround for lack of enum class support in Cython < 3.x
    # 定义 ObjSense 类，用于表示优化目标的方向，包含 MINIMIZE 和 MAXIMIZE 两个实例
    cdef cppclass ObjSense:
        pass

    # 定义 ObjSenseMINIMIZE 和 ObjSenseMAXIMIZE 两个对象，分别表示最小化和最大化的优化目标方向
    cdef ObjSense ObjSenseMINIMIZE "ObjSense::kMinimize"
    cdef ObjSense ObjSenseMAXIMIZE "ObjSense::kMaximize"

    # 定义 MatrixFormat 类，用于表示矩阵的存储格式，包含三种格式：kColwise、kRowwise、kRowwisePartitioned
    cdef cppclass MatrixFormat:
        pass

    # 定义三个对象 MatrixFormatkColwise、MatrixFormatkRowwise、MatrixFormatkRowwisePartitioned，
    # 分别表示列主存储、行主存储和分区行主存储三种矩阵存储格式
    cdef MatrixFormat MatrixFormatkColwise "MatrixFormat::kColwise"
    cdef MatrixFormat MatrixFormatkRowwise "MatrixFormat::kRowwise"
    cdef MatrixFormat MatrixFormatkRowwisePartitioned "MatrixFormat::kRowwisePartitioned"

    # cdef enum class HighsVarType(int):
    #     kContinuous "HighsVarType::kContinuous"
    #     kInteger "HighsVarType::kInteger"
    #     kSemiContinuous "HighsVarType::kSemiContinuous"
    #     kSemiInteger "HighsVarType::kSemiInteger"
    #     kImplicitInteger "HighsVarType::kImplicitInteger"

    # 定义 HighsVarType 类，用于表示变量的类型，包含五种类型：kContinuous、kInteger、kSemiContinuous、
    # kSemiInteger、kImplicitInteger
    cdef cppclass HighsVarType:
        pass

    # 定义 kContinuous、kInteger、kSemiContinuous、kSemiInteger、kImplicitInteger 五个对象，
    # 分别表示连续变量、整数变量、半连续变量、半整数变量、隐含整数变量五种变量类型
    cdef HighsVarType kContinuous "HighsVarType::kContinuous"
    cdef HighsVarType kInteger "HighsVarType::kInteger"
    cdef HighsVarType kSemiContinuous "HighsVarType::kSemiContinuous"
    cdef HighsVarType kSemiInteger "HighsVarType::kSemiInteger"
    cdef HighsVarType kImplicitInteger "HighsVarType::kImplicitInteger"
```