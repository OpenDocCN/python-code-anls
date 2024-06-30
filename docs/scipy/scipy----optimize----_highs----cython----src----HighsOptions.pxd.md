# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\HighsOptions.pxd`

```
# cython: language_level=3

# 导入需要的 C 库的标准文件，用于文件 I/O 操作
from libc.stdio cimport FILE

# 导入 C++ 标准库的数据类型和容器
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

# 从当前模块的 HConst 文件中导入 HighsOptionType 枚举类型
from .HConst cimport HighsOptionType

# 定义 C 语言外部接口声明，告知编译器将在没有全局锁定的情况下处理
cdef extern from "HighsOptions.h" nogil:

    # 定义 C++ 类 OptionRecord，表示单个选项记录
    cdef cppclass OptionRecord:
        HighsOptionType type         # 选项类型
        string name                  # 选项名称
        string description           # 选项描述
        bool advanced                # 是否为高级选项

    # 派生类 OptionRecordBool，表示布尔类型选项记录
    cdef cppclass OptionRecordBool(OptionRecord):
        bool* value                  # 指向布尔值的指针
        bool default_value           # 默认布尔值

    # 派生类 OptionRecordInt，表示整数类型选项记录
    cdef cppclass OptionRecordInt(OptionRecord):
        int* value                   # 指向整数值的指针
        int lower_bound              # 下界
        int default_value            # 默认整数值
        int upper_bound              # 上界

    # 派生类 OptionRecordDouble，表示双精度浮点数类型选项记录
    cdef cppclass OptionRecordDouble(OptionRecord):
        double* value               # 指向双精度浮点数值的指针
        double lower_bound          # 下界
        double default_value        # 默认双精度浮点数值
        double upper_bound          # 上界

    # 派生类 OptionRecordString，表示字符串类型选项记录
    cdef cppclass OptionRecordString(OptionRecord):
        string* value               # 指向字符串值的指针
        string default_value        # 默认字符串值
    # 定义一个 C++ 类 HighsOptions，用于管理 Highs Solver 的各种选项和参数

    cdef cppclass HighsOptions:
        # 以下是从 HighsOptionsStruct 中继承的选项：

        # 从命令行读取的选项
        string model_file         # 模型文件名
        string presolve           # 预处理选项
        string solver             # 求解器选项
        string parallel           # 并行计算选项
        double time_limit         # 时间限制
        string options_file       # 选项文件

        # 从文件中读取的选项
        double infinite_cost                      # 无限成本
        double infinite_bound                     # 无限边界
        double small_matrix_value                 # 小矩阵值
        double large_matrix_value                 # 大矩阵值
        double primal_feasibility_tolerance       # 原始可行性容忍度
        double dual_feasibility_tolerance         # 对偶可行性容忍度
        double ipm_optimality_tolerance           # IPM 优化容忍度
        double dual_objective_value_upper_bound   # 对偶目标值上界
        int highs_debug_level                     # Highs 调试级别
        int simplex_strategy                      # 单纯形策略
        int simplex_scale_strategy                # 单纯形缩放策略
        int simplex_crash_strategy                # 单纯形崩溃策略
        int simplex_dual_edge_weight_strategy     # 单纯形对偶边权重策略
        int simplex_primal_edge_weight_strategy   # 单纯形原始边权重策略
        int simplex_iteration_limit               # 单纯形迭代限制
        int simplex_update_limit                  # 单纯形更新限制
        int ipm_iteration_limit                   # IPM 迭代限制
        int highs_min_threads                     # Highs 最小线程数
        int highs_max_threads                     # Highs 最大线程数
        int message_level                         # 消息级别
        string solution_file                     # 解决方案文件名
        bool write_solution_to_file               # 是否将解决方案写入文件
        bool write_solution_pretty                # 是否以可读形式写入解决方案

        # 高级选项
        bool run_crossover                        # 是否运行交叉
        bool mps_parser_type_free                 # MPS 解析器类型自由
        int keep_n_rows                           # 保留的行数
        int allowed_simplex_matrix_scale_factor   # 允许的单纯形矩阵缩放因子
        int allowed_simplex_cost_scale_factor     # 允许的单纯形成本缩放因子
        int simplex_dualise_strategy              # 单纯形对偶化策略
        int simplex_permute_strategy              # 单纯形置换策略
        int dual_simplex_cleanup_strategy         # 对偶单纯形清理策略
        int simplex_price_strategy                # 单纯形价格策略
        int dual_chuzc_sort_strategy             # 对偶 Chuzc 排序策略
        bool simplex_initial_condition_check      # 单纯形初始条件检查
        double simplex_initial_condition_tolerance# 单纯形初始条件容忍度
        double dual_steepest_edge_weight_log_error_threshhold # 对偶最陡边权重对数误差阈值
        double dual_simplex_cost_perturbation_multiplier  # 对偶单纯形成本扰动乘数
        double start_crossover_tolerance          # 开始交叉容忍度
        bool less_infeasible_DSE_check           # 是否检查较少的不可行 DSE
        bool less_infeasible_DSE_choose_row      # 是否选择较少不可行 DSE 的行
        bool use_original_HFactor_logic          # 是否使用原始 HFactor 逻辑

        # MIP 求解器选项
        int mip_max_nodes                         # MIP 最大节点数
        int mip_report_level                      # MIP 报告级别

        # MIP 求解器开关
        bool mip                                  # 是否启用 MIP 求解

        # HighsPrintMessage 和 HighsLogMessage 的选项
        FILE* logfile                            # 日志文件
        FILE* output                             # 输出文件
        int message_level                        # 消息级别（再次出现，可能是重复的）

        string solution_file                     # 解决方案文件名（再次出现，可能是重复的）
        bool write_solution_to_file              # 是否将解决方案写入文件（再次出现，可能是重复的）
        bool write_solution_pretty               # 是否以可读形式写入解决方案（再次出现，可能是重复的）

        vector[OptionRecord*] records            # 选项记录的向量
```