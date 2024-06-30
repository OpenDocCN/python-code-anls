# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\Highs.pxd`

```
# cython: language_level=3

# 从 C 库中导入 FILE 类型
from libc.stdio cimport FILE

# 从 C++ 库中导入 bool 类型和 string 类型
from libcpp cimport bool
from libcpp.string cimport string

# 从自定义模块中导入以下类和枚举
from .HighsStatus cimport HighsStatus
from .HighsOptions cimport HighsOptions
from .HighsInfo cimport HighsInfo
from .HighsLp cimport (
    HighsLp,
    HighsSolution,
    HighsBasis,
    ObjSense,
)
from .HConst cimport HighsModelStatus

# 从外部头文件 "Highs.h" 中声明 Highs 类
cdef extern from "Highs.h":
    # Highs 类声明
    cdef cppclass Highs:
        # 设置 Highs 的选项
        HighsStatus passHighsOptions(const HighsOptions& options)
        # 传入线性规划模型到 Highs
        HighsStatus passModel(const HighsLp& lp)
        # 运行 Highs 求解器
        HighsStatus run()
        # 设置 Highs 的日志文件
        HighsStatus setHighsLogfile(FILE* logfile)
        # 设置 Highs 的输出文件
        HighsStatus setHighsOutput(FILE* output)
        # 将 Highs 的选项写入文件
        HighsStatus writeHighsOptions(const string filename, const bool report_only_non_default_values = true)

        # 获取模型的状态
        const HighsModelStatus & getModelStatus() const
        # 获取 Highs 的信息
        const HighsInfo& getHighsInfo "getInfo" () const
        # 将模型状态转换为字符串
        string modelStatusToString(const HighsModelStatus model_status) const
        # 获取指定信息的值
        HighsStatus getHighsInfoValue(const string& info, double& value) const
        # 获取 Highs 的选项
        const HighsOptions& getHighsOptions() const
        # 获取线性规划模型对象
        const HighsLp& getLp() const
        # 将解写入文件
        HighsStatus writeSolution(const string filename, const bool pretty) const
        # 设置基
        HighsStatus setBasis()
        # 获取解
        const HighsSolution& getSolution() const
        # 获取基
        const HighsBasis& getBasis() const
        # 改变目标函数的方向
        bool changeObjectiveSense(const ObjSense sense)
        # 设置 Highs 的选项为布尔值
        HighsStatus setHighsOptionValueBool "setOptionValue" (const string & option, const bool value)
        # 设置 Highs 的选项为整数值
        HighsStatus setHighsOptionValueInt "setOptionValue" (const string & option, const int value)
        # 设置 Highs 的选项为字符串值
        HighsStatus setHighsOptionValueStr "setOptionValue" (const string & option, const string & value)
        # 设置 Highs 的选项为浮点数值
        HighsStatus setHighsOptionValueDbl "setOptionValue" (const string & option, const double value)
        # 将原始对偶状态转换为字符串
        string primalDualStatusToString(const int primal_dual_status)
        # 重置全局调度器
        void resetGlobalScheduler(bool blocking)
```