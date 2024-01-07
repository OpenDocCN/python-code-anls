# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\Methods.cs`

```

// 引入系统诊断命名空间
using System.Diagnostics;

// 声明 DotnetUtils 命名空间
namespace DotnetUtils
{
    // 声明 ProcessResult 类，用于存储进程执行结果
    public sealed record ProcessResult(int ExitCode, string StdOut, string StdErr) {
        // 重写 ToString 方法，返回标准输出和标准错误信息
        public override string? ToString() =>
            // 返回标准输出
            StdOut +
            // 如果标准输出不为空且退出码大于0，则添加换行符
            (StdOut is not (null or "") && ExitCode > 0 ? "\n" : "") +
            // 如果退出码不为0，则返回退出码和标准错误信息，否则返回空字符串
            (ExitCode != 0 ?
                $"{ExitCode}\n{StdErr}" :
                "");
    }
}

```