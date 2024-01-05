# `00_Utilities\DotnetUtils\DotnetUtils\Methods.cs`

```
# 导入 System.Diagnostics 模块
using System.Diagnostics;

# 定义 DotnetUtils 命名空间
namespace DotnetUtils;

# 定义 Methods 类
public static class Methods {
    # 定义 RunProcess 方法，用于运行外部进程并返回结果
    public static ProcessResult RunProcess(string filename, string arguments) {
        # 创建一个新的进程对象
        var process = new Process() {
            # 设置进程的启动信息
            StartInfo = {
                FileName = filename,  # 设置要运行的程序文件名
                Arguments = arguments,  # 设置要传递给程序的参数
                UseShellExecute = false,  # 禁用使用操作系统外壳启动进程
                CreateNoWindow = true,  # 不创建新窗口
                RedirectStandardOutput = true,  # 重定向标准输出流
                RedirectStandardError = true,  # 重定向标准错误流
            },
            EnableRaisingEvents = true  # 启用进程事件
        };
        # 调用 RunProcess 方法并返回结果
        return RunProcess(process);
    }
}
        // 如果输入流不为空，将输入流写入进程的标准输入流
        process.StandardInput.WriteLine(input);
        }
        // 等待进程执行完毕
        process.WaitForExit();
        // 返回进程执行结果
        return new ProcessResult {
            Output = output,
            Error = error,
            ExitCode = process.ExitCode
        };
    }
            process.StandardInput.WriteLine(input);  // 向进程的标准输入流写入输入内容
            process.StandardInput.Close();  // 关闭进程的标准输入流
        }
        process.WaitForExit();  // 等待进程退出
        return new ProcessResult(process.ExitCode, output, error);  // 返回进程的退出码、输出和错误信息
    }

    public static Task<ProcessResult> RunProcessAsync(Process process, string input = "") {  // 异步运行进程的方法
        var tcs = new TaskCompletionSource<ProcessResult>();  // 创建一个任务完成源
        var (output, error) = ("", "");  // 初始化输出和错误信息为空字符串
        var (redirectOut, redirectErr) = (  // 获取进程是否重定向标准输出和标准错误流
            process.StartInfo.RedirectStandardOutput,
            process.StartInfo.RedirectStandardError
        );

        process.Exited += (s, e) => tcs.SetResult(new ProcessResult(process.ExitCode, output, error));  // 当进程退出时，设置任务完成源的结果为进程的退出码、输出和错误信息

        if (redirectOut) {  // 如果进程重定向了标准输出流
            process.OutputDataReceived += (s, ea) => output += ea.Data + "\n";  // 当有输出数据时，将数据添加到输出信息中
        }
        if (redirectErr) {
            // 如果需要重定向错误输出，将错误数据接收事件与错误数据字符串连接起来
            process.ErrorDataReceived += (s, ea) => error += ea.Data + "\n";
        }

        if (!process.Start()) {
            // 如果进程启动失败，抛出无效操作异常
            throw new InvalidOperationException();
        }

        if (redirectOut) { 
            // 如果需要重定向标准输出，开始异步读取标准输出流
            process.BeginOutputReadLine(); 
        }
        if (redirectErr) { 
            // 如果需要重定向错误输出，开始异步读取错误输出流
            process.BeginErrorReadLine(); 
        }
        if (!string.IsNullOrEmpty(input)) {
            // 如果输入不为空，向标准输入流写入输入，并关闭标准输入流
            process.StandardInput.WriteLine(input);
            process.StandardInput.Close();
        }

        // 返回任务的异步操作结果
        return tcs.Task;
    }
}
# 定义一个名为ProcessResult的不可变记录类型，包含三个属性：ExitCode（整数类型）、StdOut（字符串类型）、StdErr（字符串类型）
public sealed record ProcessResult(int ExitCode, string StdOut, string StdErr) {
    # 重写ToString方法，返回标准输出（StdOut）和如果标准输出不为空且退出码大于0，则添加换行符，然后根据退出码是否为0来返回不同的字符串
    public override string? ToString() =>
        StdOut +
        (StdOut is not (null or "") && ExitCode > 0 ? "\n" : "") +
        (ExitCode != 0 ?
            $"{ExitCode}\n{StdErr}" :
            "");
}
```