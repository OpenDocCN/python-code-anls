# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\Methods.cs`

```py
// 命名空间定义了一个名为 DotnetUtils 的命名空间
namespace DotnetUtils;

// 定义了一个名为 Methods 的静态类
public static class Methods {
    // 定义了一个名为 RunProcess 的公共静态方法，用于运行外部进程并返回结果
    public static ProcessResult RunProcess(string filename, string arguments) {
        // 创建一个新的进程对象
        var process = new Process() {
            StartInfo = {
                FileName = filename,  // 设置要执行的文件名
                Arguments = arguments,  // 设置要传递给进程的参数
                UseShellExecute = false,  // 禁用使用操作系统外壳启动进程
                CreateNoWindow = true,  // 在不显示窗口的情况下执行进程
                RedirectStandardOutput = true,  // 重定向标准输出流
                RedirectStandardError = true,  // 重定向标准错误流
            },
            EnableRaisingEvents = true  // 启用进程事件
        };
        return RunProcess(process);  // 调用另一个重载的 RunProcess 方法
    }

    // 定义了一个名为 RunProcess 的公共静态方法，用于运行外部进程并返回结果
    public static ProcessResult RunProcess(Process process, string input = "") {
        var (output, error) = ("", "");  // 初始化输出和错误字符串
        var (redirectOut, redirectErr) = (  // 初始化重定向标志
            process.StartInfo.RedirectStandardOutput,
            process.StartInfo.RedirectStandardError
        );
        if (redirectOut) {  // 如果重定向标准输出
            process.OutputDataReceived += (s, ea) => output += ea.Data + "\n";  // 处理标准输出数据
        }
        if (redirectErr) {  // 如果重定向标准错误
            process.ErrorDataReceived += (s, ea) => error += ea.Data + "\n";  // 处理标准错误数据
        }

        if (!process.Start()) {  // 如果无法启动进程
            throw new InvalidOperationException();  // 抛出异常
        };

        if (redirectOut) { process.BeginOutputReadLine(); }  // 开始异步读取标准输出
        if (redirectErr) { process.BeginErrorReadLine(); }  // 开始异步读取标准错误
        if (!string.IsNullOrEmpty(input)) {  // 如果输入不为空
            process.StandardInput.WriteLine(input);  // 写入输入
            process.StandardInput.Close();  // 关闭标准输入
        }
        process.WaitForExit();  // 等待进程退出
        return new ProcessResult(process.ExitCode, output, error);  // 返回进程结果
    }
}
    // 异步运行进程，并返回一个任务，该任务会在进程退出时得到结果
    public static Task<ProcessResult> RunProcessAsync(Process process, string input = "") {
        // 创建一个任务完成源
        var tcs = new TaskCompletionSource<ProcessResult>();
        // 初始化输出和错误信息
        var (output, error) = ("", "");
        // 获取是否重定向标准输出和标准错误流
        var (redirectOut, redirectErr) = (
            process.StartInfo.RedirectStandardOutput,
            process.StartInfo.RedirectStandardError
        );

        // 当进程退出时，设置任务完成并返回进程结果
        process.Exited += (s, e) => tcs.SetResult(new ProcessResult(process.ExitCode, output, error));

        // 如果重定向了标准输出流，则将输出数据添加到输出信息中
        if (redirectOut) {
            process.OutputDataReceived += (s, ea) => output += ea.Data + "\n";
        }
        // 如果重定向了标准错误流，则将错误数据添加到错误信息中
        if (redirectErr) {
            process.ErrorDataReceived += (s, ea) => error += ea.Data + "\n";
        }

        // 如果进程启动失败，则抛出异常
        if (!process.Start()) {
            // 如果进程启动失败，Exited事件会怎样？
            throw new InvalidOperationException();
        }

        // 如果重定向了标准输出流，则开始异步读取输出数据
        if (redirectOut) { process.BeginOutputReadLine(); }
        // 如果重定向了标准错误流，则开始异步读取错误数据
        if (redirectErr) { process.BeginErrorReadLine(); }
        // 如果输入不为空，则将输入写入标准输入流并关闭
        if (!string.IsNullOrEmpty(input)) {
            process.StandardInput.WriteLine(input);
            process.StandardInput.Close();
        }

        // 返回任务
        return tcs.Task;
    }
# 定义一个不可变的记录类型 ProcessResult，包含进程的退出码、标准输出和标准错误
public sealed record ProcessResult(int ExitCode, string StdOut, string StdErr) {
    # 重写 ToString 方法，返回标准输出和（如果有）退出码和标准错误
    public override string? ToString() =>
        # 返回标准输出
        StdOut +
        # 如果标准输出不为空且退出码大于0，则添加换行符
        (StdOut is not (null or "") && ExitCode > 0 ? "\n" : "") +
        # 如果退出码不等于0，则返回退出码和标准错误；否则返回空字符串
        (ExitCode != 0 ?
            $"{ExitCode}\n{StdErr}" :
            "");
}
```