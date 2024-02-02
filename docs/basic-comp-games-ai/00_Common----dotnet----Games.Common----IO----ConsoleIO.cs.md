# `basic-computer-games\00_Common\dotnet\Games.Common\IO\ConsoleIO.cs`

```py
// 使用 System 命名空间
using System;

// 声明 Games.Common.IO 命名空间
namespace Games.Common.IO;

/// <summary>
/// 一个实现了 IReadWrite 接口的类，用于从标准输入读取输入并将输出写入标准输出
/// </summary>
public sealed class ConsoleIO : TextIO
{
    // 构造函数，使用标准输入和标准输出初始化基类
    public ConsoleIO()
        : base(Console.In, Console.Out)
    {
    }

    // 重写基类的 ReadCharacter 方法，从控制台读取字符
    public override char ReadCharacter() => Console.ReadKey(intercept: true).KeyChar;
}
```