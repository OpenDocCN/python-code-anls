# `basic-computer-games\00_Common\dotnet\Games.Common\IO\ConsoleIO.cs`

```

// 使用 System 命名空间
using System;

// 声明 Games.Common.IO 命名空间
namespace Games.Common.IO;

/// <summary>
/// 通过 STDIN 进行输入读取，通过 STDOUT 进行输出写入的 <see cref="IReadWrite" /> 实现。
/// </summary>
// 声明 ConsoleIO 类，继承自 TextIO 类
public sealed class ConsoleIO : TextIO
{
    // 构造函数，使用 Console.In 作为输入流，Console.Out 作为输出流
    public ConsoleIO()
        : base(Console.In, Console.Out)
    {
    }

    // 重写 ReadCharacter 方法，从控制台读取字符
    public override char ReadCharacter() => Console.ReadKey(intercept: true).KeyChar;
}

```