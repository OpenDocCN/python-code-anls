# `d:/src/tocomm/basic-computer-games\00_Common\dotnet\Games.Common\IO\ConsoleIO.cs`

```
using System;  // 导入 System 命名空间，包含了 Console 类等

namespace Games.Common.IO;  // 声明 Games.Common.IO 命名空间

/// <summary>
/// An implementation of <see cref="IReadWrite" /> with input begin read for STDIN and output being written to
/// STDOUT.
/// </summary>
public sealed class ConsoleIO : TextIO  // 声明 ConsoleIO 类，继承自 TextIO 类
{
    public ConsoleIO()  // 声明 ConsoleIO 类的构造函数
        : base(Console.In, Console.Out)  // 调用基类 TextIO 的构造函数，传入 Console.In 和 Console.Out 作为参数
    {
    }

    public override char ReadCharacter() => Console.ReadKey(intercept: true).KeyChar;  // 重写基类的 ReadCharacter 方法，使用 Console 类的 ReadKey 方法获取按键的字符
}
```