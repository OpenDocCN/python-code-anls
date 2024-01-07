# `basic-computer-games\62_Mugwump\csharp\TextIOExtensions.cs`

```

// 命名空间声明，定义了代码所在的命名空间
namespace Mugwump;

// 定义了一个静态类，提供了模拟BASIC解释器键盘输入例程的输入方法
internal static class TextIOExtensions
{
    // 定义了一个静态方法，用于读取用户猜测的位置，并返回一个Position对象
    internal static Position ReadGuess(this TextIO io, string prompt)
    {
        // 输出空行
        io.WriteLine();
        io.WriteLine();
        // 调用io对象的Read2Numbers方法，获取用户输入的两个数字，并赋值给变量(x, y)
        var (x, y) = io.Read2Numbers(prompt);
        // 返回一个新的Position对象，以用户输入的两个数字作为参数
        return new Position(x, y);
    }
}

```