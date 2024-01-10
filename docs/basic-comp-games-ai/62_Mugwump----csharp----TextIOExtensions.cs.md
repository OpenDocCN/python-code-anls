# `basic-computer-games\62_Mugwump\csharp\TextIOExtensions.cs`

```
// 命名空间声明，定义了代码所在的命名空间
namespace Mugwump;

// 定义了一个静态类，提供了模拟BASIC解释器键盘输入例程的输入方法
internal static class TextIOExtensions
{
    // 定义了一个扩展方法，用于读取用户猜测的位置
    internal static Position ReadGuess(this TextIO io, string prompt)
    {
        // 输出空行
        io.WriteLine();
        io.WriteLine();
        // 调用Read2Numbers方法读取两个数字，并赋值给变量(x, y)
        var (x, y) = io.Read2Numbers(prompt);
        // 返回一个新的Position对象，传入x和y作为参数
        return new Position(x, y);
    }
}
```