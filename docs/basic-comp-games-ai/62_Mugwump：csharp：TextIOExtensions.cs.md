# `d:/src/tocomm/basic-computer-games\62_Mugwump\csharp\TextIOExtensions.cs`

```
namespace Mugwump; // 命名空间声明

// 提供模拟 BASIC 解释器键盘输入例程的输入方法
internal static class TextIOExtensions // 创建名为 TextIOExtensions 的静态类
{
    internal static Position ReadGuess(this TextIO io, string prompt) // 创建名为 ReadGuess 的静态方法，接受 TextIO 对象和字符串 prompt 作为参数，返回 Position 对象
    {
        io.WriteLine(); // 调用 TextIO 对象的 WriteLine 方法
        io.WriteLine(); // 再次调用 TextIO 对象的 WriteLine 方法
        var (x, y) = io.Read2Numbers(prompt); // 调用 TextIO 对象的 Read2Numbers 方法，将返回的元组赋值给变量 (x, y)
        return new Position(x, y); // 返回一个新的 Position 对象，传入 x 和 y 作为参数
    }
}
```