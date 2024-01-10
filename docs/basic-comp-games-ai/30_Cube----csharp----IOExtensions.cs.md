# `basic-computer-games\30_Cube\csharp\IOExtensions.cs`

```
# 创建名为 Cube 的命名空间，并定义一个内部静态类 IOExtensions
namespace Cube;

internal static class IOExtensions
{
    # 定义一个扩展方法 ReadWager，接收一个 IReadWrite 类型的参数 io 和一个浮点数类型的参数 balance
    internal static float? ReadWager(this IReadWrite io, float balance)
    {
        # 调用 io 对象的 Write 方法，传入 Streams.Wager 作为参数
        io.Write(Streams.Wager);
        # 如果调用 io 对象的 ReadNumber 方法返回值为 0，则返回空值
        if (io.ReadNumber("") == 0) { return null; }

        # 定义一个字符串变量 prompt，赋值为 Prompts.HowMuch
        var prompt = Prompts.HowMuch;

        # 进入无限循环
        while(true)
        {
            # 调用 io 对象的 ReadNumber 方法，传入 prompt 作为参数，将返回值赋给变量 wager
            var wager = io.ReadNumber(prompt);
            # 如果 wager 小于等于 balance，则返回 wager
            if (wager <= balance) { return wager; }

            # 将 prompt 的值更新为 Prompts.BetAgain
            prompt = Prompts.BetAgain;
        }
    }
}
```