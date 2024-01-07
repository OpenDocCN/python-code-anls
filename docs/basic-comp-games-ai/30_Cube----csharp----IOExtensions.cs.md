# `basic-computer-games\30_Cube\csharp\IOExtensions.cs`

```

# 命名空间 Cube 下的内部静态类 IOExtensions
namespace Cube;

internal static class IOExtensions
{
    # 读取赌注的扩展方法，接受一个 IReadWrite 接口和余额作为参数
    internal static float? ReadWager(this IReadWrite io, float balance)
    {
        # 写入赌注流
        io.Write(Streams.Wager);
        # 如果读取的数字为 0，则返回空值
        if (io.ReadNumber("") == 0) { return null; }

        # 设置赌注提示
        var prompt = Prompts.HowMuch;

        # 循环直到条件不满足
        while(true)
        {
            # 读取赌注，如果小于等于余额则返回赌注
            var wager = io.ReadNumber(prompt);
            if (wager <= balance) { return wager; }

            # 更新提示为重新下注
            prompt = Prompts.BetAgain;
        }
    }
}

```