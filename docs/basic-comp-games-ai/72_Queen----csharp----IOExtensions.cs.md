# `basic-computer-games\72_Queen\csharp\IOExtensions.cs`

```

# 命名空间 Queen 下的内部静态类 IOExtensions
namespace Queen;

internal static class IOExtensions
{
    # 读取用户输入的是或否，返回布尔值
    internal static bool ReadYesNo(this IReadWrite io, string prompt)
    {
        # 循环直到用户输入 yes 或 no
        while (true)
        {
            # 读取用户输入并转换为小写
            var answer = io.ReadString(prompt).ToLower();
            # 如果用户输入为 "yes"，返回 true
            if (answer == "yes") { return true; }
            # 如果用户输入为 "no"，返回 false
            if (answer == "no") { return false; }

            # 提示用户重新输入
            io.Write(Streams.YesOrNo);
        }
    }

    # 读取用户输入的位置信息，并验证是否有效，返回位置对象
    internal static Position ReadPosition(
        this IReadWrite io,
        string prompt,
        Predicate<Position> isValid,
        Stream error,
        bool repeatPrompt = false)
    {
        # 循环直到用户输入有效的位置信息
        while (true)
        {
            # 读取用户输入的数字
            var response = io.ReadNumber(prompt);
            var number = (int)response;
            var position = new Position(number);
            # 如果输入为整数且位置有效，则返回位置对象
            if (number == response && (position.IsZero || isValid(position)))
            {
                return position;
            }

            # 输出错误信息
            io.Write(error);
            # 如果不需要重复提示，则清空提示信息
            if (!repeatPrompt) { prompt = ""; }
        }
    }
}

```