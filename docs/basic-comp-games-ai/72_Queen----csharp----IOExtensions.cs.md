# `basic-computer-games\72_Queen\csharp\IOExtensions.cs`

```
// 命名空间 Queen 下的内部静态类 IOExtensions
namespace Queen
{
    internal static class IOExtensions
    {
        // 读取用户输入的 yes 或 no，返回布尔值
        internal static bool ReadYesNo(this IReadWrite io, string prompt)
        {
            // 循环直到用户输入有效的 yes 或 no
            while (true)
            {
                // 读取用户输入并转换为小写
                var answer = io.ReadString(prompt).ToLower();
                // 如果用户输入为 "yes"，返回 true
                if (answer == "yes") { return true; }
                // 如果用户输入为 "no"，返回 false
                if (answer == "no") { return false; }

                // 如果用户输入无效，提示用户重新输入
                io.Write(Streams.YesOrNo);
            }
        }

        // 读取用户输入的位置信息，返回 Position 对象
        internal static Position ReadPosition(
            this IReadWrite io,
            string prompt,
            Predicate<Position> isValid,
            Stream error,
            bool repeatPrompt = false)
        {
            // 循环直到用户输入有效的位置信息
            while (true)
            {
                // 读取用户输入的数字
                var response = io.ReadNumber(prompt);
                var number = (int)response;
                var position = new Position(number);
                // 如果输入为整数且满足条件，返回位置信息
                if (number == response && (position.IsZero || isValid(position)))
                {
                    return position;
                }

                // 如果输入无效，提示用户错误信息
                io.Write(error);
                // 如果不需要重复提示，清空提示信息
                if (!repeatPrompt) { prompt = ""; }
            }
        }
    }
}
```