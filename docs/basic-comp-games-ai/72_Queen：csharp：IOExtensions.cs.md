# `d:/src/tocomm/basic-computer-games\72_Queen\csharp\IOExtensions.cs`

```
# 定义一个命名空间 Queen
namespace Queen;

# 定义一个内部静态类 IOExtensions
internal static class IOExtensions
{
    # 定义一个扩展方法 ReadYesNo，接收一个 IReadWrite 类型的参数 io 和一个字符串 prompt
    internal static bool ReadYesNo(this IReadWrite io, string prompt)
    {
        # 创建一个无限循环
        while (true)
        {
            # 从输入流中读取一个字符串，并转换为小写
            var answer = io.ReadString(prompt).ToLower();
            # 如果答案是 "yes"，则返回 true
            if (answer == "yes") { return true; }
            # 如果答案是 "no"，则返回 false
            if (answer == "no") { return false; }

            # 向输出流写入 "Yes or No?" 的提示信息
            io.Write(Streams.YesOrNo);
        }
    }

    # 定义一个扩展方法 ReadPosition，接收一个 IReadWrite 类型的参数 io，一个字符串 prompt，一个 Predicate<Position> 类型的参数 isValid
    internal static Position ReadPosition(
        this IReadWrite io,
        string prompt,
        Predicate<Position> isValid,
        // 定义一个名为Stream的参数，表示输入输出流
        // 定义一个名为error的参数，表示错误信息
        // 定义一个名为repeatPrompt的参数，表示是否重复提示
    {
        // 创建一个无限循环，直到满足条件才会退出循环
        while (true)
        {
            // 从输入输出流中读取一个数字作为响应
            var response = io.ReadNumber(prompt);
            // 将响应转换为整数类型
            var number = (int)response;
            // 创建一个新的Position对象，传入整数类型的响应作为参数
            var position = new Position(number);
            // 如果响应是整数并且位置是零，或者位置是有效的，则返回该位置
            if (number == response && (position.IsZero || isValid(position)))
            {
                return position;
            }

            // 将错误信息写入输入输出流
            io.Write(error);
            // 如果不需要重复提示，则清空提示信息
            if (!repeatPrompt) { prompt = ""; }
        }
    }
}
```