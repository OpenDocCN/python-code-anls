# `basic-computer-games\84_Super_Star_Trek\csharp\IReadWriteExtensions.cs`

```
// 引入所需的命名空间
using System;
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Space;
using static System.StringComparison;

// 定义 IReadWriteExtensions 类
namespace SuperStarTrek;
internal static class IReadWriteExtensions
{
    // 等待用户按下除了 Enter 键以外的任意键
    internal static void WaitForAnyKeyButEnter(this IReadWrite io, string prompt)
    {
        io.Write($"Hit any key but Enter {prompt} ");
        while (io.ReadCharacter() == '\r');
    }

    // 获取坐标值
    internal static (float X, float Y) GetCoordinates(this IReadWrite io, string prompt) =>
        io.Read2Numbers($"{prompt} (X,Y)");

    // 尝试读取指定范围内的数字
    internal static bool TryReadNumberInRange(
        this IReadWrite io,
        string prompt,
        float minValue,
        float maxValue,
        out float value)
    {
        value = io.ReadNumber($"{prompt} ({minValue}-{maxValue})");

        return value >= minValue && value <= maxValue;
    }

    // 读取预期的字符串
    internal static bool ReadExpectedString(this IReadWrite io, string prompt, string trueValue) =>
        io.ReadString(prompt).Equals(trueValue, InvariantCultureIgnoreCase);

    // 读取命令
    internal static Command ReadCommand(this IReadWrite io)
    {
        while(true)
        {
            var response = io.ReadString("Command");

            if (response.Length >= 3 &&
                Enum.TryParse(response.Substring(0, 3), ignoreCase: true, out Command parsedCommand))
            {
                return parsedCommand;
            }

            io.WriteLine("Enter one of the following:");
            foreach (var command in Enum.GetValues(typeof(Command)).OfType<Command>())
            {
                io.WriteLine($"  {command}  ({command.GetDescription()})");
            }
            io.WriteLine();
        }
    }

    // 尝试读取航向
    internal static bool TryReadCourse(this IReadWrite io, string prompt, string officer, out Course course)
    {
        // 如果无法从输入中读取1到9之间的数字，则输出错误信息并返回false
        if (!io.TryReadNumberInRange(prompt, 1, 9, out var direction))
        {
            io.WriteLine($"{officer} reports, 'Incorrect course data, sir!'");
            course = default;
            return false;
        }

        // 根据输入的方向创建一个新的课程对象
        course = new Course(direction);
        return true;
    }

    // 根据给定的模式从输入中获取Yes/No的回答
    internal static bool GetYesNo(this IReadWrite io, string prompt, YesNoMode mode)
    {
        // 读取用户输入的回答并转换为大写形式
        var response = io.ReadString($"{prompt} (Y/N)").ToUpperInvariant();

        // 根据模式和用户回答返回相应的布尔值
        return (mode, response) switch
        {
            (YesNoMode.FalseOnN, "N") => false,
            (YesNoMode.FalseOnN, _) => true,
            (YesNoMode.TrueOnY, "Y") => true,
            (YesNoMode.TrueOnY, _) => false,
            _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Invalid value")
        };
    }

    // 枚举类型，表示在Y或N时返回true或false
    internal enum YesNoMode
    {
        TrueOnY,
        FalseOnN
    }
# 闭合前面的函数定义
```