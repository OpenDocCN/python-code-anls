# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\IReadWriteExtensions.cs`

```
using System;  # 导入 System 模块
using System.Linq;  # 导入 System 模块中的 Linq 功能
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using SuperStarTrek.Commands;  # 导入 SuperStarTrek.Commands 模块
using SuperStarTrek.Space;  # 导入 SuperStarTrek.Space 模块
using static System.StringComparison;  # 导入 System 模块中的 StringComparison 功能

namespace SuperStarTrek;  # 定义 SuperStarTrek 命名空间

internal static class IReadWriteExtensions  # 定义 IReadWriteExtensions 类
{
    internal static void WaitForAnyKeyButEnter(this IReadWrite io, string prompt)  # 定义 WaitForAnyKeyButEnter 方法
    {
        io.Write($"Hit any key but Enter {prompt} ");  # 在 io 上调用 Write 方法，输出提示信息
        while (io.ReadCharacter() == '\r');  # 当读取的字符为回车符时，继续循环
    }

    internal static (float X, float Y) GetCoordinates(this IReadWrite io, string prompt) =>  # 定义 GetCoordinates 方法
        io.Read2Numbers($"{prompt} (X,Y)");  # 在 io 上调用 Read2Numbers 方法，读取坐标信息
```
        // 从用户输入中读取一个数字，并确保其在指定范围内
        internal static bool TryReadNumberInRange(
            this IReadWrite io,
            string prompt,
            float minValue,
            float maxValue,
            out float value)
        {
            // 从用户输入中读取一个数字，并显示指定的提示信息
            value = io.ReadNumber($"{prompt} ({minValue}-{maxValue})");

            // 检查读取的数字是否在指定范围内，并返回结果
            return value >= minValue && value <= maxValue;
        }

        // 从用户输入中读取一个字符串，并检查是否与指定的值相等
        internal static bool ReadExpectedString(this IReadWrite io, string prompt, string trueValue) =>
            io.ReadString(prompt).Equals(trueValue, InvariantCultureIgnoreCase);

        // 从用户输入中读取一个命令
        internal static Command ReadCommand(this IReadWrite io)
        {
            // 循环读取用户输入，直到得到有效的命令
            while(true)
            {
                // 从用户输入中读取命令
                var response = io.ReadString("Command");
            // 检查响应长度是否大于等于3，并尝试解析前3个字符为枚举类型的命令
            if (response.Length >= 3 &&
                Enum.TryParse(response.Substring(0, 3), ignoreCase: true, out Command parsedCommand))
            {
                return parsedCommand; // 如果成功解析，则返回解析后的命令
            }

            io.WriteLine("Enter one of the following:"); // 输出提示信息
            // 遍历枚举类型的命令，并输出命令及其描述
            foreach (var command in Enum.GetValues(typeof(Command)).OfType<Command>())
            {
                io.WriteLine($"  {command}  ({command.GetDescription()})");
            }
            io.WriteLine(); // 输出空行
        }
    }

    internal static bool TryReadCourse(this IReadWrite io, string prompt, string officer, out Course course)
    {
        // 尝试从输入中读取一个在指定范围内的数字
        if (!io.TryReadNumberInRange(prompt, 1, 9, out var direction))
        {
            io.WriteLine($"{officer} reports, 'Incorrect course data, sir!'");  // 输出警告信息到控制台，指示船长航向数据不正确
            course = default;  // 将航向数据重置为默认值
            return false;  // 返回 false，表示获取航向数据失败
        }

        course = new Course(direction);  // 根据给定的航向数据创建新的航向对象
        return true;  // 返回 true，表示成功获取航向数据
    }

    internal static bool GetYesNo(this IReadWrite io, string prompt, YesNoMode mode)
    {
        var response = io.ReadString($"{prompt} (Y/N)").ToUpperInvariant();  // 从输入流中读取用户的响应，并转换为大写形式

        return (mode, response) switch  // 根据模式和用户响应进行匹配
        {
            (YesNoMode.FalseOnN, "N") => false,  // 如果模式为 FalseOnN 且用户响应为 "N"，返回 false
            (YesNoMode.FalseOnN, _) => true,  // 如果模式为 FalseOnN 且用户响应不为 "N"，返回 true
            (YesNoMode.TrueOnY, "Y") => true,  // 如果模式为 TrueOnY 且用户响应为 "Y"，返回 true
            (YesNoMode.TrueOnY, _) => false,  // 如果模式为 TrueOnY 且用户响应不为 "Y"，返回 false
            _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Invalid value")  // 如果模式和用户响应都不匹配上述情况，抛出参数异常
        };
    }
```
这部分代码是一个类的结束标记和一个命名空间的结束标记。

```
    internal enum YesNoMode
    {
        TrueOnY,
        FalseOnN
    }
```
这部分代码定义了一个内部枚举类型，包括了两个枚举值TrueOnY和FalseOnN。枚举类型用于定义一组命名的整数常量，可以在代码中使用这些常量来代替具体的数值，增加代码的可读性和可维护性。
```