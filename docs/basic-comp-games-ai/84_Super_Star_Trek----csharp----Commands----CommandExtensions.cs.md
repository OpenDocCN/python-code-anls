# `basic-computer-games\84_Super_Star_Trek\csharp\Commands\CommandExtensions.cs`

```

// 使用 System.Reflection 命名空间，获取程序集中的类型信息
// 使用 System.ComponentModel 命名空间，获取描述特性
namespace SuperStarTrek.Commands;

// 创建一个静态类 CommandExtensions
internal static class CommandExtensions
{
    // 创建一个静态方法，用于获取命令的描述信息
    internal static string GetDescription(this Command command) =>
        // 获取 Command 类型的字段信息
        typeof(Command)
            .GetField(command.ToString())
            // 获取字段上的 DescriptionAttribute 特性
            .GetCustomAttribute<DescriptionAttribute>()
            // 获取 DescriptionAttribute 特性的描述信息
            .Description;
}

```