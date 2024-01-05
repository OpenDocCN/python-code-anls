# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Commands\CommandExtensions.cs`

```
# 使用 System.Reflection 和 System.ComponentModel 模块
using System.Reflection;
using System.ComponentModel;

# 将命令扩展为静态类
namespace SuperStarTrek.Commands;

internal static class CommandExtensions
{
    # 将命令的描述作为扩展方法
    internal static string GetDescription(this Command command) =>
        # 获取 Command 类型的字段
        typeof(Command)
            # 获取特定命令的字段
            .GetField(command.ToString())
            # 获取字段的 DescriptionAttribute 自定义属性
            .GetCustomAttribute<DescriptionAttribute>()
            # 返回 DescriptionAttribute 的描述
            .Description;
}
```