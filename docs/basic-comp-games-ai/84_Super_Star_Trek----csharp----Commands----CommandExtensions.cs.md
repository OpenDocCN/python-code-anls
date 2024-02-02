# `basic-computer-games\84_Super_Star_Trek\csharp\Commands\CommandExtensions.cs`

```py
# 引入 System.Reflection 和 System.ComponentModel 命名空间
using System.Reflection;
using System.ComponentModel;

# 定义 SuperStarTrek.Commands 命名空间
namespace SuperStarTrek.Commands;

# 定义 CommandExtensions 静态类
internal static class CommandExtensions
{
    # 定义 GetDescription 方法，接收 Command 类型参数，返回字符串类型
    internal static string GetDescription(this Command command) =>
        # 使用反射获取 Command 类型中指定字段的 DescriptionAttribute 特性的描述信息
        typeof(Command)
            .GetField(command.ToString())
            .GetCustomAttribute<DescriptionAttribute>()
            .Description;
}
```