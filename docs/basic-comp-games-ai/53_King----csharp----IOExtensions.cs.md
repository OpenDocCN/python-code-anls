# `basic-computer-games\53_King\csharp\IOExtensions.cs`

```

// 导入命名空间 System.Diagnostics.CodeAnalysis，用于使用属性 [NotNullWhen(true)]
// 导入 King.Resources.Resource 命名空间下的所有静态成员
using System.Diagnostics.CodeAnalysis;
using static King.Resources.Resource;

// 声明命名空间 King
namespace King;

// 声明内部静态类 IOExtensions
internal static class IOExtensions
{
    // 声明扩展方法 TryReadGameData，用于尝试读取游戏数据
    internal static bool TryReadGameData(this IReadWrite io, IRandom random, [NotNullWhen(true)] out Reign? reign)
    {
        // 尝试读取保存的年份，并进行有效性检查
        // 尝试读取保存的国库金额
        // 尝试读取保存的国民数量
        // 尝试读取保存的工人数量
        // 尝试读取保存的土地数量，并进行有效性检查
        // 如果以上所有数据读取成功，则创建新的 Reign 对象，并返回 true
        // 否则返回 false
    }

    // 声明扩展方法 TryReadValue，用于尝试读取数值
    internal static bool TryReadValue(this IReadWrite io, string prompt, out float value, params ValidityTest[] tests)
    {
        // 循环读取数值，直到满足所有有效性检查
    }

    // 声明扩展方法 TryReadValue，重载版本，用于尝试读取数值
    internal static bool TryReadValue(this IReadWrite io, string prompt, out float value)
        => io.TryReadValue(prompt, _ => true, "", out value);

    // 声明扩展方法 TryReadValue，重载版本，用于尝试读取数值
    internal static bool TryReadValue(
        this IReadWrite io,
        string prompt,
        Predicate<float> isValid,
        string error,
        out float value)
        => io.TryReadValue(prompt, isValid, () => error, out value);

    // 声明扩展方法 TryReadValue，重载版本，用于尝试读取数值
    internal static bool TryReadValue(
        this IReadWrite io,
        string prompt,
        Predicate<float> isValid,
        Func<string> getError,
        out float value)
    {
        // 循环读取数值，直到满足所有有效性检查
    }
}

```