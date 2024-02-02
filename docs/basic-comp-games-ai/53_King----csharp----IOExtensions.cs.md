# `basic-computer-games\53_King\csharp\IOExtensions.cs`

```py
// 引入命名空间 System.Diagnostics.CodeAnalysis，用于指定代码中的非空引用和可空引用的行为
using System.Diagnostics.CodeAnalysis;
// 引入 King.Resources.Resource 命名空间下的 Resource 类
using static King.Resources.Resource;

// 声明 King 命名空间
namespace King;

// 声明内部静态类 IOExtensions
internal static class IOExtensions
{
    // 声明内部静态方法 TryReadGameData，用于尝试读取游戏数据
    internal static bool TryReadGameData(this IReadWrite io, IRandom random, [NotNullWhen(true)] out Reign? reign)
    {
        // 如果成功读取保存的年份、国库、国民、工人和土地
        if (io.TryReadValue(SavedYearsPrompt, v => v < Reign.MaxTerm, SavedYearsError(Reign.MaxTerm), out var years) &&
            io.TryReadValue(SavedTreasuryPrompt, out var rallods) &&
            io.TryReadValue(SavedCountrymenPrompt, out var countrymen) &&
            io.TryReadValue(SavedWorkersPrompt, out var workers) &&
            io.TryReadValue(SavedLandPrompt, v => v is > 1000 and <= 2000, SavedLandError, out var land))
        {
            // 创建新的统治对象，并返回 true
            reign = new Reign(io, random, new Country(io, random, rallods, countrymen, workers, land), years + 1);
            return true;
        }

        // 否则返回默认值，并返回 false
        reign = default;
        return false;
    }

    // 声明内部静态方法 TryReadValue，用于尝试读取值
    internal static bool TryReadValue(this IReadWrite io, string prompt, out float value, params ValidityTest[] tests)
    {
        // 循环直到成功读取值或者用户输入 0
        while (true)
        {
            // 读取用户输入的值
            var response = value = io.ReadNumber(prompt);
            // 如果用户输入 0，则返回 false
            if (response == 0) { return false; }
            // 如果所有的测试都通过，则返回 true
            if (tests.All(test => test.IsValid(response, io))) { return true; }
        } 
    }

    // 声明内部静态方法 TryReadValue，用于尝试读取值
    internal static bool TryReadValue(this IReadWrite io, string prompt, out float value)
        => io.TryReadValue(prompt, _ => true, "", out value);
    
    // 声明内部静态方法 TryReadValue，用于尝试读取值
    internal static bool TryReadValue(
        this IReadWrite io,
        string prompt,
        Predicate<float> isValid,
        string error,
        out float value)
        => io.TryReadValue(prompt, isValid, () => error, out value);

    // 声明内部静态方法 TryReadValue，用于尝试读取值
    internal static bool TryReadValue(
        this IReadWrite io,
        string prompt,
        Predicate<float> isValid,
        Func<string> getError,
        out float value)
    # 进入无限循环，直到条件不满足才退出
    while (true)
    {
        # 从输入流中读取一个数字，并显示提示信息
        value = io.ReadNumber(prompt);
        # 如果读取的值小于0，则返回false
        if (value < 0) { return false; }
        # 如果读取的值有效，则返回true
        if (isValid(value)) { return true; }
        
        # 向输出流写入错误信息
        io.Write(getError());
    }
# 闭合前面的函数定义
```