# `basic-computer-games\53_King\csharp\Reign.cs`

```

namespace King; // 命名空间声明，定义了代码所属的命名空间

internal class Reign // 定义了一个内部类 Reign
{
    public const int MaxTerm = 8; // 定义了一个常量 MaxTerm，表示最大任期为8

    private readonly IReadWrite _io; // 声明了一个私有只读字段 _io，类型为 IReadWrite
    private readonly IRandom _random; // 声明了一个私有只读字段 _random，类型为 IRandom
    private readonly Country _country; // 声明了一个私有只读字段 _country，类型为 Country
    private float _yearNumber; // 声明了一个私有字段 _yearNumber，类型为 float

    public Reign(IReadWrite io, IRandom random) // 定义了一个构造函数，接受 IReadWrite 和 IRandom 类型的参数
        : this(io, random, new Country(io, random), 1) // 调用另一个构造函数，并传入默认参数
    {
    }

    public Reign(IReadWrite io, IRandom random, Country country, float year) // 定义了一个构造函数，接受 IReadWrite、IRandom、Country 和 float 类型的参数
    {
        _io = io; // 将参数 io 赋值给私有字段 _io
        _random = random; // 将参数 random 赋值给私有字段 _random
        _country = country; // 将参数 country 赋值给私有字段 _country
        _yearNumber = year; // 将参数 year 赋值给私有字段 _yearNumber
    }

    public bool PlayYear() // 定义了一个公共方法 PlayYear，返回布尔值
    {
        var year = new Year(_country, _random, _io); // 创建了一个 Year 对象，传入 _country、_random 和 _io 参数

        _io.Write(year.Status); // 调用 _io 的 Write 方法，传入 year.Status 参数

        var result = year.GetPlayerActions() ?? year.EvaluateResults() ?? IsAtEndOfTerm(); // 使用空合并运算符获取结果
        if (result.IsGameOver) // 如果结果为游戏结束
        {
            _io.WriteLine(result.Message); // 调用 _io 的 WriteLine 方法，传入 result.Message 参数
            return false; // 返回 false
        }

        return true; // 返回 true
    }

    private Result IsAtEndOfTerm()  // 定义了一个私有方法 IsAtEndOfTerm，返回 Result 类型
        => _yearNumber == MaxTerm  // 如果 _yearNumber 等于 MaxTerm
            ? Result.GameOver(EndCongratulations(MaxTerm))  // 返回游戏结束的结果，调用 EndCongratulations 方法
            : Result.Continue; // 否则返回继续游戏的结果
}

```