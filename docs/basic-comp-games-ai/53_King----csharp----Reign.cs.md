# `basic-computer-games\53_King\csharp\Reign.cs`

```
namespace King;

internal class Reign
{
    // 定义最大任期为8
    public const int MaxTerm = 8;

    // 私有字段，用于读写操作
    private readonly IReadWrite _io;
    // 私有字段，用于生成随机数
    private readonly IRandom _random;
    // 私有字段，表示国家
    private readonly Country _country;
    // 私有字段，表示年数
    private float _yearNumber;

    // 构造函数，接受读写操作和随机数生成器
    public Reign(IReadWrite io, IRandom random)
        : this(io, random, new Country(io, random), 1)
    {
    }

    // 构造函数，接受读写操作、随机数生成器、国家和年数
    public Reign(IReadWrite io, IRandom random, Country country, float year)
    {
        _io = io;
        _random = random;
        _country = country;
        _yearNumber = year;
    }

    // 模拟一年的游戏过程
    public bool PlayYear()
    {
        // 创建一个年份对象
        var year = new Year(_country, _random, _io);

        // 将年份对象的状态写入
        _io.Write(year.Status);

        // 获取玩家的行动，如果为空则评估结果，如果还为空则判断是否任期结束
        var result = year.GetPlayerActions() ?? year.EvaluateResults() ?? IsAtEndOfTerm();
        // 如果游戏结束，则输出消息并返回false
        if (result.IsGameOver)
        {
            _io.WriteLine(result.Message);
            return false;
        }

        // 游戏继续
        return true;
    }

    // 判断是否任期结束
    private Result IsAtEndOfTerm() 
        => _yearNumber == MaxTerm 
            // 如果是最后一任，则返回游戏结束并输出祝贺消息
            ? Result.GameOver(EndCongratulations(MaxTerm)) 
            // 否则继续游戏
            : Result.Continue;
}
```