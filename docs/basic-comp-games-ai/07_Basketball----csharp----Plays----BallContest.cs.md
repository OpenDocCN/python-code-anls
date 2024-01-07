# `basic-computer-games\07_Basketball\csharp\Plays\BallContest.cs`

```

// 使用 Games.Common.IO 命名空间中的类
// 使用 Games.Common.Randomness 命名空间中的类
namespace Basketball.Plays;

// 定义一个内部类 BallContest
internal class BallContest
{
    // 保存概率值的字段
    private readonly float _probability;
    // 保存消息格式的字段
    private readonly string _messageFormat;
    // 保存 IReadWrite 接口的字段
    private readonly IReadWrite _io;
    // 保存 IRandom 接口的字段
    private readonly IRandom _random;

    // 构造函数，接受概率值、消息格式、IReadWrite 接口和 IRandom 接口作为参数
    internal BallContest(float probability, string messageFormat, IReadWrite io, IRandom random)
    {
        // 初始化字段值
        _io = io;
        _probability = probability;
        _messageFormat = messageFormat;
        _random = random;
    }

    // 解决比赛，接受 Scoreboard 对象作为参数
    internal bool Resolve(Scoreboard scoreboard)
    {
        // 根据概率值随机确定胜者
        var winner = _random.NextFloat() <= _probability ? scoreboard.Home : scoreboard.Visitors;
        // 将胜者设置为进攻方
        scoreboard.Offense = winner;
        // 输出消息到 IReadWrite 接口
        _io.WriteLine(_messageFormat, winner);
        // 返回 false
        return false;
    }
}

```