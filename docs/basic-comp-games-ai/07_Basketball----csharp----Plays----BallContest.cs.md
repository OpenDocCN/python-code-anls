# `basic-computer-games\07_Basketball\csharp\Plays\BallContest.cs`

```py
// 使用 Games.Common.IO 命名空间中的 IReadWrite 接口
// 使用 Games.Common.Randomness 命名空间中的 IRandom 接口
using Games.Common.IO;
using Games.Common.Randomness;

// 在 Basketball.Plays 命名空间中定义 BallContest 类
internal class BallContest
{
    // 保存比赛概率的私有字段
    private readonly float _probability;
    // 保存消息格式的私有字段
    private readonly string _messageFormat;
    // 保存 IReadWrite 接口的实例的私有字段
    private readonly IReadWrite _io;
    // 保存 IRandom 接口的实例的私有字段
    private readonly IRandom _random;

    // 构造函数，接受概率、消息格式、IReadWrite 接口实例和 IRandom 接口实例作为参数
    internal BallContest(float probability, string messageFormat, IReadWrite io, IRandom random)
    {
        // 初始化 _io 字段
        _io = io;
        // 初始化 _probability 字段
        _probability = probability;
        // 初始化 _messageFormat 字段
        _messageFormat = messageFormat;
        // 初始化 _random 字段
        _random = random;
    }

    // 解决比赛，接受 Scoreboard 对象作为参数
    internal bool Resolve(Scoreboard scoreboard)
    {
        // 根据概率随机确定获胜方，并将其设置为进攻方
        var winner = _random.NextFloat() <= _probability ? scoreboard.Home : scoreboard.Visitors;
        scoreboard.Offense = winner;
        // 输出消息，消息格式为 _messageFormat，获胜方为 winner
        _io.WriteLine(_messageFormat, winner);
        // 返回 false
        return false;
    }
}
```