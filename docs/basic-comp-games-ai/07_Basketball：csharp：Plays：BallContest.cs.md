# `d:/src/tocomm/basic-computer-games\07_Basketball\csharp\Plays\BallContest.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using Games.Common.Randomness;  # 导入 Games.Common.Randomness 模块

namespace Basketball.Plays;  # 命名空间 Basketball.Plays

internal class BallContest  # 内部类 BallContest
{
    private readonly float _probability;  # 只读浮点数 _probability
    private readonly string _messageFormat;  # 只读字符串 _messageFormat
    private readonly IReadWrite _io;  # 只读 IReadWrite 接口类型 _io
    private readonly IRandom _random;  # 只读 IRandom 接口类型 _random

    internal BallContest(float probability, string messageFormat, IReadWrite io, IRandom random)  # 内部 BallContest 构造函数，参数为 probability、messageFormat、io、random
    {
        _io = io;  # 将参数 io 赋值给 _io
        _probability = probability;  # 将参数 probability 赋值给 _probability
        _messageFormat = messageFormat;  # 将参数 messageFormat 赋值给 _messageFormat
        _random = random;  # 将参数 random 赋值给 _random
    }
}
    # 解析比分板，确定胜利方
    internal bool Resolve(Scoreboard scoreboard)
    {
        # 通过随机数判断胜利方，如果随机数小于等于概率，则胜利方为主队，否则为客队
        var winner = _random.NextFloat() <= _probability ? scoreboard.Home : scoreboard.Visitors;
        # 将胜利方设置为进攻方
        scoreboard.Offense = winner;
        # 输出胜利方的消息
        _io.WriteLine(_messageFormat, winner);
        # 返回 false，表示解析完成
        return false;
    }
}
```