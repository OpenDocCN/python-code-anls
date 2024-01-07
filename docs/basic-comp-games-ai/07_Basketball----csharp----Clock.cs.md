# `basic-computer-games\07_Basketball\csharp\Clock.cs`

```

// 使用 Basketball.Resources 命名空间
// 使用 Games.Common.IO 命名空间
namespace Basketball;

// Clock 类，用于管理比赛时间
internal class Clock
{
    // 用于读写操作的接口
    private readonly IReadWrite _io;
    // 当前时间
    private int time;

    // 构造函数，接受一个 IReadWrite 接口实例作为参数
    public Clock(IReadWrite io) => _io = io;

    // 判断是否是半场结束
    public bool IsHalfTime => time == 50;
    // 判断是否是比赛结束
    public bool IsFullTime => time >= 100;
    // 判断是否还剩两分钟
    public bool TwoMinutesLeft => time == 92;

    // 时间增加一秒，并根据条件执行相应操作
    public void Increment(Scoreboard scoreboard)
    {
        time += 1;
        // 如果是半场结束，显示相应信息
        if (IsHalfTime) { scoreboard.Display(Resource.Formats.EndOfFirstHalf); }
        // 如果还剩两分钟，写入相应信息
        if (TwoMinutesLeft) { _io.Write(Resource.Streams.TwoMinutesLeft); }
    }

    // 开始加时赛，时间设为 93
    public void StartOvertime() => time = 93;
}

```