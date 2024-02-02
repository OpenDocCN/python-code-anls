# `basic-computer-games\07_Basketball\csharp\Clock.cs`

```py
// 使用 Basketball.Resources 命名空间
// 使用 Games.Common.IO 命名空间
namespace Basketball
{
    // Clock 类
    internal class Clock
    {
        // 私有字段 _io，类型为 IReadWrite 接口
        private readonly IReadWrite _io;
        // 整型字段 time
        private int time;

        // Clock 类的构造函数，接受一个 IReadWrite 接口实例作为参数
        public Clock(IReadWrite io) => _io = io;

        // 只读属性 IsHalfTime，判断时间是否为半场结束
        public bool IsHalfTime => time == 50;
        // 只读属性 IsFullTime，判断时间是否为比赛结束
        public bool IsFullTime => time >= 100;
        // 只读属性 TwoMinutesLeft，判断时间是否为比赛结束前两分钟
        public bool TwoMinutesLeft => time == 92;

        // Increment 方法，接受一个 Scoreboard 实例作为参数
        public void Increment(Scoreboard scoreboard)
        {
            // 时间加一
            time += 1;
            // 如果时间为半场结束，显示 Resource.Formats.EndOfFirstHalf
            if (IsHalfTime) { scoreboard.Display(Resource.Formats.EndOfFirstHalf); }
            // 如果时间为比赛结束前两分钟，写入 Resource.Streams.TwoMinutesLeft
            if (TwoMinutesLeft) { _io.Write(Resource.Streams.TwoMinutesLeft); }
        }

        // StartOvertime 方法，将时间设置为加时赛开始的时间
        public void StartOvertime() => time = 93;
    }
}
```