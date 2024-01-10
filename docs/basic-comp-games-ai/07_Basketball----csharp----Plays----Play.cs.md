# `basic-computer-games\07_Basketball\csharp\Plays\Play.cs`

```
// 引入 Games.Common.IO 命名空间
using Games.Common.IO;
// 引入 Games.Common.Randomness 命名空间
using Games.Common.Randomness;

// 定义 Basketball.Plays 命名空间
namespace Basketball.Plays
{
    // 定义 Play 抽象类
    internal abstract class Play
    {
        // 声明私有只读字段 _io，_random，_clock
        private readonly IReadWrite _io;
        private readonly IRandom _random;
        private readonly Clock _clock;

        // 构造函数，接受 IReadWrite 类型的 io，IRandom 类型的 random，Clock 类型的 clock 参数
        public Play(IReadWrite io, IRandom random, Clock clock)
        {
            // 初始化 _io 字段
            _io = io;
            // 初始化 _random 字段
            _random = random;
            // 初始化 _clock 字段
            _clock = clock;
        }

        // 保护方法，用于增加比赛时间并检查是否到达半场
        protected bool ClockIncrementsToHalfTime(Scoreboard scoreboard)
        {
            // 增加比赛时间
            _clock.Increment(scoreboard);
            // 返回是否到达半场
            return _clock.IsHalfTime;
        }

        // 内部抽象方法，用于解决比赛得分
        internal abstract bool Resolve(Scoreboard scoreboard);

        // 保护方法，用于解决罚球情况
        protected void ResolveFreeThrows(Scoreboard scoreboard, string message) =>
            // 调用 Resolve 方法，根据概率执行不同的动作
            Resolve(message)
                .Do(0.49f, () => scoreboard.AddFreeThrows(2, "Shooter makes both shots."))
                .Or(0.75f, () => scoreboard.AddFreeThrows(1, "Shooter makes one shot and misses one."))
                .Or(() => scoreboard.AddFreeThrows(0, "Both shots missed."));

        // 保护方法，用于解决可能发生的情况
        protected Probably Resolve(string message) => Resolve(message, 1f);

        // 保护方法，用于解决可能发生的情况，并考虑防守因素
        protected Probably Resolve(string message, float defenseFactor)
        {
            // 输出消息
            _io.WriteLine(message);
            // 返回一个新的 Probably 对象，考虑防守因素
            return new Probably(defenseFactor, _random);
        }
    }
}
```