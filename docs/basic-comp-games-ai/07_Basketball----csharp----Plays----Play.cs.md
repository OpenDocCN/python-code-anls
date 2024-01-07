# `basic-computer-games\07_Basketball\csharp\Plays\Play.cs`

```

# 导入所需的命名空间
using Games.Common.IO;
using Games.Common.Randomness;

# 定义一个篮球战术的抽象类
namespace Basketball.Plays;

internal abstract class Play
{
    # 定义私有成员变量，分别为读写接口、随机数生成器和时钟
    private readonly IReadWrite _io;
    private readonly IRandom _random;
    private readonly Clock _clock;

    # 构造函数，初始化私有成员变量
    public Play(IReadWrite io, IRandom random, Clock clock)
    {
        _io = io;
        _random = random;
        _clock = clock;
    }

    # 时钟增加到半场时间
    protected bool ClockIncrementsToHalfTime(Scoreboard scoreboard)
    {
        _clock.Increment(scoreboard);
        return _clock.IsHalfTime;
    }

    # 抽象方法，用于解决比分
    internal abstract bool Resolve(Scoreboard scoreboard);

    # 解决罚球情况
    protected void ResolveFreeThrows(Scoreboard scoreboard, string message) =>
        Resolve(message)
            .Do(0.49f, () => scoreboard.AddFreeThrows(2, "Shooter makes both shots."))
            .Or(0.75f, () => scoreboard.AddFreeThrows(1, "Shooter makes one shot and misses one."))
            .Or(() => scoreboard.AddFreeThrows(0, "Both shots missed."));

    # 解决可能性，返回一个概率对象
    protected Probably Resolve(string message) => Resolve(message, 1f);

    # 解决可能性，返回一个概率对象，带有防守因素
    protected Probably Resolve(string message, float defenseFactor)
    {
        _io.WriteLine(message);
        return new Probably(defenseFactor, _random);
    }
}

```