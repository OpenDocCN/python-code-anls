# `d:/src/tocomm/basic-computer-games\07_Basketball\csharp\Plays\HomeTeamPlay.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 命名空间，以便使用其中的类和方法
using Games.Common.Randomness;  # 导入 Games.Common.Randomness 命名空间，以便使用其中的类和方法

namespace Basketball.Plays;  # 声明 Basketball.Plays 命名空间

internal class HomeTeamPlay : Play  # 声明一个名为 HomeTeamPlay 的类，继承自 Play 类
{
    private readonly TextIO _io;  # 声明一个名为 _io 的私有只读字段，类型为 TextIO
    private readonly IRandom _random;  # 声明一个名为 _random 的私有只读字段，类型为 IRandom
    private readonly Clock _clock;  # 声明一个名为 _clock 的私有只读字段，类型为 Clock
    private readonly Defense _defense;  # 声明一个名为 _defense 的私有只读字段，类型为 Defense
    private readonly BallContest _ballContest;  # 声明一个名为 _ballContest 的私有只读字段，类型为 BallContest

    public HomeTeamPlay(TextIO io, IRandom random, Clock clock, Defense defense)  # 声明一个名为 HomeTeamPlay 的构造函数，接受 TextIO、IRandom、Clock 和 Defense 类型的参数
        : base(io, random, clock)  # 调用基类 Play 的构造函数，传入 io、random 和 clock 参数
    {
        _io = io;  # 将构造函数参数 io 赋值给 _io 字段
        _random = random;  # 将构造函数参数 random 赋值给 _random 字段
        _clock = clock;  # 将构造函数参数 clock 赋值给 _clock 字段
        _defense = defense;  # 将构造函数参数 defense 赋值给 _defense 字段
        _ballContest = new BallContest(0.5f, "Shot is blocked.  Ball controlled by {0}.", _io, _random);
    }
    # 创建一个名为_ballContest的BallContest对象，传入参数0.5f, "Shot is blocked.  Ball controlled by {0}.", _io, _random

    internal override bool Resolve(Scoreboard scoreboard)
    {
        var shot = _io.ReadShot("Your shot");
        # 从_io对象中读取用户输入的投篮信息，并赋值给变量shot

        if (_random.NextFloat() >= 0.5f && _clock.IsFullTime) { return true; }
        # 如果随机数大于等于0.5并且比赛时间已满，则返回true

        if (shot is null)
        {
            _defense.Set(_io.ReadDefense("Your new defensive alignment is"));
            # 如果shot为空，则从_io对象中读取用户输入的新防守布局，并设置给_defense对象
            _io.WriteLine();
            return false;
        }

        if (shot is JumpShot jumpShot)
        {
            if (ClockIncrementsToHalfTime(scoreboard)) { return false; }
            # 如果比赛时间增加到半场时间，则返回false
            if (!Resolve(jumpShot, scoreboard)) { return false; }
            # 如果Resolve方法返回false，则返回false
        }

        do
        {
            // 如果比赛时间增加到半场时间，返回 false
            if (ClockIncrementsToHalfTime(scoreboard)) { return false; }
        } while (Resolve(shot, scoreboard)); // 解决当前比赛状态的概率结果，如果主队应继续比赛并尝试上篮，则返回 true，否则返回 false

        return false;
    }

    // Resolve* 方法解决当前比赛状态的概率结果。
    // 如果主队应继续比赛并尝试上篮，则返回 true，否则返回 false。
    private bool Resolve(JumpShot shot, Scoreboard scoreboard) =>
        Resolve(shot.ToString(), _defense / 8) // 解决当前比赛状态的概率结果，传入投篮方式和防守能力参数
            .Do(0.341f, () => scoreboard.AddBasket("Shot is good")) // 如果概率为 0.341，执行上篮得分操作
            .Or(0.682f, () => ResolveShotOffTarget(scoreboard)) // 如果概率为 0.682，执行投篮偏离目标操作
            .Or(0.782f, () => _ballContest.Resolve(scoreboard)) // 如果概率为 0.782，执行球权争夺操作
            .Or(0.843f, () => ResolveFreeThrows(scoreboard, "Shooter is fouled.  Two shots.")) // 如果概率为 0.843，执行罚球操作
            .Or(() => scoreboard.Turnover($"Charging foul.  {scoreboard.Home} loses ball.")); // 如果以上概率都不满足，执行失误操作
private bool Resolve(Shot shot, Scoreboard scoreboard) =>
    // 根据投篮结果和防守情况来解决得分情况
    Resolve(shot.ToString(), _defense / 7)
        // 如果命中率为0.4，添加篮球得分并返回true
        .Do(0.4f, () => scoreboard.AddBasket("Shot is good.  Two points."))
        // 如果命中率为0.7，解决投篮偏离篮筐的情况
        .Or(0.7f, () => ResolveShotOffTheRim(scoreboard))
        // 如果命中率为0.875，解决罚球情况
        .Or(0.875f, () => ResolveFreeThrows(scoreboard, "Shooter fouled.  Two shots."))
        // 如果命中率为0.925，解决投篮被盖帽的情况
        .Or(0.925f, () => scoreboard.Turnover($"Shot blocked. {scoreboard.Visitors}'s ball."))
        // 如果命中率为其他情况，解决犯规情况
        .Or(() => scoreboard.Turnover($"Charging foul.  {scoreboard.Home} loses ball."));

private bool ResolveShotOffTarget(Scoreboard scoreboard) =>
    // 根据投篮偏离篮筐的情况来解决得分情况
    Resolve("Shot is off target", 6 / _defense)
        // 如果命中率为0.45，解决主队篮板球情况
        .Do(0.45f, () => ResolveHomeRebound(scoreboard, ResolvePossibleSteal))
        // 如果命中率为其他情况，解决客队篮板球情况
        .Or(() => scoreboard.Turnover($"Rebound to {scoreboard.Visitors}"));

private bool ResolveHomeRebound(Scoreboard scoreboard, Action<Scoreboard> endOfPlayAction) =>
    // 解决主队篮板球情况
    Resolve($"{scoreboard.Home} controls the rebound.")
        // 如果命中率为0.4，返回true
        .Do(0.4f, () => true)
        // 如果命中率为其他情况，执行结束比赛动作
        .Or(() => endOfPlayAction.Invoke(scoreboard));

private void ResolvePossibleSteal(Scoreboard scoreboard)
{
    // 如果防守值为6且随机数大于0.6，执行抢断动作
{
    // 球权被转移
    scoreboard.Turnover();
    // 添加一条关于对方球队抢断并轻松上篮的信息
    scoreboard.AddBasket($"Pass stolen by {scoreboard.Visitors} easy layup.");
    // 输出空行
    _io.WriteLine();
}
// 输出“球传回给你”
_io.Write("Ball passed back to you. ");
}

// 解决球从篮筐上弹出的情况
private void ResolveShotOffTheRim(Scoreboard scoreboard) =>
    // 解决“投篮偏出篮筐”的情况
    Resolve("Shot is off the rim.")
        // 以2/3的概率执行以下操作
        .Do(2 / 3f, () => scoreboard.Turnover($"{scoreboard.Visitors} controls the rebound."))
        // 或者执行以下操作
        .Or(() => ResolveHomeRebound(scoreboard, _ => _io.WriteLine("Ball passed back to you.")));
}
```