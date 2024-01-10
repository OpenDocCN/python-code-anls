# `basic-computer-games\07_Basketball\csharp\Plays\VisitingTeamPlay.cs`

```
using Games.Common.IO;  // 导入 Games.Common.IO 命名空间
using Games.Common.Randomness;  // 导入 Games.Common.Randomness 命名空间

namespace Basketball.Plays;  // 定义 Basketball.Plays 命名空间

internal class VisitingTeamPlay : Play  // 定义 VisitingTeamPlay 类，继承自 Play 类
{
    private readonly TextIO _io;  // 声明私有只读字段 _io，类型为 TextIO
    private readonly IRandom _random;  // 声明私有只读字段 _random，类型为 IRandom
    private readonly Defense _defense;  // 声明私有只读字段 _defense，类型为 Defense

    public VisitingTeamPlay(TextIO io, IRandom random, Clock clock, Defense defense)  // 定义 VisitingTeamPlay 类的构造函数，接受 TextIO、IRandom、Clock 和 Defense 参数
        : base(io, random, clock)  // 调用基类 Play 的构造函数，传入 io、random 和 clock 参数
    {
        _io = io;  // 将构造函数参数 io 赋值给 _io 字段
        _random = random;  // 将构造函数参数 random 赋值给 _random 字段
        _defense = defense;  // 将构造函数参数 defense 赋值给 _defense 字段
    }

    internal override bool Resolve(Scoreboard scoreboard)  // 定义 Resolve 方法，接受 Scoreboard 参数，返回布尔值
    {
        if (ClockIncrementsToHalfTime(scoreboard)) { return false; }  // 如果 ClockIncrementsToHalfTime 方法返回 true，则返回 false

        _io.WriteLine();  // 调用 _io 对象的 WriteLine 方法

        var shot = _random.NextShot();  // 使用 _random 对象的 NextShot 方法获取 shot

        if (shot is JumpShot jumpShot)  // 如果 shot 是 JumpShot 类型的实例
        {
            var continuePlay = Resolve(jumpShot, scoreboard);  // 调用 Resolve 方法，传入 jumpShot 和 scoreboard 参数，赋值给 continuePlay
            _io.WriteLine();  // 调用 _io 对象的 WriteLine 方法
            if (!continuePlay) { return false; }  // 如果 continuePlay 为 false，则返回 false
        }

        while (true)  // 无限循环
        {
            var continuePlay = Resolve(shot, scoreboard);  // 调用 Resolve 方法，传入 shot 和 scoreboard 参数，赋值给 continuePlay
            _io.WriteLine();  // 调用 _io 对象的 WriteLine 方法
            if (!continuePlay) { return false; }  // 如果 continuePlay 为 false，则返回 false
        }
    }

    // The Resolve* methods resolve the probabilistic outcome of the current game state.
    // They return true if the Visiting team should continue the play and attempt a layup, false otherwise.
    private bool Resolve(JumpShot shot, Scoreboard scoreboard) =>  // 定义私有方法 Resolve，接受 JumpShot 和 Scoreboard 参数，返回布尔值
        Resolve(shot.ToString(), _defense / 8)  // 调用 Resolve 方法，传入 shot.ToString() 和 _defense / 8 参数
            .Do(0.35f, () => scoreboard.AddBasket("Shot is good."))  // 如果概率为 0.35，则调用 scoreboard 的 AddBasket 方法
            .Or(0.75f, () => ResolveBadShot(scoreboard, "Shot is off the rim.", _defense * 6))  // 如果概率为 0.75，则调用 ResolveBadShot 方法
            .Or(0.9f, () => ResolveFreeThrows(scoreboard, "Player fouled.  Two shots."))  // 如果概率为 0.9，则调用 ResolveFreeThrows 方法
            .Or(() => _io.WriteLine($"Offensive foul.  {scoreboard.Home}'s ball."));  // 否则调用 _io 的 WriteLine 方法

    private bool Resolve(Shot shot, Scoreboard scoreboard) =>  // 定义私有方法 Resolve，接受 Shot 和 Scoreboard 参数，返回布尔值
        Resolve(shot.ToString(), _defense / 7)  // 调用 Resolve 方法，传入 shot.ToString() 和 _defense / 7 参数
            .Do(0.413f, () => scoreboard.AddBasket("Shot is good."))  // 如果概率为 0.413，则调用 scoreboard 的 AddBasket 方法
            .Or(() => ResolveBadShot(scoreboard, "Shot is missed.", 6 / _defense));  // 否则调用 ResolveBadShot 方法
}
    # 解决比赛中出现的糟糕出手情况，根据消息和防守因素进行解决
    private bool ResolveBadShot(Scoreboard scoreboard, string message, float defenseFactor) =>
        # 调用 Resolve 方法进行解决，传入防守因素
        Resolve(message, defenseFactor)
            # 如果解决成功，执行以下操作
            .Do(0.5f, () => scoreboard.Turnover($"{scoreboard.Home} controls the rebound."))
            # 如果解决失败，执行以下操作
            .Or(() => ResolveVisitorsRebound(scoreboard));

    # 处理客队抢篮板情况
    private bool ResolveVisitorsRebound(Scoreboard scoreboard)
    {
        # 输出客队抢篮板的信息
        _io.Write($"{scoreboard.Visitors} controls the rebound.")
        # 如果防守等级为6且随机数小于等于0.25，执行以下操作
        if (_defense == 6 && _random.NextFloat() <= 0.25f)
        {
            # 输出换行
            _io.WriteLine();
            # 球权转换
            scoreboard.Turnover();
            # 添加篮球得分信息
            scoreboard.AddBasket($"Ball stolen.  Easy lay up for {scoreboard.Home}.");
            # 返回 false
            return false;
        }
        # 如果随机数小于等于0.5，执行以下操作
        if (_random.NextFloat() <= 0.5f)
        {
            # 输出换行
            _io.WriteLine();
            # 输出将球传回给客队后卫的信息
            _io.Write($"Pass back to {scoreboard.Visitors} guard.");
            # 返回 false
            return false;
        }
        # 返回 true
        return true;
    }
# 闭合前面的函数定义
```