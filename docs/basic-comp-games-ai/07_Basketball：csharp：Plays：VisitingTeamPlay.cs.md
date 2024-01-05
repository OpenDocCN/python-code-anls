# `07_Basketball\csharp\Plays\VisitingTeamPlay.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 命名空间，以便使用其中的类和方法
using Games.Common.Randomness;  # 导入 Games.Common.Randomness 命名空间，以便使用其中的类和方法

namespace Basketball.Plays;  # 声明 Basketball.Plays 命名空间

internal class VisitingTeamPlay : Play  # 声明 VisitingTeamPlay 类，继承自 Play 类
{
    private readonly TextIO _io;  # 声明私有字段 _io，类型为 TextIO
    private readonly IRandom _random;  # 声明私有字段 _random，类型为 IRandom
    private readonly Defense _defense;  # 声明私有字段 _defense，类型为 Defense

    public VisitingTeamPlay(TextIO io, IRandom random, Clock clock, Defense defense)  # 声明 VisitingTeamPlay 类的构造函数，接受 TextIO、IRandom、Clock 和 Defense 参数
        : base(io, random, clock)  # 调用基类 Play 的构造函数，传入 io、random 和 clock 参数
    {
        _io = io;  # 将构造函数参数 io 赋值给私有字段 _io
        _random = random;  # 将构造函数参数 random 赋值给私有字段 _random
        _defense = defense;  # 将构造函数参数 defense 赋值给私有字段 _defense
    }

    internal override bool Resolve(Scoreboard scoreboard)  # 声明 Resolve 方法，接受 Scoreboard 参数，并且返回布尔类型的值
        if (ClockIncrementsToHalfTime(scoreboard)) { return false; }  // 检查比赛时间是否到达半场，如果是则返回 false

        _io.WriteLine();  // 输出空行

        var shot = _random.NextShot();  // 生成下一次投篮动作

        if (shot is JumpShot jumpShot)  // 如果投篮动作是跳投
        {
            var continuePlay = Resolve(jumpShot, scoreboard);  // 解析跳投动作，更新比分
            _io.WriteLine();  // 输出空行
            if (!continuePlay) { return false; }  // 如果比赛结束，则返回 false
        }

        while (true)  // 无限循环，直到比赛结束
        {
            var continuePlay = Resolve(shot, scoreboard);  // 解析投篮动作，更新比分
            _io.WriteLine();  // 输出空行
            if (!continuePlay) { return false; }  // 如果比赛结束，则返回 false
        }
// 解析跳投动作的概率结果，根据当前比赛状态返回 true 或 false
// 如果客队应该继续比赛并尝试上篮，则返回 true，否则返回 false
private bool Resolve(JumpShot shot, Scoreboard scoreboard) =>
    // 根据跳投动作和防守能力解析概率结果
    Resolve(shot.ToString(), _defense / 8)
        // 如果概率为 0.35，执行投篮命中的操作
        .Do(0.35f, () => scoreboard.AddBasket("Shot is good."))
        // 如果概率为 0.75，执行投篮未中的操作
        .Or(0.75f, () => ResolveBadShot(scoreboard, "Shot is off the rim.", _defense * 6))
        // 如果概率为 0.9，执行罚球的操作
        .Or(0.9f, () => ResolveFreeThrows(scoreboard, "Player fouled.  Two shots."))
        // 如果以上条件都不满足，执行进攻犯规的操作
        .Or(() => _io.WriteLine($"Offensive foul.  {scoreboard.Home}'s ball."));

// 解析投篮动作的概率结果，根据当前比赛状态返回 true 或 false
private bool Resolve(Shot shot, Scoreboard scoreboard) =>
    // 根据投篮动作和防守能力解析概率结果
    Resolve(shot.ToString(), _defense / 7)
        // 如果概率为 0.413，执行投篮命中的操作
        .Do(0.413f, () => scoreboard.AddBasket("Shot is good."))
        // 如果以上条件都不满足，执行投篮未中的操作
        .Or(() => ResolveBadShot(scoreboard, "Shot is missed.", 6 / _defense));

// 解析投篮未中的概率结果，根据当前比赛状态返回 true 或 false
private bool ResolveBadShot(Scoreboard scoreboard, string message, float defenseFactor) =>
    // 根据投篮未中的情况和防守能力解析概率结果
    Resolve(message, defenseFactor)
        // 如果概率为 0.5，执行失误的操作
        .Do(0.5f, () => scoreboard.Turnover($"{scoreboard.Home} controls the rebound."))
        // 如果以上条件都不满足，执行客队抢篮板的操作
        .Or(() => ResolveVisitorsRebound(scoreboard));
# 解析访客队篮板球的情况
private bool ResolveVisitorsRebound(Scoreboard scoreboard)
{
    # 输出访客队控制篮板球的信息
    _io.Write($"{scoreboard.Visitors} controls the rebound.");
    # 如果防守队员为6且随机数小于等于0.25，则发生失误
    if (_defense == 6 && _random.NextFloat() <= 0.25f)
    {
        # 输出换行
        _io.WriteLine();
        # 记录失误并增加得分
        scoreboard.Turnover();
        scoreboard.AddBasket($"Ball stolen.  Easy lay up for {scoreboard.Home}.");
        return false;
    }
    # 如果随机数小于等于0.5，则传球回给访客队后卫
    if (_random.NextFloat() <= 0.5f)
    {
        # 输出换行
        _io.WriteLine();
        # 输出传球回给访客队后卫的信息
        _io.Write($"Pass back to {scoreboard.Visitors} guard.");
        return false;
    }
    # 其他情况返回true
    return true;
}
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```