# `basic-computer-games\07_Basketball\csharp\Plays\HomeTeamPlay.cs`

```py
// 引入所需的命名空间
using Games.Common.IO;
using Games.Common.Randomness;

// 定义篮球比赛中主队的战术类
namespace Basketball.Plays
{
    // 内部类，继承自 Play 类
    internal class HomeTeamPlay : Play
    {
        // 私有字段，用于输入输出
        private readonly TextIO _io;
        // 私有字段，用于生成随机数
        private readonly IRandom _random;
        // 私有字段，用于表示比赛时间
        private readonly Clock _clock;
        // 私有字段，用于表示防守状态
        private readonly Defense _defense;
        // 私有字段，用于表示球权争夺
        private readonly BallContest _ballContest;

        // 构造函数，初始化字段
        public HomeTeamPlay(TextIO io, IRandom random, Clock clock, Defense defense)
            : base(io, random, clock)
        {
            _io = io;
            _random = random;
            _clock = clock;
            _defense = defense;
            // 初始化球权争夺对象
            _ballContest = new BallContest(0.5f, "Shot is blocked.  Ball controlled by {0}.", _io, _random);
        }

        // 重写父类方法，解决当前比赛状态的概率性结果
        internal override bool Resolve(Scoreboard scoreboard)
        {
            // 读取投篮动作
            var shot = _io.ReadShot("Your shot");

            // 如果随机数大于等于0.5且比赛时间已满，则返回 true
            if (_random.NextFloat() >= 0.5f && _clock.IsFullTime) { return true; }

            // 如果投篮动作为空
            if (shot is null)
            {
                // 设置防守状态，并返回 false
                _defense.Set(_io.ReadDefense("Your new defensive alignment is"));
                _io.WriteLine();
                return false;
            }

            // 如果投篮动作为跳投
            if (shot is JumpShot jumpShot)
            {
                // 如果时间增加到中场休息，则返回 false
                if (ClockIncrementsToHalfTime(scoreboard)) { return false; }
                // 如果解决跳投动作的结果为 false，则返回 false
                if (!Resolve(jumpShot, scoreboard)) { return false; }
            }

            // 循环直到解决投篮动作的结果为 false
            do
            {
                // 如果时间增加到中场休息，则返回 false
                if (ClockIncrementsToHalfTime(scoreboard)) { return false; }
            } while (Resolve(shot, scoreboard));

            // 返回 false
            return false;
        }

        // Resolve* 方法解决当前比赛状态的概率性结果
        // 如果主队应继续比赛并尝试上篮，则返回 true，否则返回 false
    }
}
    # 解决跳投事件，根据得分板信息返回布尔值
    private bool Resolve(JumpShot shot, Scoreboard scoreboard) =>
        # 调用重载的 Resolve 方法，传入跳投事件的字符串表示和防守值的八分之一
        Resolve(shot.ToString(), _defense / 8)
            # 如果概率为0.341，执行得分板添加篮球得分的方法
            .Do(0.341f, () => scoreboard.AddBasket("Shot is good"))
            # 如果概率为0.682，执行解决投篮偏离目标的方法
            .Or(0.682f, () => ResolveShotOffTarget(scoreboard))
            # 如果概率为0.782，执行球权争夺解决方法
            .Or(0.782f, () => _ballContest.Resolve(scoreboard))
            # 如果概率为0.843，执行罚球解决方法
            .Or(0.843f, () => ResolveFreeThrows(scoreboard, "Shooter is fouled.  Two shots."))
            # 如果概率不在以上范围内，执行失误解决方法
            .Or(() => scoreboard.Turnover($"Charging foul.  {scoreboard.Home} loses ball."));

    # 解决投篮事件，根据得分板信息返回布尔值
    private bool Resolve(Shot shot, Scoreboard scoreboard) =>
        # 调用重载的 Resolve 方法，传入投篮事件的字符串表示和防守值的七分之一
        Resolve(shot.ToString(), _defense / 7)
            # 如果概率为0.4，执行得分板添加篮球得分的方法
            .Do(0.4f, () => scoreboard.AddBasket("Shot is good.  Two points."))
            # 如果概率为0.7，执行解决投篮偏离篮筐的方法
            .Or(0.7f, () => ResolveShotOffTheRim(scoreboard))
            # 如果概率为0.875，执行罚球解决方法
            .Or(0.875f, () => ResolveFreeThrows(scoreboard, "Shooter fouled.  Two shots."))
            # 如果概率为0.925，执行失误解决方法
            .Or(0.925f, () => scoreboard.Turnover($"Shot blocked. {scoreboard.Visitors}'s ball."))
            # 如果概率不在以上范围内，执行失误解决方法
            .Or(() => scoreboard.Turnover($"Charging foul.  {scoreboard.Home} loses ball."));

    # 解决投篮偏离目标事件，根据得分板信息返回布尔值
    private bool ResolveShotOffTarget(Scoreboard scoreboard) =>
        # 调用重载的 Resolve 方法，传入投篮偏离目标事件的字符串表示和六除以防守值
        Resolve("Shot is off target", 6 / _defense)
            # 如果概率为0.45，执行主队篮板解决方法，并传入解决可能抢断的方法
            .Do(0.45f, () => ResolveHomeRebound(scoreboard, ResolvePossibleSteal))
            # 如果概率不在以上范围内，执行失误解决方法
            .Or(() => scoreboard.Turnover($"Rebound to {scoreboard.Visitors}"));

    # 解决主队篮板事件，根据得分板信息和结束比赛动作返回布尔值
    private bool ResolveHomeRebound(Scoreboard scoreboard, Action<Scoreboard> endOfPlayAction) =>
        # 调用重载的 Resolve 方法，传入主队控制篮板的字符串表示
        Resolve($"{scoreboard.Home} controls the rebound.")
            # 如果概率为0.4，返回 true
            .Do(0.4f, () => true)
            # 如果概率不在以上范围内，执行结束比赛动作
            .Or(() => endOfPlayAction.Invoke(scoreboard));

    # 解决可能抢断事件，根据得分板信息执行相应动作
    private void ResolvePossibleSteal(Scoreboard scoreboard)
    {
        # 如果防守值为6且随机浮点数大于0.6，执行失误解决方法，并添加篮球得分信息
        if (_defense == 6 && _random.NextFloat() > 0.6f)
        {
            scoreboard.Turnover();
            scoreboard.AddBasket($"Pass stolen by {scoreboard.Visitors} easy layup.");
            _io.WriteLine();
        }
        # 输出信息
        _io.Write("Ball passed back to you. ");
    }
    # 解决投篮偏出篮筐的情况
    private void ResolveShotOffTheRim(Scoreboard scoreboard) =>
        # 调用 Resolve 方法，传入描述信息"Shot is off the rim."，并返回 Resolve 对象
        Resolve("Shot is off the rim.")
            # 调用 Do 方法，传入概率2/3和处理函数，如果概率命中，则执行处理函数，否则执行下一个 Or 方法
            .Do(2 / 3f, () => scoreboard.Turnover($"{scoreboard.Visitors} controls the rebound."))
            # 调用 Or 方法，传入处理函数，如果上一个 Do 方法未命中，则执行该处理函数
            .Or(() => ResolveHomeRebound(scoreboard, _ => _io.WriteLine("Ball passed back to you.")));
# 闭合前面的代码块
```