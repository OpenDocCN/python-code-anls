# `d:/src/tocomm/basic-computer-games\07_Basketball\csharp\Game.cs`

```
using Basketball.Plays;  # 导入篮球比赛战术的相关代码
using Basketball.Resources;  # 导入篮球比赛资源的相关代码
using Games.Common.IO;  # 导入通用游戏输入输出的相关代码
using Games.Common.Randomness;  # 导入通用游戏随机数生成的相关代码

namespace Basketball;  # 命名空间声明，定义代码所属的命名空间

internal class Game  # 定义一个内部类 Game
{
    private readonly Clock _clock;  # 声明一个只读的 Clock 类型的私有变量 _clock
    private readonly Scoreboard _scoreboard;  # 声明一个只读的 Scoreboard 类型的私有变量 _scoreboard
    private readonly TextIO _io;  # 声明一个只读的 TextIO 类型的私有变量 _io
    private readonly IRandom _random;  # 声明一个只读的 IRandom 类型的私有变量 _random

    private Game(Clock clock, Scoreboard scoreboard, TextIO io, IRandom random)  # 定义一个私有的构造函数，接受 Clock、Scoreboard、TextIO 和 IRandom 类型的参数
    {
        _clock = clock;  # 将传入的 clock 参数赋值给 _clock 变量
        _scoreboard = scoreboard;  # 将传入的 scoreboard 参数赋值给 _scoreboard 变量
        _io = io;  # 将传入的 io 参数赋值给 _io 变量
        _random = random;  # 将传入的 random 参数赋值给 _random 变量
    }

    # 创建游戏对象的静态方法，接受输入输出对象和随机数生成器作为参数
    public static Game Create(TextIO io, IRandom random)
    {
        # 输出游戏介绍
        io.Write(Resource.Streams.Introduction);

        # 创建防守对象，根据用户输入设置初始防守值
        var defense = new Defense(io.ReadDefense("Your starting defense will be"));
        # 创建时钟对象
        var clock = new Clock(io);

        # 输出空行
        io.WriteLine();

        # 创建比分板对象，包括主队和客队，根据用户输入设置客队名称
        var scoreboard = new Scoreboard(
            new Team("Dartmouth", new HomeTeamPlay(io, random, clock, defense)),
            new Team(io.ReadString("Choose your opponent"), new VisitingTeamPlay(io, random, clock, defense)),
            io);

        # 返回新创建的游戏对象
        return new Game(clock, scoreboard, io, random);
    }

    # 游戏进行方法
    public void Play()
    {
        // 创建一个 BallContest 对象，设置跳球概率为 0.4，以及控制台输出的消息模板
        var ballContest = new BallContest(0.4f, "{0} controls the tap", _io, _random);

        // 无限循环，直到游戏结束
        while (true)
        {
            // 输出"Center jump"到控制台
            _io.WriteLine("Center jump");
            // 解决跳球，更新比分
            ballContest.Resolve(_scoreboard);

            // 输出空行到控制台
            _io.WriteLine();

            // 再次无限循环，直到比赛结束或者半场结束
            while (true)
            {
                // 执行进攻防守，并返回是否比赛结束
                var isFullTime = _scoreboard.Offense.ResolvePlay(_scoreboard);
                // 如果比赛结束且游戏结束条件满足，则返回
                if (isFullTime && IsGameOver()) { return; }
                // 如果比赛处于半场休息状态，则跳出内层循环
                if (_clock.IsHalfTime) { break; }
            }
        }
    }

    // 检查游戏是否结束的私有方法
    private bool IsGameOver()
    {
        _io.WriteLine();  # 输出空行
        if (_scoreboard.ScoresAreEqual)  # 如果比分相等
        {
            _scoreboard.Display(Resource.Formats.EndOfSecondHalf);  # 显示上半场结束的消息
            _clock.StartOvertime();  # 开始加时赛
            return false;  # 返回 false
        }

        _scoreboard.Display(Resource.Formats.EndOfGame);  # 显示比赛结束的消息
        return true;  # 返回 true
    }
}
```