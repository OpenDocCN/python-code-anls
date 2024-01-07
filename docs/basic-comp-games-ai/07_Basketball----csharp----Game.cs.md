# `basic-computer-games\07_Basketball\csharp\Game.cs`

```

// 引入篮球比赛中需要使用的类和资源
using Basketball.Plays;
using Basketball.Resources;
using Games.Common.IO;
using Games.Common.Randomness;

namespace Basketball;

// 定义篮球比赛的类
internal class Game
{
    // 定义私有变量，用于存储比赛中的时钟、记分牌、文本输入输出和随机数生成器
    private readonly Clock _clock;
    private readonly Scoreboard _scoreboard;
    private readonly TextIO _io;
    private readonly IRandom _random;

    // 构造函数，用于初始化比赛中的时钟、记分牌、文本输入输出和随机数生成器
    private Game(Clock clock, Scoreboard scoreboard, TextIO io, IRandom random)
    {
        _clock = clock;
        _scoreboard = scoreboard;
        _io = io;
        _random = random;
    }

    // 创建比赛实例的静态方法，用于初始化比赛并返回比赛实例
    public static Game Create(TextIO io, IRandom random)
    {
        // 输出比赛介绍
        io.Write(Resource.Streams.Introduction);

        // 初始化防守策略和时钟
        var defense = new Defense(io.ReadDefense("Your starting defense will be"));
        var clock = new Clock(io);

        io.WriteLine();

        // 初始化记分牌和两支球队
        var scoreboard = new Scoreboard(
            new Team("Dartmouth", new HomeTeamPlay(io, random, clock, defense)),
            new Team(io.ReadString("Choose your opponent"), new VisitingTeamPlay(io, random, clock, defense)),
            io);

        // 返回初始化后的比赛实例
        return new Game(clock, scoreboard, io, random);
    }

    // 进行比赛的方法
    public void Play()
    {
        // 初始化控球比赛
        var ballContest = new BallContest(0.4f, "{0} controls the tap", _io, _random);

        // 循环进行比赛
        while (true)
        {
            // 输出比赛开始信息并解决控球比赛
            _io.WriteLine("Center jump");
            ballContest.Resolve(_scoreboard);

            _io.WriteLine();

            // 循环进行比赛的每个阶段
            while (true)
            {
                // 解决进攻阶段并判断比赛是否结束
                var isFullTime = _scoreboard.Offense.ResolvePlay(_scoreboard);
                if (isFullTime && IsGameOver()) { return; }
                if (_clock.IsHalfTime) { break; }
            }
        }
    }

    // 判断比赛是否结束的方法
    private bool IsGameOver()
    {
        _io.WriteLine();
        // 如果比分相等，则进入加时赛
        if (_scoreboard.ScoresAreEqual)
        {
            _scoreboard.Display(Resource.Formats.EndOfSecondHalf);
            _clock.StartOvertime();
            return false;
        }

        // 输出比赛结束信息并返回比赛结束
        _scoreboard.Display(Resource.Formats.EndOfGame);
        return true;
    }
}

```