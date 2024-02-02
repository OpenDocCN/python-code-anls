# `basic-computer-games\07_Basketball\csharp\Game.cs`

```py
using Basketball.Plays;  # 导入篮球比赛中的战术类
using Basketball.Resources;  # 导入篮球比赛中的资源类
using Games.Common.IO;  # 导入通用游戏输入输出类
using Games.Common.Randomness;  # 导入通用游戏随机数类

namespace Basketball;  # 命名空间篮球

internal class Game  # 内部类 Game
{
    private readonly Clock _clock;  # 只读时钟对象
    private readonly Scoreboard _scoreboard;  # 只读记分牌对象
    private readonly TextIO _io;  # 只读文本输入输出对象
    private readonly IRandom _random;  # 只读随机数对象

    private Game(Clock clock, Scoreboard scoreboard, TextIO io, IRandom random)  # Game 类的构造函数
    {
        _clock = clock;  # 初始化时钟对象
        _scoreboard = scoreboard;  # 初始化记分牌对象
        _io = io;  # 初始化文本输入输出对象
        _random = random;  # 初始化随机数对象
    }

    public static Game Create(TextIO io, IRandom random)  # 创建 Game 对象的静态方法
    {
        io.Write(Resource.Streams.Introduction);  # 输出游戏介绍

        var defense = new Defense(io.ReadDefense("Your starting defense will be"));  # 创建防守对象
        var clock = new Clock(io);  # 创建时钟对象

        io.WriteLine();  # 输出空行

        var scoreboard = new Scoreboard(  # 创建记分牌对象
            new Team("Dartmouth", new HomeTeamPlay(io, random, clock, defense)),  # 主队
            new Team(io.ReadString("Choose your opponent"), new VisitingTeamPlay(io, random, clock, defense)),  # 客队
            io);

        return new Game(clock, scoreboard, io, random);  # 返回新的 Game 对象
    }

    public void Play()  # 游戏进行方法
    {
        var ballContest = new BallContest(0.4f, "{0} controls the tap", _io, _random);  # 创建球权争夺对象

        while (true)  # 无限循环
        {
            _io.WriteLine("Center jump");  # 输出中心跳球
            ballContest.Resolve(_scoreboard);  # 解决球权争夺

            _io.WriteLine();  # 输出空行

            while (true)  # 无限循环
            {
                var isFullTime = _scoreboard.Offense.ResolvePlay(_scoreboard);  # 解决进攻
                if (isFullTime && IsGameOver()) { return; }  # 如果比赛结束，返回
                if (_clock.IsHalfTime) { break; }  # 如果是中场休息，跳出循环
            }
        }
    }

    private bool IsGameOver()  # 判断游戏是否结束的方法
    {
        _io.WriteLine();  # 输出空行
        if (_scoreboard.ScoresAreEqual)  # 如果比分相等
        {
            _scoreboard.Display(Resource.Formats.EndOfSecondHalf);  # 显示比赛结束的格式
            _clock.StartOvertime();  # 开始加时赛
            return false;  # 返回假
        }

        _scoreboard.Display(Resource.Formats.EndOfGame);  # 显示比赛结束的格式
        return true;  # 返回真
    }
}
```