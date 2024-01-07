# `basic-computer-games\26_Chomp\csharp\Game.cs`

```

namespace Chomp;

internal class Game
{
    private readonly IReadWrite _io; // 声明一个私有的 IReadWrite 接口类型的变量_io

    public Game(IReadWrite io) // 构造函数，接受一个 IReadWrite 接口类型的参数io
    {
        _io = io; // 将参数io赋值给成员变量_io
    }

    internal void Play() // 定义一个内部方法Play
    {
        _io.Write(Resource.Streams.Introduction); // 使用_io对象的Write方法输出介绍信息
        if (_io.ReadNumber("Do you want the rules (1=Yes, 0=No!)") != 0) // 使用_io对象的ReadNumber方法读取用户输入，如果不等于0
        {
            _io.Write(Resource.Streams.Rules); // 输出游戏规则
        }

        while (true) // 进入无限循环
        {
            _io.Write(Resource.Streams.HereWeGo); // 输出游戏开始提示

            var (playerCount, rowCount, columnCount) = _io.ReadParameters(); // 使用_io对象的ReadParameters方法读取玩家数量、行数和列数

            var loser = Play(new Cookie(rowCount, columnCount), new PlayerNumber(playerCount)); // 调用Play方法进行游戏，并将返回的失败者赋值给loser

            _io.WriteLine(string.Format(Resource.Formats.YouLose, loser)); // 使用_io对象的WriteLine方法输出失败者信息

            if (_io.ReadNumber("Again (1=Yes, 0=No!)") != 1) { break; } // 使用_io对象的ReadNumber方法读取用户输入，如果不等于1则跳出循环
        }
    }

    private PlayerNumber Play(Cookie cookie, PlayerNumber player) // 定义一个私有方法Play，接受Cookie对象和PlayerNumber对象作为参数
    {
        while (true) // 进入无限循环
        {
            _io.WriteLine(cookie); // 使用_io对象的WriteLine方法输出cookie信息

            var poisoned = Chomp(cookie, player); // 调用Chomp方法进行游戏，将返回值赋值给poisoned

            if (poisoned) { return player; } // 如果poisoned为true，则返回当前玩家

            player++; // 玩家数加一
        }
    }

    private bool Chomp(Cookie cookie, PlayerNumber player) // 定义一个私有方法Chomp，接受Cookie对象和PlayerNumber对象作为参数
    {
        while (true) // 进入无限循环
        {
            _io.WriteLine(string.Format(Resource.Formats.Player, player)); // 使用_io对象的WriteLine方法输出当前玩家信息

            var (row, column) = _io.Read2Numbers(Resource.Prompts.Coordinates); // 使用_io对象的Read2Numbers方法读取用户输入的行和列

            if (cookie.TryChomp((int)row, (int)column, out char chomped)) // 调用cookie对象的TryChomp方法，尝试进行游戏
            {
                return chomped == 'P'; // 如果被吃的是毒药，则返回true
            }

            _io.Write(Resource.Streams.NoFair); // 输出公平提示
        }
    }
}

```