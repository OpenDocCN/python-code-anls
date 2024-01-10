# `basic-computer-games\26_Chomp\csharp\Game.cs`

```
namespace Chomp;

internal class Game
{
    private readonly IReadWrite _io; // 声明私有的 IReadWrite 接口变量 _io

    public Game(IReadWrite io) // 构造函数，接受一个 IReadWrite 接口类型的参数
    {
        _io = io; // 将传入的参数赋值给私有变量 _io
    }

    internal void Play() // 定义一个内部方法 Play
    {
        _io.Write(Resource.Streams.Introduction); // 使用 _io 对象调用 Write 方法，输出介绍信息
        if (_io.ReadNumber("Do you want the rules (1=Yes, 0=No!)") != 0) // 使用 _io 对象调用 ReadNumber 方法，判断是否需要输出规则
        {
            _io.Write(Resource.Streams.Rules); // 使用 _io 对象调用 Write 方法，输出规则信息
        }

        while (true) // 进入循环
        {
            _io.Write(Resource.Streams.HereWeGo); // 使用 _io 对象调用 Write 方法，输出提示信息

            var (playerCount, rowCount, columnCount) = _io.ReadParameters(); // 调用 _io 对象的 ReadParameters 方法，获取玩家数量、行数和列数

            var loser = Play(new Cookie(rowCount, columnCount), new PlayerNumber(playerCount)); // 调用 Play 方法，传入 Cookie 对象和 PlayerNumber 对象，获取失败的玩家

            _io.WriteLine(string.Format(Resource.Formats.YouLose, loser)); // 使用 _io 对象调用 WriteLine 方法，输出失败玩家信息

            if (_io.ReadNumber("Again (1=Yes, 0=No!)") != 1) { break; } // 使用 _io 对象调用 ReadNumber 方法，判断是否再玩一次，如果不是则跳出循环
        }
    }

    private PlayerNumber Play(Cookie cookie, PlayerNumber player) // 定义一个私有方法 Play，接受 Cookie 对象和 PlayerNumber 对象作为参数
    {
        while (true) // 进入循环
        {
            _io.WriteLine(cookie); // 使用 _io 对象调用 WriteLine 方法，输出 Cookie 对象信息

            var poisoned = Chomp(cookie, player); // 调用 Chomp 方法，获取是否被毒害

            if (poisoned) { return player; } // 如果被毒害，则返回当前玩家

            player++; // 玩家数加一
        }
    }

    private bool Chomp(Cookie cookie, PlayerNumber player) // 定义一个私有方法 Chomp，接受 Cookie 对象和 PlayerNumber 对象作为参数
    {
        while (true) // 进入循环
        {
            _io.WriteLine(string.Format(Resource.Formats.Player, player)); // 使用 _io 对象调用 WriteLine 方法，输出当前玩家信息

            var (row, column) = _io.Read2Numbers(Resource.Prompts.Coordinates); // 调用 _io 对象的 Read2Numbers 方法，获取行和列的输入值

            if (cookie.TryChomp((int)row, (int)column, out char chomped)) // 调用 Cookie 对象的 TryChomp 方法，尝试吃掉指定位置的饼干
            {
                return chomped == 'P'; // 如果吃到毒饼干，则返回 true
            }

            _io.Write(Resource.Streams.NoFair); // 使用 _io 对象调用 Write 方法，输出公平提示信息
        }
    }
}
```