# `basic-computer-games\46_Hexapawn\csharp\GameSeries.cs`

```

using System.Collections.Generic; 
using System.Linq; 
using Games.Common.IO; 
using Games.Common.Randomness; 
using Hexapawn.Resources;

namespace Hexapawn;

// Runs series of games between the computer and the human player
internal class GameSeries
{
    private readonly TextIO _io; // 创建一个私有的文本输入输出对象_io
    private readonly Computer _computer; // 创建一个私有的计算机对象_computer
    private readonly Human _human; // 创建一个私有的人类对象_human
    private readonly Dictionary<object, int> _wins; // 创建一个私有的字典对象_wins，用于记录每个对象的胜利次数

    public GameSeries(TextIO io, IRandom random) // 构造函数，接受文本输入输出对象和随机数生成器对象作为参数
    {
        _io = io; // 初始化文本输入输出对象
        _computer = new(io, random); // 初始化计算机对象
        _human = new(io); // 初始化人类对象
        _wins = new() { [_computer] = 0, [_human] = 0 }; // 初始化字典对象，将计算机和人类对象作为键，初始胜利次数为0
    }

    public void Play() // 游戏进行方法
    {
        _io.Write(Resource.Streams.Title); // 输出游戏标题

        if (_io.GetYesNo("Instructions") == 'Y') // 如果用户选择查看游戏说明
        {
            _io.Write(Resource.Streams.Instructions); // 输出游戏说明
        }

        while (true) // 无限循环
        {
            var game = new Game(_io); // 创建一个新的游戏对象

            var winner = game.Play(_human, _computer); // 进行游戏，并返回胜利者
            _wins[winner]++; // 胜利者的胜利次数加一
            _io.WriteLine(winner == _computer ? "I win." : "You win."); // 输出胜利者是计算机还是玩家

            _io.Write($"I have won {_wins[_computer]} and you {_wins[_human]}"); // 输出计算机和玩家的胜利次数
            _io.WriteLine($" out of {_wins.Values.Sum()} games."); // 输出总游戏次数
            _io.WriteLine(); // 输出空行
        }
    }
}

```