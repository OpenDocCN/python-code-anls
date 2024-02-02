# `basic-computer-games\46_Hexapawn\csharp\GameSeries.cs`

```py
using System.Collections.Generic; // 导入泛型集合类
using System.Linq; // 导入 LINQ 查询
using Games.Common.IO; // 导入自定义的 IO 类
using Games.Common.Randomness; // 导入自定义的随机数类
using Hexapawn.Resources; // 导入游戏资源

namespace Hexapawn; // 命名空间声明

// 运行计算机和人类玩家之间的一系列游戏
internal class GameSeries // 内部类 GameSeries
{
    private readonly TextIO _io; // 只读字段，用于处理文本输入输出
    private readonly Computer _computer; // 只读字段，用于表示计算机玩家
    private readonly Human _human; // 只读字段，用于表示人类玩家
    private readonly Dictionary<object, int> _wins; // 只读字段，用于存储获胜次数

    public GameSeries(TextIO io, IRandom random) // 构造函数，接受文本输入输出和随机数生成器
    {
        _io = io; // 初始化文本输入输出
        _computer = new Computer(io, random); // 初始化计算机玩家
        _human = new Human(io); // 初始化人类玩家
        _wins = new() { [_computer] = 0, [_human] = 0 }; // 初始化获胜次数字典
    }

    public void Play() // Play 方法，用于运行游戏系列
    {
        _io.Write(Resource.Streams.Title); // 输出游戏标题

        if (_io.GetYesNo("Instructions") == 'Y') // 如果用户选择查看游戏说明
        {
            _io.Write(Resource.Streams.Instructions); // 输出游戏说明
        }

        while (true) // 无限循环
        {
            var game = new Game(_io); // 创建新的游戏实例

            var winner = game.Play(_human, _computer); // 进行游戏并获取获胜者
            _wins[winner]++; // 更新获胜者的获胜次数
            _io.WriteLine(winner == _computer ? "I win." : "You win."); // 输出获胜者信息

            _io.Write($"I have won {_wins[_computer]} and you {_wins[_human]}"); // 输出计算机和人类的获胜次数
            _io.WriteLine($" out of {_wins.Values.Sum()} games."); // 输出总游戏次数
            _io.WriteLine(); // 输出空行
        }
    }
}
```