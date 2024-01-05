# `d:/src/tocomm/basic-computer-games\46_Hexapawn\csharp\GameSeries.cs`

```
using System.Collections.Generic; // 导入用于定义和操作泛型集合的命名空间
using System.Linq; // 导入 LINQ 查询操作符的命名空间
using Games.Common.IO; // 导入自定义的 Games.Common.IO 命名空间
using Games.Common.Randomness; // 导入自定义的 Games.Common.Randomness 命名空间
using Hexapawn.Resources; // 导入自定义的 Hexapawn.Resources 命名空间

namespace Hexapawn; // 定义 Hexapawn 命名空间

// Runs series of games between the computer and the human player
internal class GameSeries // 定义名为 GameSeries 的类
{
    private readonly TextIO _io; // 声明名为 _io 的只读 TextIO 类型字段
    private readonly Computer _computer; // 声明名为 _computer 的只读 Computer 类型字段
    private readonly Human _human; // 声明名为 _human 的只读 Human 类型字段
    private readonly Dictionary<object, int> _wins; // 声明名为 _wins 的只读 Dictionary<object, int> 类型字段

    public GameSeries(TextIO io, IRandom random) // 定义 GameSeries 类的构造函数，接受 TextIO 和 IRandom 类型的参数
    {
        _io = io; // 将传入的 io 参数赋值给 _io 字段
        _computer = new Computer(io, random); // 使用传入的 io 和 random 参数创建 Computer 类的实例，并赋值给 _computer 字段
        _human = new(io);  # 创建一个新的玩家对象，将其赋值给_human变量
        _wins = new() { [_computer] = 0, [_human] = 0 };  # 创建一个新的字典对象，初始化_computer和_human的胜利次数为0
    }

    public void Play()
    {
        _io.Write(Resource.Streams.Title);  # 使用_io对象输出游戏标题

        if (_io.GetYesNo("Instructions") == 'Y')  # 通过_io对象获取用户是否需要游戏说明
        {
            _io.Write(Resource.Streams.Instructions);  # 如果用户需要说明，则使用_io对象输出游戏说明
        }

        while (true)  # 进入游戏循环
        {
            var game = new Game(_io);  # 创建一个新的游戏对象，传入_io对象作为参数

            var winner = game.Play(_human, _computer);  # 调用游戏对象的Play方法，传入_human和_computer作为参数，获取游戏胜利者
            _wins[winner]++;  # 根据游戏胜利者在_wins字典中对应的键增加胜利次数
            _io.WriteLine(winner == _computer ? "I win." : "You win.");  # 根据游戏胜利者输出对应的胜利信息
            _io.Write($"I have won {_wins[_computer]} and you {_wins[_human]}");
            # 输出计算机和玩家的胜利次数
            _io.WriteLine($" out of {_wins.Values.Sum()} games.");
            # 输出总共进行的游戏次数
            _io.WriteLine();
            # 输出空行，用于分隔不同的输出
        }
    }
}
```