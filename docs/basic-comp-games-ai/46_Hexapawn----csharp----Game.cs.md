# `basic-computer-games\46_Hexapawn\csharp\Game.cs`

```py
// 引入命名空间 System 和 Games.Common.IO
using System;
using Games.Common.IO;

// 声明命名空间 Hexapawn
namespace Hexapawn;

// 定义一个名为 Game 的内部类
internal class Game
{
    // 声明私有只读字段 _io，类型为 TextIO
    private readonly TextIO _io;
    // 声明私有只读字段 _board，类型为 Board
    private readonly Board _board;

    // 定义构造函数，接受一个 TextIO 类型的参数 io
    public Game(TextIO io)
    {
        // 实例化 Board 类，赋值给 _board 字段
        _board = new Board();
        // 将传入的 io 参数赋值给 _io 字段
        _io = io;
    }

    // 定义 Play 方法，接受一个 Human 类型的参数 human 和一个 Computer 类型的参数 computer
    public object Play(Human human, Computer computer)
    {
        // 在控制台输出当前棋盘状态
        _io.WriteLine(_board);
        // 进入游戏循环
        while(true)
        {
            // 人类玩家进行移动
            human.Move(_board);
            // 在控制台输出当前棋盘状态
            _io.WriteLine(_board);
            // 如果计算机无法移动，则返回人类玩家
            if (!computer.TryMove(_board))
            {
                return human;
            }
            // 在控制台输出当前棋盘状态
            _io.WriteLine(_board);
            // 如果计算机已经完全晋级或者人类玩家没有棋子了，则返回计算机玩家
            if (computer.IsFullyAdvanced(_board) || human.HasNoPawns(_board))
            {
                return computer;
            }
            // 如果人类玩家没有合法移动，则在控制台输出提示信息，并返回计算机玩家
            if (!human.HasLegalMove(_board))
            {
                _io.Write("You can't move, so ");
                return computer;
            }
        }
    }
}
```