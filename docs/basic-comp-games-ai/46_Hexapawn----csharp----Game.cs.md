# `basic-computer-games\46_Hexapawn\csharp\Game.cs`

```

using System; // 导入 System 命名空间
using Games.Common.IO; // 导入 Games.Common.IO 命名空间

namespace Hexapawn; // 定义 Hexapawn 命名空间

// 一个 Hexapawn 游戏
internal class Game // 定义 Game 类
{
    private readonly TextIO _io; // 声明私有的 TextIO 类型变量 _io
    private readonly Board _board; // 声明私有的 Board 类型变量 _board

    public Game(TextIO io) // Game 类的构造函数，接受一个 TextIO 类型的参数 io
    {
        _board = new Board(); // 实例化 Board 类并赋值给 _board
        _io = io; // 将传入的 io 参数赋值给 _io
    }

    public object Play(Human human, Computer computer) // Play 方法，接受一个 Human 类型参数和一个 Computer 类型参数
    {
        _io.WriteLine(_board); // 调用 _io 的 WriteLine 方法，输出 _board 的内容
        while(true) // 进入无限循环
        {
            human.Move(_board); // 调用 human 的 Move 方法，传入 _board 参数
            _io.WriteLine(_board); // 再次输出 _board 的内容
            if (!computer.TryMove(_board)) // 如果 computer 的 TryMove 方法返回 false
            {
                return human; // 返回 human
            }
            _io.WriteLine(_board); // 再次输出 _board 的内容
            if (computer.IsFullyAdvanced(_board) || human.HasNoPawns(_board)) // 如果 computer 已经完全晋升或者 human 没有棋子了
            {
                return computer; // 返回 computer
            }
            if (!human.HasLegalMove(_board)) // 如果 human 没有合法的移动
            {
                _io.Write("You can't move, so "); // 输出提示信息
                return computer; // 返回 computer
            }
        }
    }
}

```