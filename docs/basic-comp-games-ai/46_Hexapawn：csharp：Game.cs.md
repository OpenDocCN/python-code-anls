# `d:/src/tocomm/basic-computer-games\46_Hexapawn\csharp\Game.cs`

```
using System;  # 导入系统模块
using Games.Common.IO;  # 导入游戏通用输入输出模块

namespace Hexapawn;  # 命名空间为 Hexapawn

// 一个 Hexapawn 游戏
internal class Game  # 内部类 Game
{
    private readonly TextIO _io;  # 只读的文本输入输出对象 _io
    private readonly Board _board;  # 只读的棋盘对象 _board

    public Game(TextIO io)  # Game 类的构造函数，接受一个 TextIO 对象 io 作为参数
    {
        _board = new Board();  # 创建一个新的棋盘对象并赋值给 _board
        _io = io;  # 将传入的 io 对象赋值给 _io
    }

    public object Play(Human human, Computer computer)  # Play 方法，接受一个 Human 对象和一个 Computer 对象作为参数
    {
        _io.WriteLine(_board);  # 使用 _io 对象输出 _board 的内容
        while(true)  # 创建一个无限循环，直到满足某个条件才会退出循环
        {
            human.Move(_board);  # 人类玩家进行移动操作
            _io.WriteLine(_board);  # 输出当前棋盘状态
            if (!computer.TryMove(_board))  # 如果电脑玩家无法移动
            {
                return human;  # 返回人类玩家为胜利者
            }
            _io.WriteLine(_board);  # 输出当前棋盘状态
            if (computer.IsFullyAdvanced(_board) || human.HasNoPawns(_board))  # 如果电脑玩家已经完全晋级或者人类玩家没有棋子了
            {
                return computer;  # 返回电脑玩家为胜利者
            }
            if (!human.HasLegalMove(_board))  # 如果人类玩家没有合法的移动
            {
                _io.Write("You can't move, so ");  # 输出提示信息
                return computer;  # 返回电脑玩家为胜利者
            }
        }
    }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```