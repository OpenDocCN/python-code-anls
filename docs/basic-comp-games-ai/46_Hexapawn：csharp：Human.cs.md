# `46_Hexapawn\csharp\Human.cs`

```
using System;  # 导入 System 模块
using System.Linq;  # 导入 System 模块中的 Linq 功能
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using static Hexapawn.Cell;  # 导入 Hexapawn.Cell 模块中的所有内容
using static Hexapawn.Move;  # 导入 Hexapawn.Move 模块中的所有内容
using static Hexapawn.Pawn;  # 导入 Hexapawn.Pawn 模块中的所有内容

namespace Hexapawn;  # 定义 Hexapawn 命名空间

internal class Human  # 定义 Human 类
{
    private readonly TextIO _io;  # 声明私有变量 _io，类型为 TextIO

    public Human(TextIO io)  # 定义 Human 类的构造函数，参数为 io
    {
        _io = io;  # 将传入的 io 参数赋值给私有变量 _io
    }

    public void Move(Board board)  # 定义 Move 方法，参数为 board
    {
        while (true)
        {
            // 从输入输出对象中读取玩家的移动
            var move = _io.ReadMove("Your move");

            // 尝试在棋盘上执行移动，如果成功则返回
            if (TryExecute(board, move)) { return; }

            // 如果移动非法，则输出提示信息
            _io.WriteLine("Illegal move.");
        }
    }

    // 检查是否存在合法的移动
    public bool HasLegalMove(Board board)
    {
        // 遍历所有大于3的单元格
        foreach (var from in AllCells.Where(c => c > 3))
        {
            // 如果该单元格上的棋子不是白色，则继续下一个单元格
            if (board[from] != White) { continue; }

            // 如果该单元格上的棋子是白色，并且存在合法的移动，则返回true
            if (HasLegalMove(board, from))
            {
                return true;
            }
    }

    return false;
}
```

这段代码是一个C#程序中的一些方法。下面是对每个方法的注释：

1. private bool HasLegalMove(Board board, Cell from) =>
   - 这是一个私有方法，用于检查给定的棋盘和起始位置是否有合法的移动。它返回一个布尔值。

2. public bool HasNoPawns(Board board) => board.All(c => c != White);
   - 这是一个公共方法，用于检查给定的棋盘上是否没有白色的棋子。它返回一个布尔值。

3. public bool TryExecute(Board board, Move move)
   - 这是一个公共方法，用于尝试在给定的棋盘上执行移动。它接受一个棋盘和一个移动作为参数，并返回一个布尔值。

在这段代码中，缺少了一些注释，因此无法准确地解释每个语句的作用。
            move.Execute(board);  # 调用移动对象的Execute方法，传入棋盘参数，执行移动操作
            return true;  # 返回true，表示移动操作成功
        }

        return false;  # 如果没有匹配的移动操作，返回false
    }
}
```