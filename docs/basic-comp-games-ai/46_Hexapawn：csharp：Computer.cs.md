# `46_Hexapawn\csharp\Computer.cs`

```
using System;  // 导入 System 命名空间，包含常用的数据类型和基本类库
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间，包含泛型集合类
using System.Linq;  // 导入 System.Linq 命名空间，包含 LINQ 查询功能
using Games.Common.IO;  // 导入 Games.Common.IO 命名空间，包含游戏常用的输入输出功能
using Games.Common.Randomness;  // 导入 Games.Common.Randomness 命名空间，包含游戏常用的随机数生成功能
using static Hexapawn.Pawn;  // 导入 Hexapawn.Pawn 类的静态成员

namespace Hexapawn;  // 声明 Hexapawn 命名空间

/// <summary>
/// Encapsulates the logic of the computer player.
/// </summary>
internal class Computer  // 声明名为 Computer 的内部类
{
    private readonly TextIO _io;  // 声明私有的只读字段 _io，类型为 TextIO
    private readonly IRandom _random;  // 声明私有的只读字段 _random，类型为 IRandom
    private readonly Dictionary<Board, List<Move>> _potentialMoves;  // 声明私有的只读字段 _potentialMoves，类型为 Dictionary<Board, List<Move>>
    private (List<Move>, Move) _lastMove;  // 声明私有的字段 _lastMove，类型为元组 (List<Move>, Move)
    public Computer(TextIO io, IRandom random)  // 声明公共的构造函数 Computer，接受 TextIO 和 IRandom 类型的参数
    {
        _io = io;  // 将 io 模块赋值给变量 _io
        _random = random;  // 将 random 模块赋值给变量 _random

        // 这个字典实现了原始代码中的数据，它编码了计算机可以进行合法移动的棋盘位置，以及每个位置可能移动的列表：
        //   900 DATA -1,-1,-1,1,0,0,0,1,1,-1,-1,-1,0,1,0,1,0,1
        //   905 DATA -1,0,-1,-1,1,0,0,0,1,0,-1,-1,1,-1,0,0,0,1
        //   910 DATA -1,0,-1,1,1,0,0,1,0,-1,-1,0,1,0,1,0,0,1
        //   915 DATA 0,-1,-1,0,-1,1,1,0,0,0,-1,-1,-1,1,1,1,0,0
        //   920 DATA -1,0,-1,-1,0,1,0,1,0,0,-1,-1,0,1,0,0,0,1
        //   925 DATA 0,-1,-1,0,1,0,1,0,0,-1,0,-1,1,0,0,0,0,1
        //   930 DATA 0,0,-1,-1,-1,1,0,0,0,-1,0,0,1,1,1,0,0,0
        //   935 DATA 0,-1,0,-1,1,1,0,0,0,-1,0,0,-1,-1,1,0,0,0
        //   940 DATA 0,0,-1,-1,1,0,0,0,0,0,-1,0,1,-1,0,0,0,0
        //   945 DATA -1,0,0,-1,1,0,0,0,0
        //   950 DATA 24,25,36,0,14,15,36,0,15,35,36,47,36,58,59,0
        //   955 DATA 15,35,36,0,24,25,26,0,26,57,58,0
        //   960 DATA 26,35,0,0,47,48,0,0,35,36,0,0,35,36,0,0
        //   965 DATA 36,0,0,0,47,58,0,0,15,0,0,0
        //   970 DATA 26,47,0,0,47,58,0,0,35,36,47,0,28,58,0,0,15,47,0,0
        // 创建一个空的字典_potentialMoves，用于存储棋盘状态和对应的可行走步
        _potentialMoves = new()
        {
            // 以特定的棋盘状态为键，对应的可行走步为值，存储在_potentialMoves字典中
            [new(Black, Black, Black, White, None,  None,  None,  White, White)] = Moves((2, 4), (2, 5), (3, 6)),
            [new(Black, Black, Black, None,  White, None,  White, None,  White)] = Moves((1, 4), (1, 5), (3, 6)),
            [new(Black, None,  Black, Black, White, None,  None,  None,  White)] = Moves((1, 5), (3, 5), (3, 6), (4, 7)),
            [new(None,  Black, Black, White, Black, None,  None,  None,  White)] = Moves((3, 6), (5, 8), (5, 9)),
            [new(Black, None,  Black, White, White, None,  None,  White, None)]  = Moves((1, 5), (3, 5), (3, 6)),
            [new(Black, Black, None,  White, None,  White, None,  None,  White)] = Moves((2, 4), (2, 5), (2, 6)),
            [new(None,  Black, Black, None,  Black, White, White, None,  None)]  = Moves((2, 6), (5, 7), (5, 8)),
            [new(None,  Black, Black, Black, White, White, White, None,  None)]  = Moves((2, 6), (3, 5)),
            [new(Black, None,  Black, Black, None,  White, None,  White, None)]  = Moves((4, 7), (4, 8)),
        }
            [new(None,  Black, Black, None,  White, None,  None,  None,  White)] = Moves((3, 5), (3, 6)),
            [new(None,  Black, Black, None,  White, None,  White, None,  None)]  = Moves((3, 5), (3, 6)),
            [new(Black, None,  Black, White, None,  None,  None,  None,  White)] = Moves((3, 6)),
            [new(None,  None,  Black, Black, Black, White, None,  None,  None)]  = Moves((4, 7), (5, 8)),
            [new(Black, None,  None,  White, White, White, None,  None,  None)]  = Moves((1, 5)),
            [new(None,  Black, None,  Black, White, White, None,  None,  None)]  = Moves((2, 6), (4, 7)),
            [new(Black, None,  None,  Black, Black, White, None,  None,  None)]  = Moves((4, 7), (5, 8)),
            [new(None,  None,  Black, Black, White, None,  None,  None,  None)]  = Moves((3, 5), (3, 6), (4, 7)),
            [new(None,  Black, None,  White, Black, None,  None,  None,  None)]  = Moves((2, 8), (5, 8)),
            [new(Black, None,  None,  Black, White, None,  None,  None,  None)]  = Moves((1, 5), (4, 7))
        };
    }
```
这部分代码是一个C#语言中的字典初始化，使用了键值对的形式来初始化一个字典。

```
    // Try to make a move. We first try to find a legal move for the current board position.
    public bool TryMove(Board board)
    {
        if (TryGetMoves(board, out var moves, out var reflected) &&
            TrySelectMove(moves, out var move))
        {
            // We've found a move, so we record it as the last move made, and then announce and make the move.
```
这部分代码是一个C#语言中的函数定义，注释解释了函数的作用。在函数内部，首先调用了TryGetMoves和TrySelectMove函数，然后根据返回值进行了相应的操作。
            _lastMove = (moves, move);  // 将moves和move存储在_lastMove中，用于记录上一次的移动
            // 如果我们从棋盘的反射匹配中找到了移动，我们需要执行反射移动。
            if (reflected) { move = move.Reflected; }  // 如果reflected为true，则执行反射移动
            _io.WriteLine($"I move {move}");  // 输出计算机的移动
            move.Execute(board);  // 执行移动
            return true;  // 返回true表示找到了合法的移动
        }
        // 我们没有找到这个棋盘位置的移动，因此从未来的考虑中移除导致这个棋盘位置的上一个移动。
        // 我们不想再次执行该移动，因为现在我们知道它不是一个获胜的移动。
        ExcludeLastMoveFromFuturePlay();  // 从未来的考虑中排除上一个移动
        return false;  // 返回false表示没有找到合法的移动
    }

    // 在潜在移动字典中查找给定棋盘及其反射的移动。如果找到，则有一系列潜在的移动。
    // 如果在字典中找不到棋盘，则计算机没有合法的移动，人类玩家获胜。
    private bool TryGetMoves(Board board, out List<Move> moves, out bool reflected)
    {
        if (_potentialMoves.TryGetValue(board, out moves))  // 尝试从潜在移动字典中获取给定棋盘的移动列表
        {
            reflected = false;  # 初始化变量reflected为false
            return true;  # 返回true
        }
        if (_potentialMoves.TryGetValue(board.Reflected, out moves))  # 如果_potentialMoves字典中包含board.Reflected对应的值，并将其赋给moves
        {
            reflected = true;  # 将reflected设置为true
            return true;  # 返回true
        }
        reflected = default;  # 将reflected设置为默认值
        return false;  # 返回false
    }

    // 从列表中获取一个随机移动。如果列表为空，则我们之前已经排除了所有非获胜移动，因此我们认输比赛。
    private bool TrySelectMove(List<Move> moves, out Move move)  # 尝试从列表中选择一个移动，并将其赋给move
    {
        if (moves.Any())  # 如果moves列表不为空
        {
            move = moves[_random.Next(moves.Count)];  # 从moves列表中随机选择一个移动，并将其赋给move
            return true;  # 返回 true，表示条件满足
        }
        _io.WriteLine("I resign.");  # 输出字符串 "I resign." 到控制台
        move = null;  # 将 move 变量赋值为 null
        return false;  # 返回 false，表示条件不满足
    }

    private void ExcludeLastMoveFromFuturePlay()  # 定义一个名为 ExcludeLastMoveFromFuturePlay 的私有方法
    {
        var (moves, move) = _lastMove;  # 从 _lastMove 中获取 moves 和 move 变量
        moves.Remove(move);  # 从 moves 列表中移除 move 变量
    }

    private static List<Move> Moves(params Move[] moves) => moves.ToList();  # 定义一个静态方法 Moves，接受一个或多个 Move 类型的参数，并将它们转换为列表返回

    public bool IsFullyAdvanced(Board board) =>  # 定义一个名为 IsFullyAdvanced 的公有方法，接受一个 Board 类型的参数，并返回一个布尔值
        board[9] == Black || board[8] == Black || board[7] == Black;  # 检查 board 中索引为 9、8、7 的位置是否包含 Black 颜色
}
```