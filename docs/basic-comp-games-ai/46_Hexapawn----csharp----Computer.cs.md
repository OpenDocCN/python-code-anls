# `basic-computer-games\46_Hexapawn\csharp\Computer.cs`

```py
using System; // 导入系统命名空间
using System.Collections.Generic; // 导入集合类命名空间
using System.Linq; // 导入 LINQ 命名空间
using Games.Common.IO; // 导入自定义的 IO 类命名空间
using Games.Common.Randomness; // 导入自定义的随机数类命名空间
using static Hexapawn.Pawn; // 导入 Hexapawn.Pawn 类的静态成员

namespace Hexapawn; // 声明 Hexapawn 命名空间

/// <summary>
/// Encapsulates the logic of the computer player.
/// </summary>
internal class Computer // 声明名为 Computer 的内部类
{
    private readonly TextIO _io; // 声明私有的 TextIO 类型字段 _io
    private readonly IRandom _random; // 声明私有的 IRandom 类型字段 _random
    private readonly Dictionary<Board, List<Move>> _potentialMoves; // 声明私有的 Board 到 Move 列表的字典 _potentialMoves
    private (List<Move>, Move) _lastMove; // 声明私有的元组类型字段 _lastMove，包含一个 Move 列表和一个 Move 对象
    public Computer(TextIO io, IRandom random) // 声明公共的构造函数，接受 TextIO 和 IRandom 类型的参数
    }

    // Try to make a move. We first try to find a legal move for the current board position.
    public bool TryMove(Board board) // 声明公共的方法 TryMove，接受 Board 类型的参数 board
    {
        if (TryGetMoves(board, out var moves, out var reflected) && // 如果 TryGetMoves 方法返回 true，并且将 moves 和 reflected 赋值
            TrySelectMove(moves, out var move)) // 如果 TrySelectMove 方法返回 true，并且将 move 赋值
        {
            // We've found a move, so we record it as the last move made, and then announce and make the move.
            _lastMove = (moves, move); // 将 moves 和 move 组成的元组赋值给 _lastMove
            // If we found the move from a reflacted match of the board we need to make the reflected move.
            if (reflected) { move = move.Reflected; } // 如果 reflected 为 true，则将 move 的反射赋值给 move
            _io.WriteLine($"I move {move}"); // 输出移动信息
            move.Execute(board); // 执行移动操作
            return true; // 返回 true
        }
        // We haven't found a move for this board position, so remove the previous move that led to this board
        // position from future consideration. We don't want to make that move again, because we now know it's a
        // non-winning move.
        ExcludeLastMoveFromFuturePlay(); // 从未来的考虑中排除上一个导致当前棋盘位置的移动
        return false; // 返回 false
    }

    // Looks up the given board and its reflection in the potential moves dictionary. If it's found then we have a
    // list of potential moves. If the board is not found in the dictionary then the computer has no legal moves,
    // and the human player wins.
    private bool TryGetMoves(Board board, out List<Move> moves, out bool reflected) // 声明私有的方法 TryGetMoves，接受 Board 类型的参数 board，以及输出参数 moves 和 reflected
    {
        // 如果_potentialMoves字典中包含给定的board键，则将对应的值赋给moves，并返回true
        if (_potentialMoves.TryGetValue(board, out moves))
        {
            // 将reflected设置为false
            reflected = false;
            // 返回true
            return true;
        }
        // 如果_potentialMoves字典中包含给定board的反射键，则将对应的值赋给moves，并返回true
        if (_potentialMoves.TryGetValue(board.Reflected, out moves))
        {
            // 将reflected设置为true
            reflected = true;
            // 返回true
            return true;
        }
        // 将reflected设置为默认值
        reflected = default;
        // 返回false
        return false;
    }

    // 从moves列表中随机选择一个移动。如果列表为空，则说明之前已经排除了所有非获胜移动，因此放弃游戏。
    private bool TrySelectMove(List<Move> moves, out Move move)
    {
        // 如果moves列表不为空
        if (moves.Any())
        {
            // 从moves列表中随机选择一个移动
            move = moves[_random.Next(moves.Count)];
            // 返回true
            return true;
        }
        // 输出"I resign."
        _io.WriteLine("I resign.");
        // 将move设置为null
        move = null;
        // 返回false
        return false;
    }

    // 从未来的游戏中排除最后一个移动
    private void ExcludeLastMoveFromFuturePlay()
    {
        // 从_lastMove元组中获取moves和move
        var (moves, move) = _lastMove;
        // 从moves列表中移除move
        moves.Remove(move);
    }

    // 创建包含给定移动的列表
    private static List<Move> Moves(params Move[] moves) => moves.ToList();

    // 检查是否棋盘已经完全进阶
    public bool IsFullyAdvanced(Board board) =>
        board[9] == Black || board[8] == Black || board[7] == Black;
# 闭合前面的函数定义
```