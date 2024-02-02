# `basic-computer-games\46_Hexapawn\csharp\Human.cs`

```py
// 引入必要的命名空间
using System;
using System.Linq;
using Games.Common.IO;
using static Hexapawn.Cell;
using static Hexapawn.Move;
using static Hexapawn.Pawn;

// 定义 Hexapawn 命名空间
namespace Hexapawn;

// 定义 Human 类
internal class Human
{
    // 声明私有字段 _io，用于处理文本输入输出
    private readonly TextIO _io;

    // Human 类的构造函数，接受 TextIO 对象作为参数
    public Human(TextIO io)
    {
        _io = io;
    }

    // 定义 Move 方法，用于处理玩家移动
    public void Move(Board board)
    {
        // 循环，直到玩家输入合法的移动
        while (true)
        {
            // 从玩家输入中读取移动
            var move = _io.ReadMove("Your move");

            // 尝试执行移动，如果成功则返回
            if (TryExecute(board, move)) { return; }

            // 如果移动非法，则输出提示信息
            _io.WriteLine("Illegal move.");
        }
    }

    // 判断玩家是否有合法的移动
    public bool HasLegalMove(Board board)
    {
        // 遍历所有白色棋子的位置
        foreach (var from in AllCells.Where(c => c > 3))
        {
            // 如果当前位置没有白色棋子，则继续下一个位置
            if (board[from] != White) { continue; }

            // 如果当前位置有合法的移动，则返回 true
            if (HasLegalMove(board, from))
            {
                return true;
            }
        }

        // 如果没有找到合法的移动，则返回 false
        return false;
    }

    // 判断特定位置的棋子是否有合法的移动
    private bool HasLegalMove(Board board, Cell from) =>
        Right(from).IsRightDiagonalToCapture(board) ||
        Straight(from).IsStraightMoveToEmptySpace(board) ||
        from > 4 && Left(from).IsLeftDiagonalToCapture(board);

    // 判断棋盘上是否没有白色棋子
    public bool HasNoPawns(Board board) => board.All(c => c != White);

    // 尝试执行玩家的移动
    public bool TryExecute(Board board, Move move)
    {
        // 如果移动的起始位置没有白色棋子，则返回 false
        if (board[move.From] != White) { return false; }

        // 如果移动是直线移动到空位、左斜线吃子或右斜线吃子，则执行移动并返回 true
        if (move.IsStraightMoveToEmptySpace(board) ||
            move.IsLeftDiagonalToCapture(board) ||
            move.IsRightDiagonalToCapture(board))
        {
            move.Execute(board);
            return true;
        }

        // 如果移动非法，则返回 false
        return false;
    }
}
```