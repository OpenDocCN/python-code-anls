# `basic-computer-games\67_One_Check\csharp\Board.cs`

```
namespace OneCheck;

internal class Board
{
    private readonly bool[][] _checkers;  // 二维数组，表示棋盘上每个位置是否有棋子
    private int _pieceCount;  // 棋子数量
    private int _moveCount;  // 移动次数

    public Board()
    {
        _checkers = 
            Enumerable.Range(0, 8)
                .Select(r => Enumerable.Range(0, 8)
                    .Select(c => r <= 1 || r >= 6 || c <= 1 || c >= 6).ToArray())
                .ToArray();  // 初始化棋盘，设置边界和初始棋子位置
        _pieceCount = 48;  // 初始棋子数量
    }

    private bool this[int index]  // 索引器，用于访问棋盘上的位置
    {
        get => _checkers[index / 8][index % 8];  // 获取指定位置的棋子状态
        set => _checkers[index / 8][index % 8] = value;  // 设置指定位置的棋子状态
    }

    public bool PlayMove(IReadWrite io)  // 玩家进行移动
    {
        while (true)
        {
            var from = (int)io.ReadNumber(Prompts.From);  // 读取玩家输入的起始位置
            if (from == 0) { return false; }  // 如果输入为0，结束游戏

            var move = new Move { From = from - 1, To = (int)io.ReadNumber(Prompts.To) - 1 };  // 读取玩家输入的移动信息

            if (TryMove(move))  // 尝试进行移动
            { 
                _moveCount++;  // 移动次数加一
                return true;  // 移动成功，返回true
            }

            io.Write(Streams.IllegalMove);  // 移动非法，提示玩家重新输入
        }
    }

    public bool TryMove(Move move)  // 尝试进行移动
    {
        if (move.IsInRange && move.IsTwoSpacesDiagonally && IsPieceJumpingPieceToEmptySpace(move))  // 判断移动是否合法
        {
            this[move.From] = false;  // 起始位置棋子移除
            this[move.Jumped] = false;  // 被跳过的位置棋子移除
            this[move.To] = true;  // 目标位置放置棋子
            _pieceCount--;  // 棋子数量减一
            return true;  // 移动成功，返回true
        }

        return false;  // 移动失败，返回false
    }

    private bool IsPieceJumpingPieceToEmptySpace(Move move) => this[move.From] && this[move.Jumped] && !this[move.To];  // 判断是否符合棋子跳跃规则

    public string GetReport() => string.Format(Formats.Results, _moveCount, _pieceCount);  // 获取游戏结果报告

    public override string ToString() => 
        string.Join(Environment.NewLine, _checkers.Select(r => string.Join(" ", r.Select(c => c ? " 1" : " 0"))));  // 将棋盘状态转换为字符串
}
```