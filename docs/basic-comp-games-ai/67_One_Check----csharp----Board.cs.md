# `basic-computer-games\67_One_Check\csharp\Board.cs`

```

namespace OneCheck;

internal class Board
{
    private readonly bool[][] _checkers;  // 二维布尔数组，表示棋盘上的棋子分布
    private int _pieceCount;  // 棋子数量
    private int _moveCount;  // 移动次数

    public Board()
    {
        // 初始化棋盘，将边界和初始棋子位置设置为 true，其余位置设置为 false
        _checkers = 
            Enumerable.Range(0, 8)
                .Select(r => Enumerable.Range(0, 8)
                    .Select(c => r <= 1 || r >= 6 || c <= 1 || c >= 6).ToArray())
                .ToArray();
        _pieceCount = 48;  // 初始棋子数量为 48
    }

    private bool this[int index]
    {
        get => _checkers[index / 8][index % 8];  // 获取指定位置的棋子状态
        set => _checkers[index / 8][index % 8] = value;  // 设置指定位置的棋子状态
    }

    public bool PlayMove(IReadWrite io)
    {
        while (true)
        {
            var from = (int)io.ReadNumber(Prompts.From);  // 从输入流中读取起始位置
            if (from == 0) { return false; }  // 如果起始位置为 0，结束游戏

            var move = new Move { From = from - 1, To = (int)io.ReadNumber(Prompts.To) - 1 };  // 读取移动的起始和目标位置

            if (TryMove(move))  // 尝试移动棋子
            { 
                _moveCount++;  // 移动次数加一
                return true;  // 移动成功，返回 true
            }

            io.Write(Streams.IllegalMove);  // 移动非法，向输出流写入提示信息
        }
    }

    public bool TryMove(Move move)
    {
        if (move.IsInRange && move.IsTwoSpacesDiagonally && IsPieceJumpingPieceToEmptySpace(move))
        {
            this[move.From] = false;  // 起始位置的棋子移动后消失
            this[move.Jumped] = false;  // 被跳过的位置的棋子消失
            this[move.To] = true;  // 目标位置出现新的棋子
            _pieceCount--;  // 棋子数量减一
            return true;  // 移动成功，返回 true
        }

        return false;  // 移动失败，返回 false
    }

    private bool IsPieceJumpingPieceToEmptySpace(Move move) => this[move.From] && this[move.Jumped] && !this[move.To];  // 判断棋子是否从起始位置跳过其他棋子到空位置

    public string GetReport() => string.Format(Formats.Results, _moveCount, _pieceCount);  // 获取游戏报告信息

    public override string ToString() => 
        string.Join(Environment.NewLine, _checkers.Select(r => string.Join(" ", r.Select(c => c ? " 1" : " 0"))));  // 将棋盘状态转换为字符串形式
}

```