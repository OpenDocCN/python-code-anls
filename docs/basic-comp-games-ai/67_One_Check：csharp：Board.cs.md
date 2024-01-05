# `67_One_Check\csharp\Board.cs`

```
namespace OneCheck;

internal class Board
{
    private readonly bool[][] _checkers;  // 二维布尔数组，用于表示棋盘上的棋子位置
    private int _pieceCount;  // 棋子数量
    private int _moveCount;  // 移动次数

    public Board()
    {
        _checkers = 
            Enumerable.Range(0, 8)
                .Select(r => Enumerable.Range(0, 8)
                    .Select(c => r <= 1 || r >= 6 || c <= 1 || c >= 6).ToArray())
                .ToArray();  // 初始化棋盘布局，将边界位置和初始棋子位置设置为 true，其余位置设置为 false
        _pieceCount = 48;  // 初始化棋子数量为 48
    }

    private bool this[int index]  // 索引器，用于访问棋盘上的特定位置
    {
        get => _checkers[index / 8][index % 8];
        // 从二维数组_checkers中获取指定索引位置的值

        set => _checkers[index / 8][index % 8] = value;
        // 将指定索引位置的值设置为给定的value
    }

    public bool PlayMove(IReadWrite io)
    {
        while (true)
        {
            var from = (int)io.ReadNumber(Prompts.From);
            // 从输入流io中读取起始位置的数字

            if (from == 0) { return false; }
            // 如果起始位置为0，则返回false

            var move = new Move { From = from - 1, To = (int)io.ReadNumber(Prompts.To) - 1 };
            // 创建一个Move对象，设置起始位置和目标位置的值

            if (TryMove(move)) 
            { 
                _moveCount++;
                return true; 
            }
            // 如果尝试移动成功，则增加移动计数并返回true

            io.Write(Streams.IllegalMove);
            // 向输出流io写入非法移动的消息
    public bool TryMove(Move move)
    {
        // 检查移动是否在范围内，并且是否是斜向移动两个空格
        if (move.IsInRange && move.IsTwoSpacesDiagonally && IsPieceJumpingPieceToEmptySpace(move))
        {
            // 如果满足条件，更新棋盘状态
            this[move.From] = false;
            this[move.Jumped] = false;
            this[move.To] = true;
            _pieceCount--;  // 更新棋子数量
            return true;  // 返回移动成功
        }

        return false;  // 返回移动失败
    }

    // 检查棋子是否跳过其他棋子到空格上
    private bool IsPieceJumpingPieceToEmptySpace(Move move) => this[move.From] && this[move.Jumped] && !this[move.To];

    // 获取游戏报告
    public string GetReport() => string.Format(Formats.Results, _moveCount, _pieceCount);
# 将_checkers列表中的每个子列表转换为字符串，子列表中的布尔值True转换为" 1"，False转换为" 0"，然后用空格连接成一个字符串
# 然后将所有子列表连接成一个字符串，每个子列表用换行符分隔，最终返回这个字符串
```