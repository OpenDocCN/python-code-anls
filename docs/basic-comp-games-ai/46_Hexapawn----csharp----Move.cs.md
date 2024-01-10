# `basic-computer-games\46_Hexapawn\csharp\Move.cs`

```
// 使用 Hexapawn.Pawn 命名空间下的静态成员
using static Hexapawn.Pawn;

// 声明 Hexapawn 命名空间
namespace Hexapawn;

/// <summary>
/// 表示一个可能合法的移动
/// </summary>
internal class Move
{
    // 私有字段，表示起始位置
    private readonly Cell _from;
    // 私有字段，表示目标位置
    private readonly Cell _to;
    // 私有字段，表示度量值
    private readonly int _metric;

    // 构造函数，接受起始位置和目标位置
    public Move(Cell from, Cell to)
    {
        _from = from;
        _to = to;
        // 计算度量值
        _metric = _from - _to;
    }

    // 解构函数，用于解构 Move 对象
    public void Deconstruct(out Cell from, out Cell to)
    {
        from = _from;
        to = _to;
    }

    // 属性，返回起始位置
    public Cell From => _from;

    // 返回当前移动的镜像，以棋盘中心列为对称轴
    public Move Reflected => (_from.Reflected, _to.Reflected);

    // 允许将包含两个整数的元组隐式转换为 Move 对象
    public static implicit operator Move((int From, int To) value) => new(value.From, value.To);

    // 尝试根据浮点坐标创建 Move 对象
    public static bool TryCreate(float input1, float input2, out Move move)
    {
        if (Cell.TryCreate(input1, out var from) &&
            Cell.TryCreate(input2, out var to))
        {
            move = (from, to);
            return true;
        }

        move = default;
        return false;
    }

    // 静态方法，返回向右移动的 Move 对象
    public static Move Right(Cell from) => (from, from - 2);
    // 静态方法，返回向前移动的 Move 对象
    public static Move Straight(Cell from) => (from, from - 3);
    // 静态方法，返回向左移动的 Move 对象
    public static Move Left(Cell from) => (from, from - 4);

    // 判断是否是向前移动到空位
    public bool IsStraightMoveToEmptySpace(Board board) => _metric == 3 && board[_to] == None;

    // 判断是否是向左对角线移动并进行吃子
    public bool IsLeftDiagonalToCapture(Board board) => _metric == 4 && _from != 7 && board[_to] == Black;

    // 判断是否是向右对角线移动并进行吃子
    public bool IsRightDiagonalToCapture(Board board) =>
        _metric == 2 && _from != 9 && _from != 6 && board[_to] == Black;

    // 执行移动
    public void Execute(Board board)
    {
        board[_to] = board[_from];
        board[_from] = None;
    }

    // 重写 ToString 方法，返回移动的起始位置和目标位置
    public override string ToString() => $"from {_from} to {_to}";
}
```