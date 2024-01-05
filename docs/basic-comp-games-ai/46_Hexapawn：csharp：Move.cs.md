# `d:/src/tocomm/basic-computer-games\46_Hexapawn\csharp\Move.cs`

```
using static Hexapawn.Pawn;  # 导入 Hexapawn.Pawn 命名空间中的所有静态成员

namespace Hexapawn;  # 声明 Hexapawn 命名空间

/// <summary>
/// Represents a move which may, or may not, be legal.
/// </summary>
internal class Move  # 定义一个名为 Move 的类，表示一个可能合法或不合法的移动
{
    private readonly Cell _from;  # 声明一个名为 _from 的只读私有成员变量，表示移动的起始位置
    private readonly Cell _to;  # 声明一个名为 _to 的只读私有成员变量，表示移动的目标位置
    private readonly int _metric;  # 声明一个名为 _metric 的只读私有成员变量，表示移动的度量值

    public Move(Cell from, Cell to)  # 定义一个名为 Move 的公共构造函数，接受起始位置和目标位置作为参数
    {
        _from = from;  # 将参数 from 赋值给 _from
        _to = to;  # 将参数 to 赋值给 _to
        _metric = _from - _to;  # 计算起始位置和目标位置的差值，并赋值给 _metric
    }
    // 将当前移动的镜像产生在棋盘的中心列周围
    public Move Reflected => (_from.Reflected, _to.Reflected);

    // 允许将两个整数的元组隐式转换为 Move 对象
    public static implicit operator Move((int From, int To) value) => new(value.From, value.To);

    // 接受浮点坐标，尝试创建一个 Move 对象
    public static bool TryCreate(float input1, float input2, out Move move)
    {
        // 如果可以创建起始位置和目标位置的 Cell 对象
        if (Cell.TryCreate(input1, out var from) &&
            Cell.TryCreate(input2, out var to))
        {
            move = (from, to);  # 将移动起始位置和目标位置存储在变量move中
            return true;  # 返回true，表示移动成功
        }

        move = default;  # 将move重置为默认值
        return false;  # 返回false，表示移动失败
    }

    public static Move Right(Cell from) => (from, from - 2);  # 定义向右移动的方法，并返回移动的起始位置和目标位置
    public static Move Straight(Cell from) => (from, from - 3);  # 定义向直线移动的方法，并返回移动的起始位置和目标位置
    public static Move Left(Cell from) => (from, from - 4);  # 定义向左移动的方法，并返回移动的起始位置和目标位置

    public bool IsStraightMoveToEmptySpace(Board board) => _metric == 3 && board[_to] == None;  # 检查是否可以向空格直线移动

    public bool IsLeftDiagonalToCapture(Board board) => _metric == 4 && _from != 7 && board[_to] == Black;  # 检查是否可以向左对角线移动并捕获对方棋子

    public bool IsRightDiagonalToCapture(Board board) =>
        _metric == 2 && _from != 9 && _from != 6 && board[_to] == Black;  # 检查是否可以向右对角线移动并捕获对方棋子

    public void Execute(Board board)  # 执行移动操作
    {
        board[_to] = board[_from];  // 将目标位置的棋子设置为源位置的棋子
        board[_from] = None;  // 将源位置的棋子设置为None，表示该位置没有棋子
    }

    public override string ToString() => $"from {_from} to {_to}";  // 重写ToString方法，返回移动的源位置和目标位置
}
```