# `56_Life_for_Two\csharp\Board.cs`

```
    // 使用 System.Collections 命名空间
    using System.Collections;
    // 使用 System.Text 命名空间
    using System.Text;

    // 声明 LifeforTwo 命名空间
    namespace LifeforTwo
    {
        // 声明 Board 类并实现 IEnumerable<Coordinates> 接口
        internal class Board : IEnumerable<Coordinates>
        {
            // 声明私有的 Piece 类型的二维数组 _cells，大小为 7x7
            private readonly Piece[,] _cells = new Piece[7, 7];
            // 声明私有的 Dictionary<int, int> 类型的 _cellCounts 字段，并初始化
            private readonly Dictionary<int, int> _cellCounts = 
                new() { [Piece.None] = 0, [Piece.Player1] = 0, [Piece.Player2] = 0 };

            // 声明索引器，根据坐标获取或设置 Piece 类型的值
            public Piece this[Coordinates coordinates]
            {
                get => this[coordinates.X, coordinates.Y];
                set => this[coordinates.X, coordinates.Y] = value;
            }

            // 声明私有的索引器，根据坐标获取 Piece 类型的值
            private Piece this[int x, int y]
            {
                get => _cells[x, y];
        set
        {
            // 如果要设置的位置不为空，则将原来位置的计数减一
            if (!_cells[x, y].IsEmpty) { _cellCounts[_cells[x, y]] -= 1; }
            // 设置新的位置
            _cells[x, y] = value;
            // 将新位置的计数加一
            _cellCounts[value] += 1;
        }
    }

    // 返回玩家1的棋子数量
    public int Player1Count => _cellCounts[Piece.Player1];
    // 返回玩家2的棋子数量
    public int Player2Count => _cellCounts[Piece.Player2];

    // 判断给定坐标的位置是否为空
    internal bool IsEmptyAt(Coordinates coordinates) => this[coordinates].IsEmpty;

    // 清空给定坐标的位置
    internal void ClearCell(Coordinates coordinates) => this[coordinates] = Piece.NewNone();
    // 在给定坐标添加玩家1的棋子
    internal void AddPlayer1Piece(Coordinates coordinates) => this[coordinates] = Piece.NewPlayer1();
    // 在给定坐标添加玩家2的棋子
    internal void AddPlayer2Piece(Coordinates coordinates) => this[coordinates] = Piece.NewPlayer2();

    // 重写 ToString 方法
    public override string ToString()
    {
        // 创建一个 StringBuilder 对象
        var builder = new StringBuilder();
        for (var y = 0; y <= 6; y++)
        {
            // 在每一行末尾添加换行符
            builder.AppendLine();
            for (var x = 0; x <= 6; x++)
            {
                // 获取指定位置单元格的显示内容并添加到字符串构建器中
                builder.Append(GetCellDisplay(x, y));
            }
        }

        // 将字符串构建器中的内容转换为字符串并返回
        return builder.ToString();
    }

    // 根据指定的坐标获取单元格的显示内容
    private string GetCellDisplay(int x, int y) =>
        // 使用模式匹配根据坐标位置返回对应的显示内容
        (x, y) switch
        {
            (0 or 6, _) => $" {y % 6} ",  // 如果 x 为 0 或 6，则显示 y 对 6 取余的结果
            (_, 0 or 6) => $" {x % 6} ",  // 如果 y 为 0 或 6，则显示 x 对 6 取余的结果
            _ => $" {this[x, y]} "  // 其他情况显示指定位置单元格的内容
        };
# 定义一个公共的迭代器方法，用于返回坐标对象的集合
public IEnumerator<Coordinates> GetEnumerator()
{
    # 使用嵌套循环遍历 x 和 y 坐标的可能取值
    for (var x = 1; x <= 5; x++)
    {
        for (var y = 1; y <= 5; y++)
        {
            # 使用 yield 关键字返回一个新的坐标对象
            yield return new(x, y);
        }
    }
}

# 实现显式接口成员，返回迭代器方法
IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
```
在这段代码中，定义了一个公共的迭代器方法 `GetEnumerator`，用于返回坐标对象的集合。在方法内部使用了嵌套循环来遍历 x 和 y 坐标的可能取值，并使用 `yield` 关键字返回一个新的坐标对象。同时，还实现了显式接口成员 `IEnumerable.GetEnumerator`，并在其中返回了迭代器方法 `GetEnumerator`。
```