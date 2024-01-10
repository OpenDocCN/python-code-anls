# `basic-computer-games\56_Life_for_Two\csharp\Board.cs`

```
{
    // 命名空间 LifeforTwo 中定义了 Board 类，实现了 IEnumerable 接口
    internal class Board : IEnumerable<Coordinates>
    {
        // 7x7 的 Piece 数组，表示游戏棋盘
        private readonly Piece[,] _cells = new Piece[7, 7];
        // 用于统计不同类型 Piece 的数量的字典
        private readonly Dictionary<int, int> _cellCounts = 
            new() { [Piece.None] = 0, [Piece.Player1] = 0, [Piece.Player2] = 0 };

        // 索引器，通过坐标获取或设置对应位置的 Piece
        public Piece this[Coordinates coordinates]
        {
            get => this[coordinates.X, coordinates.Y];
            set => this[coordinates.X, coordinates.Y] = value;
        }

        // 私有索引器，通过坐标获取或设置对应位置的 Piece
        private Piece this[int x, int y]
        {
            get => _cells[x, y];
            set
            {
                // 如果原位置不为空，则更新对应类型 Piece 的数量
                if (!_cells[x, y].IsEmpty) { _cellCounts[_cells[x, y]] -= 1; }
                _cells[x, y] = value;
                _cellCounts[value] += 1;
            }
        }

        // 获取 Player1 类型 Piece 的数量
        public int Player1Count => _cellCounts[Piece.Player1];
        // 获取 Player2 类型 Piece 的数量
        public int Player2Count => _cellCounts[Piece.Player2];

        // 判断指定坐标位置是否为空
        internal bool IsEmptyAt(Coordinates coordinates) => this[coordinates].IsEmpty;

        // 清空指定坐标位置的 Piece
        internal void ClearCell(Coordinates coordinates) => this[coordinates] = Piece.NewNone();
        // 在指定坐标位置添加 Player1 类型的 Piece
        internal void AddPlayer1Piece(Coordinates coordinates) => this[coordinates] = Piece.NewPlayer1();
        // 在指定坐标位置添加 Player2 类型的 Piece
        internal void AddPlayer2Piece(Coordinates coordinates) => this[coordinates] = Piece.NewPlayer2();

        // 重写 ToString 方法，返回棋盘的字符串表示
        public override string ToString()
        {
            var builder = new StringBuilder();

            for (var y = 0; y <= 6; y++)
            {
                builder.AppendLine();
                for (var x = 0; x <= 6; x++)
                {
                    builder.Append(GetCellDisplay(x, y));
                }
            }

            return builder.ToString();
        }

        // 根据坐标位置返回对应的棋盘显示字符
        private string GetCellDisplay(int x, int y) =>
            (x, y) switch
            {
                (0 or 6, _) => $" {y % 6} ",  // 第一列和最后一列显示行号
                (_, 0 or 6) => $" {x % 6} ",  // 第一行和最后一行显示列号
                _ => $" {this[x, y]} "        // 其他位置显示对应的 Piece
            };

        // 实现 IEnumerable 接口的 GetEnumerator 方法
        public IEnumerator<Coordinates> GetEnumerator()
    # 创建一个循环，x从1到5，y从1到5，用于生成坐标对
    {
        for (var x = 1; x <= 5; x++)
        {
            for (var y = 1; y <= 5; y++)
            {
                # 使用 yield 返回一个新的坐标对
                yield return new(x, y);
            }
        }
    }

    # 实现 IEnumerable 接口的 GetEnumerator 方法
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
# 闭合前面的函数定义
```