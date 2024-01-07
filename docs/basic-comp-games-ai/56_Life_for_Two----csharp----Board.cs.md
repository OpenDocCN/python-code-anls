# `basic-computer-games\56_Life_for_Two\csharp\Board.cs`

```

// 使用 System.Collections 和 System.Text 命名空间
using System.Collections;
using System.Text;

// LifeforTwo 命名空间下的 Board 类，实现了 IEnumerable 接口
namespace LifeforTwo
{
    internal class Board : IEnumerable<Coordinates>
    {
        // 7x7 的 Piece 数组
        private readonly Piece[,] _cells = new Piece[7, 7];
        // 用于统计不同 Piece 类型的数量的字典
        private readonly Dictionary<int, int> _cellCounts = 
            new() { [Piece.None] = 0, [Piece.Player1] = 0, [Piece.Player2] = 0 };

        // 索引器，通过坐标获取或设置 Piece
        public Piece this[Coordinates coordinates]
        {
            get => this[coordinates.X, coordinates.Y];
            set => this[coordinates.X, coordinates.Y] = value;
        }

        // 通过坐标获取或设置 Piece
        private Piece this[int x, int y]
        {
            get => _cells[x, y];
            set
            {
                // 如果原来的位置不为空，则对应类型数量减一
                if (!_cells[x, y].IsEmpty) { _cellCounts[_cells[x, y]] -= 1; }
                // 设置新的 Piece，并对应类型数量加一
                _cells[x, y] = value;
                _cellCounts[value] += 1;
            }
        }

        // 获取 Player1 类型的 Piece 数量
        public int Player1Count => _cellCounts[Piece.Player1];
        // 获取 Player2 类型的 Piece 数量
        public int Player2Count => _cellCounts[Piece.Player2];

        // 判断指定坐标是否为空
        internal bool IsEmptyAt(Coordinates coordinates) => this[coordinates].IsEmpty;

        // 清空指定坐标的 Piece
        internal void ClearCell(Coordinates coordinates) => this[coordinates] = Piece.NewNone();
        // 在指定坐标添加 Player1 类型的 Piece
        internal void AddPlayer1Piece(Coordinates coordinates) => this[coordinates] = Piece.NewPlayer1();
        // 在指定坐标添加 Player2 类型的 Piece
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

        // 获取指定坐标的显示字符串
        private string GetCellDisplay(int x, int y) =>
            (x, y) switch
            {
                (0 or 6, _) => $" {y % 6} ",
                (_, 0 or 6) => $" {x % 6} ",
                _ => $" {this[x, y]} "
            };

        // 实现 IEnumerable 接口的 GetEnumerator 方法
        public IEnumerator<Coordinates> GetEnumerator()
        {
            for (var x = 1; x <= 5; x++)
            {
                for (var y = 1; y <= 5; y++)
                {
                    yield return new Coordinates(x, y);
                }
            }
        }

        // 实现 IEnumerable 接口的 GetEnumerator 方法
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}

```