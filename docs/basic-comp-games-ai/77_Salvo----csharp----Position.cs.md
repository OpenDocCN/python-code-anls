# `basic-computer-games\77_Salvo\csharp\Position.cs`

```py
// 命名空间 Salvo
namespace Salvo
{
    // 定义一个内部的结构体 Position，包含 X 和 Y 两个坐标
    internal record struct Position(Coordinate X, Coordinate Y)
    {
        // 判断位置是否在有效范围内
        public bool IsInRange => X.IsInRange && Y.IsInRange;
        // 判断位置是否在对角线上
        public bool IsOnDiagonal => X == Y;

        // 根据给定的坐标创建一个 Position 对象
        public static Position Create((float X, float Y) coordinates) => new(coordinates.X, coordinates.Y);

        // 尝试根据给定的坐标创建一个有效的 Position 对象
        public static bool TryCreateValid((float X, float Y) coordinates, out Position position)
        {
            // 如果 X 和 Y 均为有效坐标，则创建 Position 对象并返回 true
            if (Coordinate.TryCreateValid(coordinates.X, out var x) && Coordinate.TryCreateValid(coordinates.Y, out var y))
            {
                position = new(x, y);
                return true;
            }

            // 否则返回 false
            position = default;
            return false;
        }

        // 返回所有可能的 Position 对象
        public static IEnumerable<Position> All
            => Coordinate.Range.SelectMany(x => Coordinate.Range.Select(y => new Position(x, y)));

        // 返回邻居位置的集合
        public IEnumerable<Position> Neighbours
        {
            get
            {
                foreach (var offset in Offset.Units)
                {
                    var neighbour = this + offset;
                    if (neighbour.IsInRange) { yield return neighbour; }
                }
            }
        }

        // 计算到另一个 Position 对象的距离
        internal float DistanceTo(Position other)
        {
            var (deltaX, deltaY) = (X - other.X, Y - other.Y);
            return (float)Math.Sqrt(deltaX * deltaX + deltaY * deltaY);
        }

        // 将位置带入有效范围内
        internal Position BringIntoRange(IRandom random)
            => IsInRange ? this : new(X.BringIntoRange(random), Y.BringIntoRange(random));

        // 重载 + 运算符，实现 Position 对象的加法
        public static Position operator +(Position position, Offset offset) 
            => new(position.X + offset.X, position.Y + offset.Y);

        // 隐式转换，将整数转换为 Position 对象
        public static implicit operator Position(int value) => new(value, value);

        // 重写 ToString 方法，返回 Position 对象的字符串表示
        public override string ToString() => $"{X}{Y}";
    }
}
```