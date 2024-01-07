# `basic-computer-games\84_Super_Star_Trek\csharp\Space\Coordinates.cs`

```

// 引入命名空间 System 和 SuperStarTrek.Utils
using System;
using SuperStarTrek.Utils;

// 声明 SuperStarTrek.Space 命名空间
namespace SuperStarTrek.Space
{
    // 表示星系中一个象限的坐标，或者一个象限中的一个区域的坐标
    // 注意原点在左上角，x 增大向下，y 增大向右
    internal record Coordinates
    {
        // 构造函数，初始化坐标的 x 和 y 值
        internal Coordinates(int x, int y)
        {
            // 使用 Validated 方法验证 x 和 y 的值
            X = Validated(x, nameof(x));
            Y = Validated(y, nameof(y));

            // 计算区域索引和子区域索引
            RegionIndex = (X << 1) + (Y >> 2);
            SubRegionIndex = Y % 4;
        }

        // 获取 x 坐标
        internal int X { get; }

        // 获取 y 坐标
        internal int Y { get; }

        // 获取区域索引
        internal int RegionIndex { get; }

        // 获取子区域索引
        internal int SubRegionIndex { get; }

        // 静态方法，验证值是否在 0 到 7 之间
        private static int Validated(int value, string argumentName)
        {
            if (value >= 0 && value <= 7) { return value; }

            throw new ArgumentOutOfRangeException(argumentName, value, "Must be 0 to 7 inclusive");
        }

        // 静态方法，验证值是否在 0 到 7 之间
        private static bool IsValid(int value) => value >= 0 && value <= 7;

        // 重写 ToString 方法，返回坐标的字符串表示
        public override string ToString() => $"{X+1} , {Y+1}";

        // 解构方法，将 x 和 y 分解出来
        internal void Deconstruct(out int x, out int y)
        {
            x = X;
            y = Y;
        }

        // 静态方法，尝试根据浮点数值创建坐标
        internal static bool TryCreate(float x, float y, out Coordinates coordinates)
        {
            // 对 x 和 y 进行四舍五入
            var roundedX = Round(x);
            var roundedY = Round(y);

            // 如果四舍五入后的值在 0 到 7 之间，则创建坐标并返回 true
            if (IsValid(roundedX) && IsValid(roundedY))
            {
                coordinates = new Coordinates(roundedX, roundedY);
                return true;
            }

            // 否则返回 false
            coordinates = default;
            return false;

            // 静态方法，对浮点数值进行四舍五入
            static int Round(float value) => (int)Math.Round(value, MidpointRounding.AwayFromZero);
        }

        // 获取到目标坐标的方向和距离
        internal (float Direction, float Distance) GetDirectionAndDistanceTo(Coordinates destination) =>
            DirectionAndDistance.From(this).To(destination);

        // 获取到目标坐标的距离
        internal float GetDistanceTo(Coordinates destination)
        {
            var (_, distance) = GetDirectionAndDistanceTo(destination);
            return distance;
        }
    }
}

```