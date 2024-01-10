# `basic-computer-games\77_Salvo\csharp\Coordinate.cs`

```
// 命名空间 Salvo
namespace Salvo
{
    // 定义一个内部的记录结构 Coordinate，包含一个整数值
    internal record struct Coordinate(int Value)
    {
        // 定义常量 MinValue 和 MaxValue
        public const int MinValue = 1;
        public const int MaxValue = 10;

        // 定义静态属性 Range，返回一个范围在 1 到 10 的 Coordinate 序列
        public static IEnumerable<Coordinate> Range => Enumerable.Range(1, 10).Select(v => new Coordinate(v));

        // 定义属性 IsInRange，判断当前 Coordinate 的值是否在 MinValue 和 MaxValue 之间
        public bool IsInRange => Value is >= MinValue and <= MaxValue;

        // 定义静态方法 Create，根据浮点数值创建 Coordinate 对象
        public static Coordinate Create(float value) => new((int)value);

        // 定义静态方法 TryCreateValid，尝试根据浮点数值创建有效的 Coordinate 对象
        public static bool TryCreateValid(float value, out Coordinate coordinate)
        {
            coordinate = default;
            if (value != (int)value) { return false; }

            var result = Create(value);

            if (result.IsInRange)
            {
                coordinate = result;
                return true;
            }

            return false;
        }

        // 定义方法 BringIntoRange，将当前 Coordinate 的值调整到 MinValue 和 MaxValue 之间
        public Coordinate BringIntoRange(IRandom random)
            => Value switch
            {
                < MinValue => new(MinValue + (int)random.NextFloat(2.5F)),
                > MaxValue => new(MaxValue - (int)random.NextFloat(2.5F)),
                _ => this
            };

        // 定义隐式转换操作符，将浮点数值转换为 Coordinate 对象
        public static implicit operator Coordinate(float value) => Create(value);
        // 定义隐式转换操作符，将 Coordinate 对象转换为整数值
        public static implicit operator int(Coordinate coordinate) => coordinate.Value;

        // 定义加法操作符，实现 Coordinate 对象与整数值的加法
        public static Coordinate operator +(Coordinate coordinate, int offset) => new(coordinate.Value + offset);
        // 定义减法操作符，实现两个 Coordinate 对象之间的减法
        public static int operator -(Coordinate a, Coordinate b) => a.Value - b.Value;

        // 重写 ToString 方法，返回 Coordinate 对象的值的字符串表示
        public override string ToString() => $" {Value} ";
    }
}
```