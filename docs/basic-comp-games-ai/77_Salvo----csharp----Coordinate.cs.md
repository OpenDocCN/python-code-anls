# `77_Salvo\csharp\Coordinate.cs`

```
namespace Salvo;

internal record struct Coordinate(int Value)  // 定义了一个名为 Coordinate 的记录结构体，包含一个整数值字段 Value
{
    public const int MinValue = 1;  // 定义了一个名为 MinValue 的常量，值为 1
    public const int MaxValue = 10;  // 定义了一个名为 MaxValue 的常量，值为 10

    public static IEnumerable<Coordinate> Range => Enumerable.Range(1, 10).Select(v => new Coordinate(v));  // 定义了一个名为 Range 的静态属性，返回一个包含从 1 到 10 的 Coordinate 对象的可枚举集合

    public bool IsInRange => Value is >= MinValue and <= MaxValue;  // 定义了一个名为 IsInRange 的属性，用于检查 Value 是否在 MinValue 和 MaxValue 之间

    public static Coordinate Create(float value) => new((int)value);  // 定义了一个名为 Create 的静态方法，用于根据给定的浮点数值创建一个 Coordinate 对象

    public static bool TryCreateValid(float value, out Coordinate coordinate)  // 定义了一个名为 TryCreateValid 的静态方法，用于尝试根据给定的浮点数值创建一个有效的 Coordinate 对象
    {
        coordinate = default;  // 将 coordinate 初始化为默认值
        if (value != (int)value) { return false; }  // 如果 value 不等于其整数部分，则返回 false

        var result = Create(value);  // 调用 Create 方法创建一个 Coordinate 对象
        if (result.IsInRange)
        {
            // 如果结果在范围内，将坐标设置为结果并返回 true
            coordinate = result;
            return true;
        }

        // 如果结果不在范围内，返回 false
        return false;
    }

    // 将坐标值调整到范围内
    public Coordinate BringIntoRange(IRandom random)
        => Value switch
        {
            // 如果值小于最小值，返回一个新的坐标，值为最小值加上一个随机浮点数
            < MinValue => new Coordinate(MinValue + (int)random.NextFloat(2.5F)),
            // 如果值大于最大值，返回一个新的坐标，值为最大值减去一个随机浮点数
            > MaxValue => new Coordinate(MaxValue - (int)random.NextFloat(2.5F)),
            // 如果值在范围内，返回原始坐标
            _ => this
        };

    // 将浮点数隐式转换为坐标
    public static implicit operator Coordinate(float value) => Create(value);
    // 将坐标隐式转换为整数
    public static implicit operator int(Coordinate coordinate) => coordinate.Value;
    public static Coordinate operator +(Coordinate coordinate, int offset) => new(coordinate.Value + offset);
    // 定义了一个重载的加法运算符，使得可以对 Coordinate 对象和整数进行相加操作，返回一个新的 Coordinate 对象

    public static int operator -(Coordinate a, Coordinate b) => a.Value - b.Value;
    // 定义了一个重载的减法运算符，使得可以对两个 Coordinate 对象进行相减操作，返回一个整数值

    public override string ToString() => $" {Value} ";
    // 重写了 ToString 方法，返回 Coordinate 对象的值的字符串表示形式
}
```