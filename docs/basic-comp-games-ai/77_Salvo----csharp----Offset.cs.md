# `basic-computer-games\77_Salvo\csharp\Offset.cs`

```
// 命名空间 Salvo
namespace Salvo;

// 定义一个内部的记录结构 Offset，包含 X 和 Y 两个整型字段
internal record struct Offset(int X, int Y)
{
    // 定义一个公共的静态只读的 Zero 偏移量，初始值为 (0, 0)
    public static readonly Offset Zero = 0;

    // 定义偏移量与整数相乘的运算符重载，返回新的偏移量
    public static Offset operator *(Offset offset, int scale) => new(offset.X * scale, offset.Y * scale);

    // 定义从整数到偏移量的隐式转换
    public static implicit operator Offset(int value) => new(value, value);

    // 定义一个返回偏移量单位的可枚举集合
    public static IEnumerable<Offset> Units
    {
        get
        {
            // 遍历 x 从 -1 到 1
            for (int x = -1; x <= 1; x++)
            {
                // 遍历 y 从 -1 到 1
                for (int y = -1; y <= 1; y++)
                {
                    // 创建一个新的偏移量对象
                    var offset = new Offset(x, y);
                    // 如果偏移量不等于 Zero，则返回该偏移量
                    if (offset != Zero) { yield return offset; }
                }
            }
        }
    }
}
```