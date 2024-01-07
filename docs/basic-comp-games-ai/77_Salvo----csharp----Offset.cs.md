# `basic-computer-games\77_Salvo\csharp\Offset.cs`

```

namespace Salvo;

// 定义一个命名空间 Salvo

internal record struct Offset(int X, int Y)
{
    // 定义一个内部的结构体 Offset，包含 X 和 Y 两个整型字段

    // 定义一个静态的只读的 Offset 对象 Zero，值为 (0, 0)
    public static readonly Offset Zero = 0;

    // 定义一个重载运算符 *，实现 Offset 对象与整数的乘法操作
    public static Offset operator *(Offset offset, int scale) => new(offset.X * scale, offset.Y * scale);

    // 定义一个隐式转换，将整数转换为 Offset 对象
    public static implicit operator Offset(int value) => new(value, value);

    // 定义一个静态属性 Units，返回一个包含九个 Offset 对象的 IEnumerable
    public static IEnumerable<Offset> Units
    {
        get
        {
            // 遍历 -1 到 1 的 x 值
            for (int x = -1; x <= 1; x++)
            {
                // 遍历 -1 到 1 的 y 值
                for (int y = -1; y <= 1; y++)
                {
                    // 创建一个新的 Offset 对象
                    var offset = new Offset(x, y);
                    // 如果 offset 不等于 Zero，则将其返回
                    if (offset != Zero) { yield return offset; }
                }
            }
        }
    }
}

```