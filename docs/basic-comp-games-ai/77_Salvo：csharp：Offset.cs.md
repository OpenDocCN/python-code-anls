# `d:/src/tocomm/basic-computer-games\77_Salvo\csharp\Offset.cs`

```
namespace Salvo;  // 命名空间声明

internal record struct Offset(int X, int Y)  // 定义名为 Offset 的结构体，包含 X 和 Y 两个整型字段
{
    public static readonly Offset Zero = 0;  // 定义名为 Zero 的静态只读 Offset 对象，值为 0

    public static Offset operator *(Offset offset, int scale) => new(offset.X * scale, offset.Y * scale);  // 定义名为 * 的运算符重载，实现 Offset 对象与整数的乘法操作

    public static implicit operator Offset(int value) => new(value, value);  // 定义隐式转换操作符，将整数转换为 Offset 对象

    public static IEnumerable<Offset> Units  // 定义名为 Units 的静态属性，返回 Offset 对象的集合
    {
        get
        {
            for (int x = -1; x <= 1; x++)  // 循环遍历 x 值
            {
                for (int y = -1; y <= 1; y++)  // 循环遍历 y 值
                {
                    var offset = new Offset(x, y);  // 创建新的 Offset 对象
                    if (offset != Zero) { yield return offset; }  // 如果 offset 不等于 Zero，则将其返回
                }
            }
        }
    }
抱歉，给定的代码片段不完整，无法为其添加注释。
```