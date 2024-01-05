# `61_Math_Dice\csharp\StringExtensions.cs`

```
// 命名空间 MathDice，定义了一个静态类 StringExtensions
namespace MathDice
{
    // 定义了一个静态类 StringExtensions
    public static class StringExtensions
    {
        // 声明一个私有常量 ConsoleWidth，表示控制台的默认宽度为 120
        private const int ConsoleWidth = 120; // default console width

        // 定义了一个扩展方法 CentreAlign，用于将字符串居中对齐
        public static string CentreAlign(this string value)
        {
            // 计算需要添加的空格数，使得字符串居中
            int spaces = ConsoleWidth - value.Length;
            // 计算左侧填充空格数
            int leftPadding = spaces / 2 + value.Length;

            // 返回经过左侧填充和右侧填充后的字符串，使其居中对齐
            return value.PadLeft(leftPadding).PadRight(ConsoleWidth);
        }
    }
}
```