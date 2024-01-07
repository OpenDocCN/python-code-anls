# `basic-computer-games\61_Math_Dice\csharp\StringExtensions.cs`

```

// 命名空间 MathDice，定义了一个静态类 StringExtensions
namespace MathDice
{
    // 定义了一个静态类 StringExtensions
    public static class StringExtensions
    {
        // 定义了一个私有常量，表示控制台的默认宽度
        private const int ConsoleWidth = 120; // default console width

        // 定义了一个扩展方法，用于将字符串居中对齐
        public static string CentreAlign(this string value)
        {
            // 计算需要添加的空格数，使得字符串居中
            int spaces = ConsoleWidth - value.Length;
            int leftPadding = spaces / 2 + value.Length;

            // 在字符串左侧添加左填充空格，右侧添加右填充空格，使得字符串居中
            return value.PadLeft(leftPadding).PadRight(ConsoleWidth);
        }
    }
}

```