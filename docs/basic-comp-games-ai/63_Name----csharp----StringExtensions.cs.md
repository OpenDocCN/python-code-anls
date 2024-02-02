# `basic-computer-games\63_Name\csharp\StringExtensions.cs`

```py
// 命名空间定义
namespace Name
{
    // 定义静态类 StringExtensions
    public static class StringExtensions
    {
        // 定义私有常量，表示控制台宽度
        private const int ConsoleWidth = 120; // default console width

        // 定义字符串居中对齐的扩展方法
        public static string CentreAlign(this string value)
        {
            // 计算需要添加的空格数
            int spaces = ConsoleWidth - value.Length;
            // 计算左侧填充空格数
            int leftPadding = spaces / 2 + value.Length;

            // 返回居中对齐后的字符串
            return value.PadLeft(leftPadding).PadRight(ConsoleWidth);
        }

        // 定义字符串反转的扩展方法
        public static string Reverse(this string value)
        {
            // 如果输入字符串为空，则返回空
            if (value is null)
            {
                return null;
            }

            // 将字符串转换为字符数组，反转数组中的字符
            char[] characterArray = value.ToCharArray();
            Array.Reverse(characterArray);
            // 返回反转后的字符串
            return new String(characterArray);
        }

        // 定义字符串排序的扩展方法
        public static string Sort(this string value)
        {
            // 如果输入字符串为空，则返回空
            if (value is null)
            {
                return null;
            }

            // 将字符串转换为字符数组，对数组中的字符进行排序
            char[] characters = value.ToCharArray();
            Array.Sort(characters);
            // 返回排序后的字符串
            return new string(characters);
        }
    }
}
```