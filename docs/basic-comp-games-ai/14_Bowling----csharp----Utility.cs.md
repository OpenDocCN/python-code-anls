# `basic-computer-games\14_Bowling\csharp\Utility.cs`

```
// 命名空间 Bowling，包含 Bowling 相关的类和方法
namespace Bowling
{
    // 内部静态类 Utility，包含一些通用的辅助方法
    internal static class Utility
    {
        // 返回指定宽度的整数的字符串表示，不足部分在左侧填充空格
        public static string PadInt(int value, int width)
        {
            return value.ToString().PadLeft(width);
        }
        // 从控制台输入获取整数
        public static int InputInt()
        {
            // 循环直到成功获取整数输入
            while (true)
            {
                // 尝试将输入的字符串转换为整数，如果成功则返回整数值，否则提示重新输入
                if (int.TryParse(InputString(), out int i))
                    return i;
                else
                    PrintString("!NUMBER EXPECTED - RETRY INPUT LINE");
            }
        }
        // 从控制台输入获取字符串
        public static string InputString()
        {
            // 输出提示符“? ”，并获取用户输入的字符串，转换为大写后返回
            PrintString("? ", false);
            var input = Console.ReadLine();
            return input == null ? string.Empty : input.ToUpper();
        }
        // 在控制台输出整数，可选择是否换行
        public static void PrintInt(int value, bool newLine = false)
        {
            PrintString($"{value} ", newLine);
        }
        // 在控制台输出字符串，可选择是否换行
        public static void PrintString(bool newLine = true)
        {
            PrintString(0, string.Empty);
        }
        // 在控制台输出指定缩进量的字符串，可选择是否换行
        public static void PrintString(int tab, bool newLine = true)
        {
            PrintString(tab, string.Empty, newLine);
        }
        // 在控制台输出字符串，可选择是否换行
        public static void PrintString(string value, bool newLine = true)
        {
            PrintString(0, value, newLine);
        }
        // 在控制台输出指定缩进量的字符串，可选择是否换行
        public static void PrintString(int tab, string value, bool newLine = true)
        {
            // 在控制台输出指定数量的空格，然后输出字符串值
            Console.Write(new String(' ', tab));
            Console.Write(value);
            // 如果需要换行，则输出换行符
            if (newLine) Console.WriteLine();
        }
    }
}
```