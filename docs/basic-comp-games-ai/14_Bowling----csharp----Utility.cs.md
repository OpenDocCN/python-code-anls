# `basic-computer-games\14_Bowling\csharp\Utility.cs`

```

// 命名空间 Bowling
namespace Bowling
{
    // 内部静态类 Utility
    internal static class Utility
    {
        // 返回指定宽度的整数的字符串表示形式，左对齐，不足部分用空格填充
        public static string PadInt(int value, int width)
        {
            return value.ToString().PadLeft(width);
        }
        // 获取用户输入的整数
        public static int InputInt()
        {
            // 循环直到用户输入有效的整数
            while (true)
            {
                // 尝试将输入的字符串转换为整数，如果成功则返回该整数
                if (int.TryParse(InputString(), out int i))
                    return i;
                // 如果转换失败，则打印错误信息并重新输入
                else
                    PrintString("!NUMBER EXPECTED - RETRY INPUT LINE");
            }
        }
        // 获取用户输入的字符串
        public static string InputString()
        {
            // 打印提示符，获取用户输入并转换为大写
            PrintString("? ", false);
            var input = Console.ReadLine();
            return input == null ? string.Empty : input.ToUpper();
        }
        // 打印整数
        public static void PrintInt(int value, bool newLine = false)
        {
            PrintString($"{value} ", newLine);
        }
        // 打印字符串
        public static void PrintString(bool newLine = true)
        {
            PrintString(0, string.Empty);
        }
        // 打印带有制表符的字符串
        public static void PrintString(int tab, bool newLine = true)
        {
            PrintString(tab, string.Empty, newLine);
        }
        // 打印字符串
        public static void PrintString(string value, bool newLine = true)
        {
            PrintString(0, value, newLine);
        }
        // 打印带有制表符的字符串
        public static void PrintString(int tab, string value, bool newLine = true)
        {
            // 在控制台上打印制表符和值，如果需要换行则换行
            Console.Write(new String(' ', tab));
            Console.Write(value);
            if (newLine) Console.WriteLine();
        }
    }
}

```