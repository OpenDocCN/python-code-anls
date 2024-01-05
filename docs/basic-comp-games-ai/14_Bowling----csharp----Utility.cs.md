# `14_Bowling\csharp\Utility.cs`

```
# 导入所需的命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bowling
{
    // 创建一个静态类 Utility
    internal static class Utility
    {
        // 创建一个静态方法 PadInt，用于将整数转换为指定宽度的字符串
        public static string PadInt(int value, int width)
        {
            return value.ToString().PadLeft(width);
        }
        // 创建一个静态方法 InputInt，用于接收用户输入的整数
        public static int InputInt()
        {
            // 创建一个无限循环，直到用户输入正确的整数为止
            while (true)
            {
                // 如果用户输入的是整数，则将其转换为整数并返回
                if (int.TryParse(InputString(), out int i))
                    return i;  // 返回整数值
                else
                    PrintString("!NUMBER EXPECTED - RETRY INPUT LINE");
            }
        }
        public static string InputString()
        {
            PrintString("? ", false);  # 打印提示符，等待用户输入字符串
            var input = Console.ReadLine();  # 读取用户输入的字符串
            return input == null ? string.Empty : input.ToUpper();  # 将用户输入的字符串转换为大写并返回
        }
        public static void PrintInt(int value, bool newLine = false)
        {
            PrintString($"{value} ", newLine);  # 打印整数值，并根据参数决定是否换行
        }
        public static void PrintString(bool newLine = true)
        {
            PrintString(0, string.Empty);  # 调用重载的 PrintString 方法，传入默认参数
        }
        public static void PrintString(int tab, bool newLine = true)
        {
            PrintString(tab, string.Empty, newLine);
        }
        public static void PrintString(string value, bool newLine = true)
        {
            PrintString(0, value, newLine);
        }
        public static void PrintString(int tab, string value, bool newLine = true)
        {
            Console.Write(new String(' ', tab));  // 在控制台上打印指定数量的空格
            Console.Write(value);  // 在控制台上打印指定的字符串
            if (newLine) Console.WriteLine();  // 如果需要换行，则在控制台上打印换行符
        }
    }
}
```