# `basic-computer-games\10_Blackjack\csharp\Prompt.cs`

```

// 引入 System 命名空间
using System;

// 定义名为 Prompt 的静态类
namespace Blackjack
{
    public static class Prompt
    {
        // 用于提示用户输入 yes 或 no，并返回对应的布尔值
        public static bool ForYesNo(string prompt)
        {
            // 循环直到用户输入有效的值
            while(true)
            {
                // 提示用户输入
                Console.Write("{0} ", prompt);
                // 读取用户输入
                var input = Console.ReadLine();
                // 如果输入以 "y" 开头（不区分大小写），返回 true
                if (input.StartsWith("y", StringComparison.InvariantCultureIgnoreCase))
                    return true;
                // 如果输入以 "n" 开头（不区分大小写），返回 false
                if (input.StartsWith("n", StringComparison.InvariantCultureIgnoreCase))
                    return false;
                // 如果输入不符合要求，提示用户重新输入
                WriteNotUnderstood();
            }
        }

        // 用于提示用户输入整数，并返回该整数
        public static int ForInteger(string prompt, int minimum = 1, int maximum = int.MaxValue)
        {
            // 循环直到用户输入有效的整数
            while (true)
            {
                // 提示用户输入
                Console.Write("{0} ", prompt);
                // 尝试将用户输入转换为整数
                if (!int.TryParse(Console.ReadLine(), out var number))
                    // 如果无法转换，提示用户重新输入
                    WriteNotUnderstood();
                else if (number < minimum || number > maximum)
                    // 如果输入的整数不在指定范围内，提示用户重新输入
                    Console.WriteLine("Sorry, I need a number between {0} and {1}.", minimum, maximum);
                else
                    // 返回输入的整数
                    return number;
            }
        }

        // 用于提示用户输入指定字符，并返回该字符
        public static string ForCommandCharacter(string prompt, string allowedCharacters)
        {
            // 循环直到用户输入有效的字符
            while (true)
            {
                // 提示用户输入
                Console.Write("{0} ", prompt);
                // 读取用户输入
                var input = Console.ReadLine();
                if (input.Length > 0)
                {
                    // 获取输入的第一个字符
                    var character = input.Substring(0, 1);
                    // 查找输入的字符在允许的字符中的位置
                    var characterIndex = allowedCharacters.IndexOf(character, StringComparison.InvariantCultureIgnoreCase);
                    if (characterIndex != -1)
                        // 如果输入的字符在允许的字符中，返回该字符
                        return allowedCharacters.Substring(characterIndex, 1);
                }

                // 如果输入的字符不在允许的字符中，提示用户重新输入
                Console.WriteLine("Type one of {0} please", String.Join(", ", allowedCharacters.ToCharArray()));
            }
        }

        // 用于提示用户输入不被理解的信息
        private static void WriteNotUnderstood()
        {
            Console.WriteLine("Sorry, I didn't understand.");
        }
    }
}

```