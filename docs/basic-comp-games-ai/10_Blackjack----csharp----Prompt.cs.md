# `basic-computer-games\10_Blackjack\csharp\Prompt.cs`

```
# 引入系统命名空间
using System;

# 创建名为Blackjack的命名空间
namespace Blackjack
{
    # 创建名为Prompt的静态类
    public static class Prompt
    // 询问用户一个问题，接受 yes 或 no 作为回答，返回布尔值
    public static bool ForYesNo(string prompt)
    {
        // 无限循环，直到得到有效的回答
        while(true)
        {
            // 输出提示信息
            Console.Write("{0} ", prompt);
            // 读取用户输入
            var input = Console.ReadLine();
            // 如果输入以 "y" 开头（不区分大小写），返回 true
            if (input.StartsWith("y", StringComparison.InvariantCultureIgnoreCase))
                return true;
            // 如果输入以 "n" 开头（不区分大小写），返回 false
            if (input.StartsWith("n", StringComparison.InvariantCultureIgnoreCase))
                return false;
            // 如果输入不符合要求，输出提示信息
            WriteNotUnderstood();
        }
    }

    // 询问用户一个问题，接受整数作为回答，返回整数值
    public static int ForInteger(string prompt, int minimum = 1, int maximum = int.MaxValue)
    {
        // 无限循环，直到得到有效的整数值
        while (true)
        {
            // 输出提示信息
            Console.Write("{0} ", prompt);
            // 尝试将用户输入转换为整数
            if (!int.TryParse(Console.ReadLine(), out var number))
                // 如果无法转换，输出提示信息
                WriteNotUnderstood();
            else if (number < minimum || number > maximum)
                // 如果输入的值不在指定范围内，输出提示信息
                Console.WriteLine("Sorry, I need a number between {0} and {1}.", minimum, maximum);
            else
                // 返回有效的整数值
                return number;
        }
    }

    // 询问用户一个问题，接受特定字符作为回答，返回符合要求的字符
    public static string ForCommandCharacter(string prompt, string allowedCharacters)
    {
        // 无限循环，直到得到有效的字符
        while (true)
        {
            // 输出提示信息
            Console.Write("{0} ", prompt);
            // 读取用户输入
            var input = Console.ReadLine();
            if (input.Length > 0)
            {
                // 获取输入的第一个字符
                var character = input.Substring(0, 1);
                // 在允许的字符列表中查找输入的字符
                var characterIndex = allowedCharacters.IndexOf(character, StringComparison.InvariantCultureIgnoreCase);
                if (characterIndex != -1)
                    // 如果找到输入的字符，返回该字符
                    return allowedCharacters.Substring(characterIndex, 1);
            }

            // 如果输入不符合要求，输出提示信息
            Console.WriteLine("Type one of {0} please", String.Join(", ", allowedCharacters.ToCharArray()));
        }
    }

    // 输出未能理解的提示信息
    private static void WriteNotUnderstood()
    {
        Console.WriteLine("Sorry, I didn't understand.");
    }
# 闭合前面的函数定义
```