# `basic-computer-games\90_Tower\csharp\UI\Input.cs`

```

// 引入系统和集合类库
using System;
using System.Collections.Generic;

namespace Tower.UI
{
    // 提供模拟BASIC解释器键盘输入例程的输入方法
    internal static class Input
    {
        // 输出提示信息
        private static void Prompt(string text = "") => Console.Write($"{text}? ");

        // 读取Yes/No类型的输入
        internal static bool ReadYesNo(string prompt, string retryPrompt)
        {
            var response = ReadString(prompt);

            while (true)
            {
                if (response.Equals("No", StringComparison.InvariantCultureIgnoreCase)) { return false; }
                if (response.Equals("Yes", StringComparison.InvariantCultureIgnoreCase)) { return true; }
                response = ReadString(retryPrompt);
            }
        }

        // 尝试读取数字类型的输入
        internal static bool TryReadNumber(Prompt prompt, out int number)
        {
            var message = prompt.Message;

            for (int retryCount = 0; retryCount <= prompt.RetriesAllowed; retryCount++)
            {
                if (retryCount > 0) { Console.WriteLine(prompt.RetryMessage); }

                if (prompt.TryValidateResponse(ReadNumber(message), out number)) { return true; }

                if (!prompt.RepeatPrompt) { message = ""; }
            }

            Console.WriteLine(prompt.QuitMessage);

            number = 0;
            return false;
        }

        // 读取数字类型的输入
        private static float ReadNumber(string prompt)
        {
            Prompt(prompt);

            while (true)
            {
                var inputValues = ReadStrings();

                if (TryParseNumber(inputValues[0], out var number))
                {
                    if (inputValues.Length > 1)
                    {
                        Console.WriteLine("!Extra input ingored");
                    }

                    return number;
                }
            }
        }

        // 读取字符串类型的输入
        private static string ReadString(string prompt)
        {
            Prompt(prompt);

            var inputValues = ReadStrings();
            if (inputValues.Length > 1)
            {
                Console.WriteLine("!Extra input ingored");
            }
            return inputValues[0];
        }

        // 读取以逗号分隔的输入字符串
        private static string[] ReadStrings() => Console.ReadLine().Split(',', StringSplitOptions.TrimEntries);

        // 尝试解析输入的字符串为数字类型
        private static bool TryParseNumber(string text, out float number)
        {
            if (float.TryParse(text, out number)) { return true; }

            Console.WriteLine("!Number expected - retry input line");
            number = default;
            return false;
        }
    }
}

```