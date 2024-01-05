# `90_Tower\csharp\UI\Input.cs`

```
using System;  // 导入 System 命名空间，包含了基本的系统类型和基本类
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间，包含了泛型集合类

namespace Tower.UI  // 命名空间 Tower.UI
{
    // 提供模拟 BASIC 解释器键盘输入例程的输入方法
    internal static class Input  // 内部静态类 Input
    {
        private static void Prompt(string text = "") => Console.Write($"{text}? ");  // 定义私有静态方法 Prompt，用于在控制台输出提示信息

        internal static bool ReadYesNo(string prompt, string retryPrompt)  // 定义内部静态方法 ReadYesNo，返回布尔值，接受两个字符串参数
        {
            var response = ReadString(prompt);  // 声明并初始化变量 response，调用 ReadString 方法并传入 prompt 参数

            while (true)  // 进入无限循环
            {
                if (response.Equals("No", StringComparison.InvariantCultureIgnoreCase)) { return false; }  // 如果 response 等于 "No"，则返回 false
                if (response.Equals("Yes", StringComparison.InvariantCultureIgnoreCase)) { return true; }  // 如果 response 等于 "Yes"，则返回 true
                response = ReadString(retryPrompt);  // 调用 ReadString 方法并传入 retryPrompt 参数，将结果赋值给 response
            }
        }

        // 尝试从用户输入中读取一个数字
        internal static bool TryReadNumber(Prompt prompt, out int number)
        {
            // 获取提示消息
            var message = prompt.Message;

            // 循环尝试读取用户输入，直到达到最大重试次数
            for (int retryCount = 0; retryCount <= prompt.RetriesAllowed; retryCount++)
            {
                // 如果不是第一次重试，则打印重试消息
                if (retryCount > 0) { Console.WriteLine(prompt.RetryMessage); }

                // 尝试从用户输入中读取数字，并验证输入是否有效
                if (prompt.TryValidateResponse(ReadNumber(message), out number)) { return true; }

                // 如果不需要重复提示，则清空消息
                if (!prompt.RepeatPrompt) { message = ""; }
            }

            // 打印退出消息
            Console.WriteLine(prompt.QuitMessage);

            // 将 number 设置为默认值 0，并返回 false
            number = 0;
            return false;
        }
        # 读取一个数字并返回
        private static float ReadNumber(string prompt)
        {
            # 显示提示信息
            Prompt(prompt);

            # 无限循环，直到成功读取到数字
            while (true)
            {
                # 读取用户输入的字符串
                var inputValues = ReadStrings();

                # 尝试将输入的字符串解析为数字
                if (TryParseNumber(inputValues[0], out var number)):
                    # 如果成功解析为数字
                    if (inputValues.Length > 1):
                        # 如果输入包含多余的内容，打印警告信息
                        Console.WriteLine("!Extra input ingored");
                    # 返回成功解析的数字
                    return number;
            }
        }
        # 读取字符串并返回
        private static string ReadString(string prompt)
        {
            # 调用Prompt方法显示提示信息
            Prompt(prompt);

            # 读取输入的字符串并以逗号分隔，去除空格
            var inputValues = ReadStrings();
            # 如果输入的字符串数组长度大于1，输出警告信息
            if (inputValues.Length > 1)
            {
                Console.WriteLine("!Extra input ingored");
            }
            # 返回输入的第一个字符串
            return inputValues[0];
        }

        # 以逗号分隔并去除空格的方式读取输入的字符串并返回字符串数组
        private static string[] ReadStrings() => Console.ReadLine().Split(',', StringSplitOptions.TrimEntries);

        # 尝试将输入的字符串转换为浮点数，如果成功则返回true并将结果赋值给number，否则输出警告信息
        private static bool TryParseNumber(string text, out float number)
        {
            if (float.TryParse(text, out number)) { return true; }

            Console.WriteLine("!Number expected - retry input line");
            number = default;  # 将变量 number 的值设置为默认值 default
            return false;  # 返回布尔值 false
        }
    }
}
```