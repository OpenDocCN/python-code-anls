# `08_Batnum\csharp\ConsoleUtilities.cs`

```
        // Ask the user a question and expects a comma separated pair of numbers representing a number range in response
        // the range provided must have a maximum which is greater than the minimum
        /// <summary>
        /// Ask the user a question and expects a comma separated pair of numbers representing a number range in response
        /// the range provided must have a maximum which is greater than the minimum
        /// </summary>
        /// <param name="question">The question to ask</param>
        /// <param name="minimum">The minimum value expected</param>
        /// <param name="maximum">The maximum value expected</param>
        /// <returns>A pair of numbers representing the minimum and maximum of the range</returns>
        public static (int min, int max) AskNumberRangeQuestion(string question, Func<int, int, bool> Validate)
        {
            // Code to ask the user a question and get the response
            // Validate the response using the provided validation function
            // Return the pair of numbers representing the minimum and maximum of the range
        }
            while (true)
            {
                Console.Write(question);  # 输出提示信息
                Console.Write(" ");  # 输出空格
                string[] rawInput = Console.ReadLine().Split(',');  # 读取用户输入并按逗号分割成字符串数组
                if (rawInput.Length == 2)  # 判断输入的字符串数组长度是否为2
                {
                    if (int.TryParse(rawInput[0], out int min) && int.TryParse(rawInput[1], out int max))  # 尝试将字符串转换为整数，如果成功则赋值给min和max
                    {
                        if (Validate(min, max))  # 调用Validate函数验证min和max是否符合要求
                        {
                            return (min, max);  # 如果符合要求则返回min和max的元组
                        }
                    }
                }
                Console.WriteLine();  # 输出空行
            }
        }

        /// <summary>
        /// <summary>
        /// Ask the user a question and expects a number in response
        /// </summary>
        /// <param name="question">The question to ask</param>
        /// <param name="minimum">A minimum value expected</param>
        /// <param name="maximum">A maximum value expected</param>
        /// <returns>The number the user entered</returns>
        public static int AskNumberQuestion(string question, Func<int, bool> Validate)
        {
            // 创建一个函数，用于向用户提问并期望得到一个数字作为回答
            while (true)
            {
                // 输出问题到控制台
                Console.Write(question);
                Console.Write(" ");
                // 读取用户输入的内容
                string rawInput = Console.ReadLine();
                // 尝试将用户输入的内容转换为整数
                if (int.TryParse(rawInput, out int number))
                {
                    // 如果转换成功，使用传入的验证函数对数字进行验证
                    if (Validate(number))
                    {
                        // 如果验证通过，返回用户输入的数字
                        return number;
                    }
                }
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Align content to center of console.
        /// </summary>
        /// <param name="content">Content to center</param>
        /// <returns>Center aligned text</returns>
        public static string CenterText(string content)
        {
            // 获取控制台窗口的宽度
            int windowWidth = Console.WindowWidth;
            // 返回居中对齐的文本
            return String.Format("{0," + ((windowWidth / 2) + (content.Length / 2)) + "}", content);
        }

        /// <summary>
        ///     Writes the specified data, followed by the current line terminator, to the standard output stream, while wrapping lines that would otherwise break words.
        ///     source: https://stackoverflow.com/questions/20534318/make-console-writeline-wrap-words-instead-of-letters
        /// </summary>
        /// <param name="paragraph">The value to write.</param>
        /// <param name="tabSize">The value that indicates the column width of tab characters.</param>
        public static void WriteLineWordWrap(string paragraph, int tabSize = 4)
        {
            // 将制表符替换为指定列宽的空格，并根据换行符拆分段落为行数组
            string[] lines = paragraph
                .Replace("\t", new String(' ', tabSize))
                .Split(new string[] { Environment.NewLine }, StringSplitOptions.None);

            // 遍历每一行
            for (int i = 0; i < lines.Length; i++)
            {
                // 获取当前行的内容
                string process = lines[i];
                // 创建一个空列表用于存储换行后的内容
                List<String> wrapped = new List<string>();

                // 当行的长度超过控制台窗口宽度时，进行换行处理
                while (process.Length > Console.WindowWidth)
                {
                    // 在当前行的指定位置（控制台窗口宽度内或当前行长度内的较小值）处查找空格，作为换行位置
                    int wrapAt = process.LastIndexOf(' ', Math.Min(Console.WindowWidth - 1, process.Length));
                    // 如果没有找到合适的换行位置，则跳出循环
                    if (wrapAt <= 0) break;

                    // 将当前行的内容从开头到换行位置的部分添加到换行后的列表中
                    wrapped.Add(process.Substring(0, wrapAt));
                    // 移除已经添加到换行后列表中的部分
                    process = process.Remove(0, wrapAt + 1);
                }
# 遍历字符串列表 wrapped 中的每个字符串，将其赋值给变量 wrap，然后打印出来
foreach (string wrap in wrapped)
{
    Console.WriteLine(wrap);
}

# 打印变量 process 的值
Console.WriteLine(process);
```