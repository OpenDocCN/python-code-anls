# `d:/src/tocomm/basic-computer-games\60_Mastermind\csharp\Controller.cs`

```
        /// </remarks>
        private static readonly ImmutableDictionary<char, int> ColorMap = 
            Colors.ToImmutableDictionary(c => c.Letter, c => c.Value);

        /// <summary>
        /// Prompts the user to select a color from the available options.
        /// </summary>
        /// <returns>The integer value representing the selected color.</returns>
        public static int GetColorChoice()
        {
            // Display the available color options to the user
            Console.WriteLine("Available colors:");
            foreach (var color in Colors)
            {
                Console.WriteLine($"{color.Letter}: {color.Name}");
            }

            // Prompt the user to select a color
            Console.Write("Enter the letter for your chosen color: ");
            char choice = char.ToUpper(Console.ReadKey().KeyChar);
            Console.WriteLine();

            // Validate the user's input and return the corresponding color value
            if (ColorMap.TryGetValue(choice, out int colorValue))
            {
                return colorValue;
            }
            else
            {
                Console.WriteLine("Invalid color choice. Please try again.");
                return GetColorChoice();
            }
        }
    }
}
        /// <remarks>
        /// 使用 Colors.List 中的颜色信息和索引创建不可变字典，以颜色的简称作为键，索引作为值
        /// </remarks>
        private static ImmutableDictionary<char, int> ColorsByKey = Colors.List
            .Select((info, index) => (key: info.ShortName, index))
            .ToImmutableDictionary(entry => entry.key, entry => entry.index);

        /// <summary>
        /// 获取秘密代码中要使用的颜色数量。
        /// </summary>
        public static int GetNumberOfColors()
        {
            // 获取颜色列表的长度作为最大颜色数量
            var maximumColors = Colors.List.Length;
            var colors = 0;

            // 当颜色数量小于1或大于最大颜色数量时，循环提示用户输入正确的颜色数量
            while (colors < 1 || colors > maximumColors)
            {
                colors = GetInteger(View.PromptNumberOfColors); // 从视图中获取用户输入的颜色数量
                if (colors > maximumColors)
                    View.NotifyTooManyColors(maximumColors); // 如果输入的颜色数量超过最大颜色数量，通知用户输入过多颜色
            }
        /// <summary>
        /// Gets the number of positions in the secret code.
        /// </summary>
        /// <returns></returns>
        public static int GetNumberOfPositions()
        {
            // Note: We should probably ensure that the user enters a sane
            //  number of positions here.  (Things go south pretty quickly
            //  with a large number of positions.)  But since the original
            //  program did not, neither will we.
            // 获取秘密代码中位置的数量
            return GetInteger(View.PromptNumberOfPositions);
        }

        /// <summary>
        /// Gets the number of rounds to play.
        /// </summary>
        // 获取要玩的回合数
        public static int GetNumberOfRounds()
        {
            // 注意：无意义的回合数（如0或负数）是无害的，但验证仍然是有意义的。
            return GetInteger(View.PromptNumberOfRounds);
        }

        /// <summary>
        /// 从用户获取命令。
        /// </summary>
        /// <param name="moveNumber">
        /// 当前移动的编号。
        /// </param>
        /// <param name="positions">
        /// 代码位置的数量。
        /// </param>
        /// <param name="colors">
        /// 代码颜色的最大数量。
        /// </param>
        /// <returns>
        /// 输入的命令和猜测（如果适用）。
        # 定义一个名为 GetCommand 的函数，接受三个参数：moveNumber, positions, colors
        def GetCommand(moveNumber, positions, colors):
            # 进入一个无限循环
            while True:
                # 调用 View 模块的 PromptGuess 函数，传入 moveNumber 参数
                View.PromptGuess(moveNumber)
                # 从控制台读取用户输入
                input = Console.ReadLine()
                # 如果用户输入为 null，则退出程序
                if input is None:
                    Environment.Exit(0)
                # 将用户输入转换为大写形式
                switch (input.ToUpperInvariant())
                # 根据用户输入的不同情况进行处理
                switch (input.ToUpperInvariant())
                    # 如果用户输入为 "BOARD"，则返回一个元组 (Command.ShowBoard, None)
                    case "BOARD":
                        return (Command.ShowBoard, None)
                    # 如果用户输入为 "QUIT"，则返回一个元组 (Command.Quit, None)
                    case "QUIT":
                        return (Command.Quit, None)
                    # 如果用户输入不是 "BOARD" 或 "QUIT"，且输入长度不等于 positions，则调用 View 模块的 NotifyBadNumberOfPositions 函数
                    default:
                        if len(input) != positions:
                            View.NotifyBadNumberOfPositions()
                        else
                        if (input.FindFirstIndex(c => !TranslateColor(c).HasValue) is int invalidPosition)
                            View.NotifyInvalidColor(input[invalidPosition]);
                        else
                            return (Command.MakeGuess, new Code(input.Select(c => TranslateColor(c)!.Value)));

                        break;
                }
            }
        }
```
这段代码是一个条件语句，首先检查输入是否包含无效的颜色，如果有，则通知视图显示无效颜色，并且不执行后续操作；如果没有无效颜色，则返回一个包含转换后的颜色值的元组(Command.MakeGuess, new Code(input.Select(c => TranslateColor(c)!.Value)))。

```
        /// <summary>
        /// Waits until the user indicates that he or she is ready to continue.
        /// </summary>
        public static void WaitUntilReady()
        {
            View.PromptReady();
            var input = Console.ReadLine();
            if (input is null)
                Environment.Exit(0);
```
这段代码定义了一个名为WaitUntilReady的方法，该方法等待用户指示准备好继续。首先调用视图的PromptReady方法，然后读取用户的输入。如果输入为null，则退出程序。
        }

        /// <summary>
        /// Gets the number of blacks and whites for the given code from the
        /// user.
        /// </summary>
        public static (int blacks, int whites) GetBlacksWhites(Code code)
        {
            // 无限循环，直到用户输入正确的值
            while (true)
            {
                // 提示用户输入黑白棋的数量
                View.PromptBlacksWhites(code);

                // 读取用户输入
                var input = Console.ReadLine();
                // 如果用户输入为空，则退出程序
                if (input is null)
                    Environment.Exit(0);

                // 将用户输入按逗号分割成两部分
                var parts = input.Split(',');

                // 如果分割后的部分数量不等于2，则提示用户重新输入
                if (parts.Length != 2)
                    View.PromptTwoValues();
                else
                if (!Int32.TryParse(parts[0], out var blacks) || !Int32.TryParse(parts[1], out var whites))
                    View.PromptValidInteger();  // 如果无法将输入的字符串转换为整数，提示用户输入有效的整数
                else
                    return (blacks, whites);  // 如果能成功转换为整数，返回转换后的整数值
            }
        }

        /// <summary>
        /// 从用户获取一个整数值。
        /// </summary>
        private static int GetInteger(Action prompt)
        {
            while (true)
            {
                prompt();  // 调用传入的提示方法，提示用户输入整数

                var input = Console.ReadLine();  // 从控制台读取用户输入的字符串
                if (input is null)  // 如果用户输入为null
                    Environment.Exit(0);  // 退出程序
                if (Int32.TryParse(input, out var result))
                    return result;  // 如果输入可以成功转换为整数，则返回转换后的整数
                else
                    View.PromptValidInteger();  // 如果输入无法转换为整数，则提示用户输入有效的整数
            }
        }

        /// <summary>
        /// Translates the given character into the corresponding color.
        /// </summary>
        private static int? TranslateColor(char c) =>
            ColorsByKey.TryGetValue(c, out var index) ? index : null;  // 将给定的字符转换为对应的颜色索引，如果找到对应的颜色索引则返回索引，否则返回空值
    }
}
```