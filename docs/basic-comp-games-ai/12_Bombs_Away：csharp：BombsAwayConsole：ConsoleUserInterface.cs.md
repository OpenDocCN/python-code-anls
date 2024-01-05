# `d:/src/tocomm/basic-computer-games\12_Bombs_Away\csharp\BombsAwayConsole\ConsoleUserInterface.cs`

```
/// <param name="choices">List of choices to display.</param>
    public int Choose(string message, List<string> choices)
    {
        // 输出消息到控制台
        Console.WriteLine(message);
        // 遍历选择列表，显示每个选项和对应的索引
        for (int i = 0; i < choices.Count; i++)
        {
            Console.WriteLine($"{i+1}. {choices[i]}");
        }
        // 读取用户输入的选择
        int chosenIndex;
        while (!int.TryParse(Console.ReadLine(), out chosenIndex) || chosenIndex < 1 || chosenIndex > choices.Count)
        {
            // 如果用户输入无效，提示用户重新输入
            Console.WriteLine("Invalid input. Please choose a valid index.");
        }
        // 返回用户选择的索引
        return chosenIndex - 1;
    }
}
    # 定义一个方法，用于在控制台上显示选择项，并返回用户选择的选项索引
    def Choose(message, choices):
        # 将选择项和索引组合成带索引的选择项列表
        choicesWithIndexes = [f"{choice}({index + 1})" for index, choice in enumerate(choices)]
        # 将带索引的选择项列表转换成字符串
        choiceText = ", ".join(choicesWithIndexes)
        # 在控制台上输出消息和选择项
        Output(f"{message} -- {choiceText}")

        # 将选择项转换成控制台键集合
        allowedKeys = ConsoleKeysFromList(choices)
        choice = None
        # 循环读取用户输入的控制台键，直到用户输入有效的选择项
        while choice is None:
            choice = ReadChoice(allowedKeys)
            if choice is None:
                Output("TRY AGAIN...")
        
        # 返回用户选择的选项索引
        return int(choice)
        return ListIndexFromConsoleKey(choice.Value);
    }
    # 将给定列表转换为其对应的<see cref="ConsoleKey"/>。这将生成将第一个元素映射到<see cref="ConsoleKey.D1"/>，第二个元素映射到<see cref="ConsoleKey.D2"/>，依此类推，直到列表的最后一个元素。
    # <param name="list">要转换为<see cref="ConsoleKey"/>的列表。</param>
    # <returns>来自<paramref name="list"/>的<see cref="ConsoleKey"/>等效项。</returns>
    private ISet<ConsoleKey> ConsoleKeysFromList(IList<string> list)
    {
        # 生成从<see cref="ConsoleKey.D1"/>到列表长度的范围内的整数
        IEnumerable<int> indexes = Enumerable.Range((int)ConsoleKey.D1, list.Count);
        # 将整数转换为<ConsoleKey>类型，并将其存储在HashSet中
        return new HashSet<ConsoleKey>(indexes.Cast<ConsoleKey>());
    }

    # 将给定的控制台键转换为其列表索引等效项。这假设该键是从<see cref="ConsoleKeysFromList(IList{string})"/>生成的。
    /// <param name="key">Key to convert to its list index equivalent.</param>
    /// <returns>List index equivalent of key.</returns>
    private int ListIndexFromConsoleKey(ConsoleKey key)
    {
        // 将控制台键转换为列表索引的等价值
        return key - ConsoleKey.D1;
    }

    /// <summary>
    /// Read a key from the console and return it if it is in the given allowed keys.
    /// </summary>
    /// <param name="allowedKeys">Allowed keys.</param>
    /// <returns>Key read from <see cref="Console"/>, if it is in <paramref name="allowedKeys"/>; null otherwise./></returns>
    private ConsoleKey? ReadChoice(ISet<ConsoleKey> allowedKeys)
    {
        // 从控制台读取一个键，并且如果它在允许的键集合中，则返回它
        ConsoleKeyInfo keyInfo = ReadKey();
        return allowedKeys.Contains(keyInfo.Key) ? keyInfo.Key : null;
    }

    /// <summary>
    /// Read key from <see cref="Console"/>.
    /// </summary>
    /// <summary>
    /// 读取用户在控制台输入的按键信息
    /// </summary>
    /// <returns>从控制台读取的按键信息</returns>
    private ConsoleKeyInfo ReadKey()
    {
        ConsoleKeyInfo result = Console.ReadKey(intercept: false);
        // 在控制台输出一个空行，以便显示的按键信息单独显示在一行
        Console.WriteLine();
        return result;
    }

    /// <summary>
    /// 允许用户在控制台选择 'Y' 或 'N'
    /// </summary>
    /// <param name="message">要显示的消息</param>
    /// <returns>如果用户选择 'Y' 则返回 true，如果用户选择 'N' 则返回 false</returns>
    public bool ChooseYesOrNo(string message)
    {
        Output(message);
        ConsoleKey? choice;
        do
        {
            // 从包含 Y 和 N 的集合中读取用户的选择
            choice = ReadChoice(new HashSet<ConsoleKey>(new[] { ConsoleKey.Y, ConsoleKey.N }));
            // 如果用户的选择为空
            if (choice is null)
            {
                // 输出提示信息
                Output("ENTER Y OR N");
            }
        }
        // 当用户的选择为空时重复循环
        while (choice is null);

        // 返回用户选择是否为 Y
        return choice.Value == ConsoleKey.Y;
    }

    /// <summary>
    /// 从 <see cref="Console"/> 读取一行并获取整数值。
    /// </summary>
    /// <returns>从 <see cref="Console"/> 读取的整数值。</returns>
    public int InputInteger()
    {
        // 初始化结果是否有效的标志
        bool resultIsValid;
        // 初始化结果变量
        int result;
        do
        {
            // 从控制台读取用户输入的字符串
            string? integerText = Console.ReadLine();
            // 尝试将输入的字符串转换为整数，如果成功则将结果存入result中，返回true；否则返回false
            resultIsValid = int.TryParse(integerText, out result);
            // 如果转换失败，则输出提示信息
            if (!resultIsValid)
            {
                Output("PLEASE ENTER A NUMBER");
            }
        }
        // 当转换失败时继续循环，直到用户输入有效的整数
        while (!resultIsValid);

        // 返回转换成功的整数结果
        return result;
    }
}
```