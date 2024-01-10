# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayConsole\ConsoleUserInterface.cs`

```
// 实现了 IUserInterface 接口，通过控制台进行输入输出
internal class ConsoleUserInterface : BombsAwayGame.IUserInterface
{
    // 将消息写入控制台
    public void Output(string message)
    {
        Console.WriteLine(message);
    }

    // 显示带有索引的选项，允许用户通过索引选择
    public int Choose(string message, IList<string> choices)
    {
        // 为选项添加索引
        IEnumerable<string> choicesWithIndexes = choices.Select((choice, index) => $"{choice}({index + 1})");
        string choiceText = string.Join(", ", choicesWithIndexes);
        Output($"{message} -- {choiceText}");

        // 将选项转换为对应的控制台键
        ISet<ConsoleKey> allowedKeys = ConsoleKeysFromList(choices);
        ConsoleKey? choice;
        do
        {
            // 读取用户输入的选择
            choice = ReadChoice(allowedKeys);
            if (choice is null)
            {
                Output("TRY AGAIN...");
            }
        }
        while (choice is null);

        // 返回用户选择的索引
        return ListIndexFromConsoleKey(choice.Value);
    }

    // 将给定列表转换为对应的 ConsoleKey。这将生成将第一个元素映射到 ConsoleKey.D1，第二个元素映射到 ConsoleKey.D2，依此类推，直到列表的最后一个元素。
    private ISet<ConsoleKey> ConsoleKeysFromList(IList<string> list)
    {
        // 创建一个整数序列，从 D1 开始，数量为列表的长度
        IEnumerable<int> indexes = Enumerable.Range((int)ConsoleKey.D1, list.Count);
        // 将整数序列转换为 ConsoleKey 类型的哈希集合，并返回
        return new HashSet<ConsoleKey>(indexes.Cast<ConsoleKey>());
    }

    /// <summary>
    /// 将给定的控制台键转换为其列表索引的等价值。这假设键是由 <see cref="ConsoleKeysFromList(IList{string})"/> 生成的
    /// </summary>
    /// <param name="key">要转换为其列表索引等价值的键。</param>
    /// <returns>键的列表索引等价值。</returns>
    private int ListIndexFromConsoleKey(ConsoleKey key)
    {
        // 返回键值减去 D1 的结果
        return key - ConsoleKey.D1;
    }

    /// <summary>
    /// 从控制台读取一个键，并在其在允许的键集合中时返回它。
    /// </summary>
    /// <param name="allowedKeys">允许的键。</param>
    /// <returns>从 <see cref="Console"/> 读取的键，如果它在 <paramref name="allowedKeys"/> 中；否则返回 null。</returns>
    private ConsoleKey? ReadChoice(ISet<ConsoleKey> allowedKeys)
    {
        // 读取控制台键信息
        ConsoleKeyInfo keyInfo = ReadKey();
        // 如果允许的键集合包含读取的键，则返回该键；否则返回 null
        return allowedKeys.Contains(keyInfo.Key) ? keyInfo.Key : null;
    }

    /// <summary>
    /// 从 <see cref="Console"/> 读取键。
    /// </summary>
    /// <returns>从 <see cref="Console"/> 读取的键。</returns>
    private ConsoleKeyInfo ReadKey()
    {
        // 从控制台读取键信息，不拦截按键
        ConsoleKeyInfo result = Console.ReadKey(intercept: false);
        // 在控制台上写入一个空行，以便显示的键在自己的一行上
        Console.WriteLine();
        return result;
    }

    /// <summary>
    /// 允许用户从 <see cref="Console"/> 选择 'Y' 或 'N'。
    /// </summary>
    /// <param name="message">要显示的消息。</param>
    /// <returns>如果用户选择 'Y' 则返回 true，如果用户选择 'N' 则返回 false。</returns>
    public bool ChooseYesOrNo(string message)
    {
        // 输出消息到控制台
        Output(message);
        // 定义控制台按键选择
        ConsoleKey? choice;
        // 循环直到用户输入有效的选择
        do
        {
            // 读取用户输入的选择
            choice = ReadChoice(new HashSet<ConsoleKey>(new[] { ConsoleKey.Y, ConsoleKey.N }));
            // 如果选择为空
            if (choice is null)
            {
                // 输出提示信息
                Output("ENTER Y OR N");
            }
        }
        // 当选择为空时继续循环
        while (choice is null);

        // 返回用户选择是否为 Y
        return choice.Value == ConsoleKey.Y;
    }

    /// <summary>
    /// Get integer by reading a line from <see cref="Console"/>.
    /// </summary>
    /// <returns>Integer read from <see cref="Console"/>.</returns>
    // 从控制台读取一行并返回整数
    public int InputInteger()
    {
        // 定义结果是否有效的标志
        bool resultIsValid;
        // 定义结果
        int result;
        // 循环直到用户输入有效的整数
        do
        {
            // 从控制台读取一行文本
            string? integerText = Console.ReadLine();
            // 尝试将文本转换为整数，并将结果存储到 result 中
            resultIsValid = int.TryParse(integerText, out result);
            // 如果结果无效
            if (!resultIsValid)
            {
                // 输出提示信息
                Output("PLEASE ENTER A NUMBER");
            }
        }
        // 当结果无效时继续循环
        while (!resultIsValid);

        // 返回有效的整数结果
        return result;
    }
# 闭合前面的函数定义
```