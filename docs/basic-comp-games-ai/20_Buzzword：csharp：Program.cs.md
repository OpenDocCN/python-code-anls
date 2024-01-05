# `d:/src/tocomm/basic-computer-games\20_Buzzword\csharp\Program.cs`

```
// 引入 System 命名空间
using System;

namespace Buzzword
{
    class Program
    {
        /// <summary>
        /// 显示标题
        /// </summary>
        static void Header()
        {
            // 打印标题
            Console.WriteLine("Buzzword generator".PadLeft(26));
            // 打印创建信息
            Console.WriteLine("Creating Computing Morristown, New Jersey".PadLeft(15));
            // 打印空行
            Console.WriteLine();
            // 打印空行
            Console.WriteLine();
            // 打印空行
            Console.WriteLine();
        }

        // 用户输入提示信息
        static string keys = "type a 'Y' for another phrase or 'N' to quit";
        /// <summary>
        /// Displays instructions.
        /// </summary>
        static void Instructions()
        {
            // 打印程序的使用说明
            Console.WriteLine("This program prints highly acceptable phrases in\n"
            + "'educator-speak' that you can work into reports\n"
            + "and speeches. Whenever a question mark is printed,\n"
            + $"{keys}.");
            Console.WriteLine();
            Console.WriteLine();
            Console.Write("Here's the first phrase:");
        }

        // 定义一个字符串数组，包含一些教育术语
        static string[] Words = new[]
            { "ability", "basal", "behavioral", "child-centered",
            "differentiated", "discovery", "flexible", "heterogenous",
            "homogeneous", "manipulative", "modular", "tavistock",
            "individualized", "learning", "evaluative", "objective",
/// <summary>
/// 创建一个字符串数组，包含教育术语
/// </summary>
string[] educationTerms = { 
    "cognitive", "enrichment", "scheduling", "humanistic",
    "integrated", "non-graded", "training", "vertical age",
    "motivational", "creative", "grouping", "modification",
    "accountability", "process", "core curriculum", "algorithm",
    "performance", "reinforcement", "open classroom", "resource",
    "structure", "facility", "environment" };

/// <summary>
/// 将给定字符串的第一个字母大写
/// </summary>
/// <param name="input">要处理的字符串</param>
/// <returns>string</returns>
static string Capitalize(string input)
{
    if (string.IsNullOrWhiteSpace(input))  // 检查输入字符串是否为空或只包含空格
        return string.Empty;  // 如果是，返回空字符串

    return char.ToUpper(input[0]) + input[1..];  // 将输入字符串的第一个字母大写，然后返回
}
        // 使用种子值1486初始化随机数生成器，以获得与原始效果相同的效果，至少在第一个短语中
        static readonly Random rnd = new Random(1486);

        /// <summary>
        /// 从Words数组中可用的单词中生成随机短语。
        /// </summary>
        /// <returns>表示随机短语的字符串，其中第一个字母大写。</returns>
        static string GeneratePhrase()
        {
            // 从0开始索引，因此必须减少生成的数字
            return $"{Capitalize(Words[rnd.Next(13)])} "
                + $"{Words[rnd.Next(13, 26)]} "
                + $"{Words[rnd.Next(26, 39)]}";
        }

        /// <summary>
        /// 处理用户输入。在输入错误时，它会在无限循环中显示有关有效键的信息。
        /// </summary>
        /// <returns>True if user pressed 'Y', false if 'N'.</returns>
        static bool Decision()
        {
            // 创建一个函数，用于用户做出决定，返回用户按下的键对应的布尔值
            while (true)
            {
                // 提示用户输入
                Console.Write("?");
                // 读取用户输入的按键
                var answer = Console.ReadKey();
                // 如果用户按下 'Y' 键，返回 true
                if (answer.Key == ConsoleKey.Y)
                    return true;
                // 如果用户按下 'N' 键，返回 false
                else if (answer.Key == ConsoleKey.N)
                    return false;
                // 如果用户按下其他键，提示用户重新输入
                else
                    Console.WriteLine($"\n{keys}");
            }
        }

        static void Main(string[] args)
        {
            // 调用 Header 函数，显示程序的标题
            Header();
            // 调用 Instructions 函数，显示程序的使用说明
            Instructions();
# 进入一个无限循环，直到条件为假
while (true)
{
    # 打印空行
    Console.WriteLine();
    # 调用GeneratePhrase函数并打印结果
    Console.WriteLine(GeneratePhrase());
    # 打印空行
    Console.WriteLine();

    # 调用Decision函数，如果返回值为假则跳出循环
    if (!Decision())
        break;
}

# 打印结束语句
Console.WriteLine("\nCome back when you need help with another report!");
```