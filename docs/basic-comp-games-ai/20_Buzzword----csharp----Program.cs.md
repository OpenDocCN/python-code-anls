# `basic-computer-games\20_Buzzword\csharp\Program.cs`

```

// 命名空间Buzzword
namespace Buzzword
{
    // 程序类
    class Program
    {
        /// <summary>
        /// 显示标题
        /// </summary>
        static void Header()
        {
            Console.WriteLine("Buzzword generator".PadLeft(26));
            Console.WriteLine("Creating Computing Morristown, New Jersey".PadLeft(15));
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        // 用户输入的可能键信息
        static string keys = "type a 'Y' for another phrase or 'N' to quit";

        /// <summary>
        /// 显示说明
        /// </summary>
        static void Instructions()
        {
            Console.WriteLine("This program prints highly acceptable phrases in\n"
            + "'educator-speak' that you can work into reports\n"
            + "and speeches. Whenever a question mark is printed,\n"
            + $"{keys}.");
            Console.WriteLine();
            Console.WriteLine();
            Console.Write("Here's the first phrase:");
        }

        // 单词数组
        static string[] Words = new[]
            { "ability", "basal", "behavioral", "child-centered",
            "differentiated", "discovery", "flexible", "heterogenous",
            "homogeneous", "manipulative", "modular", "tavistock",
            "individualized", "learning", "evaluative", "objective",
            "cognitive", "enrichment", "scheduling", "humanistic",
            "integrated", "non-graded", "training", "vertical age",
            "motivational", "creative", "grouping", "modification",
            "accountability", "process", "core curriculum", "algorithm",
            "performance", "reinforcement", "open classroom", "resource",
            "structure", "facility", "environment" };

        /// <summary>
        /// 将给定字符串的第一个字母大写
        /// </summary>
        /// <param name="input"></param>
        /// <returns>字符串</returns>
        static string Capitalize(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
                return string.Empty;

            return char.ToUpper(input[0]) + input[1..];
        }

        // 种子已经计算，以获得与原始效果相同的效果，至少在第一个短语中
        static readonly Random rnd = new Random(1486);

        /// <summary>
        /// 从Words数组中生成随机短语
        /// </summary>
        /// <returns>表示随机短语的字符串，其中第一个字母大写</returns>
        static string GeneratePhrase()
        {
            // 从0开始索引，因此必须减少生成的数字
            return $"{Capitalize(Words[rnd.Next(13)])} "
                + $"{Words[rnd.Next(13, 26)]} "
                + $"{Words[rnd.Next(26, 39)]}";
        }

        /// <summary>
        /// 处理用户输入。在错误输入时，显示有关有效键的信息
        /// </summary>
        /// <returns>如果用户按下'Y'，则返回True，如果按下'N'，则返回False</returns>
        static bool Decision()
        {
            while (true)
            {
                Console.Write("?");
                var answer = Console.ReadKey();
                if (answer.Key == ConsoleKey.Y)
                    return true;
                else if (answer.Key == ConsoleKey.N)
                    return false;
                else
                    Console.WriteLine($"\n{keys}");
            }
        }

        static void Main(string[] args)
        {
            Header();
            Instructions();

            while (true)
            {
                Console.WriteLine();
                Console.WriteLine(GeneratePhrase());
                Console.WriteLine();

                if (!Decision())
                    break;
            }

            Console.WriteLine("\nCome back when you need help with another report!");
        }
    }
}

```