# `d:/src/tocomm/basic-computer-games\44_Hangman\csharp\Program.cs`

```
// 导入所需的命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

// 命名空间 Hangman
namespace Hangman
{
    /// <summary>
    /// C# version of the game "Hangman" from the book BASIC Computer Games.
    /// </summary>
    // 静态类 Program
    static class Program
    {
        // 主函数
        static void Main()
        {
            // 输出游戏标题
            Console.WriteLine(Tab(32) + "HANGMAN");
            // 输出游戏信息
            Console.WriteLine(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            // 输出空行
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            // 调用主循环函数
            MainLoop();
            Console.WriteLine();  // 输出空行
            Console.WriteLine("IT'S BEEN FUN!  BYE FOR NOW.");  // 输出结束游戏的提示信息
        }

        static void MainLoop()
        {
            var words = GetWords();  // 调用GetWords函数获取单词列表
            var stillPlaying = true;  // 设置游戏状态为继续进行

            while (stillPlaying)  // 当游戏状态为继续进行时
            {
                if (words.Count == 0)  // 如果单词列表为空
                {
                    Console.WriteLine("YOU DID ALL THE WORDS!!");  // 输出提示信息
                    break;  // 跳出循环
                }

                // 生成一个从0到单词数量减一的随机数（C#数组是从0开始的）
                var rnd = new Random();  // 创建Random对象
                var randomNumber = rnd.Next(words.Count - 1);  // 生成随机数
                // 从单词列表中随机选择一个单词并将其从列表中移除
                var word = words[randomNumber];
                words.Remove(word);

                GameLoop(word);

                // 游戏结束，询问玩家是否想要另一个单词
                Console.WriteLine("WANT ANOTHER WORD? ");
                var response = Console.ReadLine();
                if (response == null || response.ToUpper() != "YES")
                {
                    stillPlaying = false;   // 如果玩家没有回答“是”，则退出循环
                }
            }
        }

        static void GameLoop(string word)
        {
            var graphic = new Graphic();
            // 初始化错误猜测次数
            var wrongGuesses = 0;
            // 初始化猜测次数
            var numberOfGuesses = 0;
            // 初始化已使用的字母列表
            var usedLetters = new List<char>();

            // 初始化用户可见的单词，初始时全部为破折号
            var displayedWord = new char[word.Length];
            for (var i = 0; i < word.Length; i++)
            {
                displayedWord[i] = '-';
            }

            // 初始化游戏状态为继续进行
            var stillPlaying = true;
            // 当游戏仍在进行时
            while (stillPlaying)
            {
                // 从玩家获取猜测的字母
                var guess = GetLetterFromPlayer(displayedWord, usedLetters);
                // 将猜测的字母添加到已使用的字母列表中
                usedLetters.Add(guess);
                // 增加猜测次数
                numberOfGuesses++;
                // 初始化正确猜测的字母数量
                var correctLetterCount = 0;
                // 检查玩家猜测的字母是否在单词中出现
                for(var i = 0; i < word.Length; i++)
                {
                    // 检查玩家猜测的字母是否在单词中出现，如果出现则更新显示的单词
                    if (word[i] == guess)
                    {
                        correctLetterCount++; // 记录猜对的字母数量
                        displayedWord[i] = guess; // 更新显示的单词
                    }
                }

                // 如果没有猜对的字母，则增加错误猜测次数并绘制Hangman图形
                if (correctLetterCount == 0)
                {
                    // 错误的猜测
                    Console.WriteLine("SORRY, THAT LETTER ISN'T IN THE WORD."); // 提示玩家猜错
                    wrongGuesses++; // 增加错误猜测次数
                    DrawBody(graphic, wrongGuesses); // 绘制Hangman图形
                    if (wrongGuesses == 10)
                    {
                        // 玩家已经用尽所有猜测次数，结束游戏循环
                        Console.WriteLine($"SORRY, YOU LOSE.  THE WORD WAS {word}"); // 提示玩家已经输掉游戏并显示正确的单词
                        Console.Write("YOU MISSED THAT ONE.  DO YOU "); // 提示玩家已经输掉游戏
                        stillPlaying = false; // 设置游戏结束标志为true
                    }
                }
                else
                {
                    // 玩家猜对了一个字母。让我们看看单词中是否还有未猜的字母。
                    if (displayedWord.Contains('-'))
                    {
                        Console.WriteLine(displayedWord);

                        // 给玩家一个猜测整个单词的机会。
                        var wordGuess = GetWordFromPlayer();
                        if (word == wordGuess)
                        {
                            // 玩家找到了单词。标记为找到。
                            Console.WriteLine("YOU FOUND THE WORD!");
                            stillPlaying = false;   // 退出游戏循环。
                        }
                        else
                        {
                            // 玩家没有猜对单词。继续游戏循环。
// 在控制台打印错误提示信息
Console.WriteLine("WRONG.  TRY ANOTHER LETTER.");

// 如果玩家猜测的字母已经全部猜中
Console.WriteLine("YOU FOUND THE WORD!");
stillPlaying = false;   // 退出游戏循环

// 游戏循环结束
// 显示当前猜测单词的状态和所有已经猜过的字母，并从玩家那里获取新的猜测
/// <summary>
/// 显示当前猜测单词的状态和所有已经猜过的字母，并从玩家那里获取新的猜测
/// </summary>
/// <param name="displayedWord">表示当前猜测单词状态的字符数组</param>
/// <param name="usedLetters">表示到目前为止已经猜过的所有字母的字符列表</param>
/// <returns>玩家刚刚输入的猜测字母</returns>
private static char GetLetterFromPlayer(char[] displayedWord, List<char> usedLetters)
            while (true)    // 无限循环，除非玩家输入一个未使用过的字母。
            {
                Console.WriteLine();    // 输出空行
                Console.WriteLine(displayedWord);    // 输出当前猜测的单词状态
                Console.WriteLine();    // 输出空行
                Console.WriteLine();    // 输出空行
                Console.WriteLine("HERE ARE THE LETTERS YOU USED:");    // 输出提示信息
                for (var i = 0; i < usedLetters.Count; i++)    // 遍历已使用的字母列表
                {
                    Console.Write(usedLetters[i]);    // 输出已使用的字母

                    // 如果不是最后一个字母，输出逗号
                    if (i != usedLetters.Count - 1)
                    {
                        Console.Write(",");
                    }
                }

                Console.WriteLine();    // 输出空行
                Console.WriteLine("WHAT IS YOUR GUESS?");  // 输出提示信息，要求玩家输入猜测的字母
                var guess = char.ToUpper(Console.ReadKey().KeyChar);  // 从控制台读取玩家输入的字符，并转换为大写
                Console.WriteLine();  // 输出空行

                if (usedLetters.Contains(guess))  // 检查已经猜过的字母列表中是否包含当前猜测的字母
                {
                    // After this the loop will continue.  // 如果包含，则输出提示信息，继续循环
                    Console.WriteLine("YOU GUESSED THAT LETTER BEFORE!");
                }
                else
                {
                    // Break out of the loop by returning guessed letter.  // 如果不包含，则返回猜测的字母，结束循环
                    return guess;
                }
            }
        }

        /// <summary>
        /// Gets a word guess from the player.  // 从玩家获取一个单词的猜测
        /// <returns>The guessed word.</returns>
        private static string GetWordFromPlayer()
        {
            while (true)    // Infinite loop, unless the player enters something.
            {
                Console.WriteLine("WHAT IS YOUR GUESS FOR THE WORD? ");  // Prompt the player to enter their guess for the word
                var guess = Console.ReadLine();  // Read the player's input
                if (guess != null)  // Check if the input is not null
                {
                    return guess.ToUpper();  // Return the input in uppercase
                }
            }
        }

        /// <summary>
        /// Draw body after wrong guess.
        /// </summary>
        /// <param name="graphic">The instance of the Graphic class being used.</param>
        /// <param name="wrongGuesses">Number of wrong guesses.</param>
        private static void DrawBody(Graphic graphic, int wrongGuesses)
        {
            # 根据错误的猜测次数进行不同的操作
            switch (wrongGuesses)
                    {
                        # 如果错误次数为1，打印信息并添加头部图形
                        case 1:
                            Console.WriteLine("FIRST, WE DRAW A HEAD.");
                            graphic.AddHead();
                            break;
                        # 如果错误次数为2，打印信息并添加身体图形
                        case 2:
                            Console.WriteLine("NOW WE DRAW A BODY.");
                            graphic.AddBody();
                            break;
                        # 如果错误次数为3，打印信息并添加右臂图形
                        case 3:
                            Console.WriteLine("NEXT WE DRAW AN ARM.");
                            graphic.AddRightArm();
                            break;
                        # 如果错误次数为4，打印信息并添加左臂图形
                        case 4:
                            Console.WriteLine("THIS TIME IT'S THE OTHER ARM.");
                            graphic.AddLeftArm();
                            break;
                        # 如果错误次数为5，进行其他操作
# 打印消息到控制台，表示现在要绘制正确的腿
Console.WriteLine("NOW, LET'S DRAW THE RIGHT LEG.")
# 调用图形对象的添加右腿方法
graphic.AddRightLeg()
# 退出 switch 语句
break
# 打印消息到控制台，表示这次要绘制左腿
Console.WriteLine("THIS TIME WE DRAW THE LEFT LEG.")
# 调用图形对象的添加左腿方法
graphic.AddLeftLeg()
# 退出 switch 语句
break
# 打印消息到控制台，表示现在要举起一只手
Console.WriteLine("NOW WE PUT UP A HAND.")
# 调用图形对象的添加右手方法
graphic.AddRightHand()
# 退出 switch 语句
break
# 打印消息到控制台，表示接下来要举起另一只手
Console.WriteLine("NEXT THE OTHER HAND.")
# 调用图形对象的添加左手方法
graphic.AddLeftHand()
# 退出 switch 语句
break
# 打印消息到控制台，表示现在要绘制一只脚
Console.WriteLine("NOW WE DRAW ONE FOOT.")
# 调用图形对象的添加右脚方法
graphic.AddRightFoot()
# 退出 switch 语句
break
                            Console.WriteLine("HERE'S THE OTHER FOOT -- YOU'RE HUNG!!");
                            // 打印消息到控制台
                            graphic.AddLeftFoot();
                            // 添加左脚的图形
                            break;
                    }
                    // 打印图形
                    graphic.Print();
        }

        /// <summary>
        /// Get a list of words to use in the game.
        /// </summary>
        /// <returns>List of strings.</returns>
        // 获取游戏中要使用的单词列表
        private static List<string> GetWords() => new()
        {
            "GUM",
            "SIN",
            "FOR",
            "CRY",
            "LUG",
            "BYE",
            "FLY",
# 创建一个包含字符串的列表
words = [
    "UGLY",
    "EACH",
    "FROM",
    "WORK",
    "TALK",
    "WITH",
    "SELF",
    "PIZZA",
    "THING",
    "FEIGN",
    "FIEND",
    "ELBOW",
    "FAULT",
    "DIRTY",
    "BUDGET",
    "SPIRIT",
    "QUAINT",
    "MAIDEN",
    "ESCORT",
    "PICKAX"
]
# 创建一个包含字符串的列表
words = [
    "EXAMPLE",
    "TENSION",
    "QUININE",
    "KIDNEY",
    "REPLICA",
    "SLEEPER",
    "TRIANGLE",
    "KANGAROO",
    "MAHOGANY",
    "SERGEANT",
    "SEQUENCE",
    "MOUSTACHE",
    "DANGEROUS",
    "SCIENTIST",
    "DIFFERENT",
    "QUIESCENT",
    "MAGISTRATE",
    "ERRONEOUSLY",
    "LOUDSPEAKER",
    "PHYTOTOXIC",
]
            "MATRIMONIAL",  // 添加字符串 "MATRIMONIAL" 到字符串数组中
            "PARASYMPATHOMIMETIC",  // 添加字符串 "PARASYMPATHOMIMETIC" 到字符串数组中
            "THIGMOTROPISM"  // 添加字符串 "THIGMOTROPISM" 到字符串数组中
        };

        /// <summary>
        /// Leave a number of spaces empty.
        /// </summary>
        /// <param name="length">Number of spaces.</param>
        /// <returns>The result string.</returns>
        private static string Tab(int length) => new string(' ', length);  // 创建一个由指定数量空格组成的字符串并返回
    }
}
```