# `d:/src/tocomm/basic-computer-games\96_Word\csharp\Program.cs`

```
        // 在这里列出了可能被选为获胜单词的潜在单词列表。
        private string[] words = { "DINKY", "SMOKE", "WATER", "GRASS", "TRAIN", "MIGHT", "FIRST",
         "CANDY", "CHAMP", "WOULD", "CLUMP", "DOPEY" };

        /// <summary>
        /// 输出游戏的说明。
        /// </summary>
        private void intro()
        {
            // 输出游戏标题
            Console.WriteLine("WORD".PadLeft(37));
            // 输出游戏信息
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".PadLeft(59));
            // 输出猜词游戏的提示信息
            Console.WriteLine("I am thinking of a word -- you guess it. I will give you");
            Console.WriteLine("clues to help you get it. Good luck!!");
        }

        /// <summary>
        /// 允许用户输入猜测的单词，并对这些猜测进行基本验证
        /// </summary>
        /// <returns>用户输入的猜测</returns>
        private string get_guess()
        {
            string guess = "";

            while (guess.Length == 0)
            {
                // 提示用户猜一个五个字母的单词
                Console.WriteLine($"{Environment.NewLine}Guess a five letter word. ");
                guess = Console.ReadLine().ToUpper();

                // 对用户输入的猜测进行验证
                if ((guess.Length != 5) || (guess.Equals("?")) || (!guess.All(char.IsLetter)))
        {
            // 初始化猜测值为空字符串
            guess = "";
            // 输出提示信息，要求用户重新开始猜测
            Console.WriteLine("You must guess a five letter word. Start again.");
        }
    }

    // 返回猜测值
    return guess;
}

/// <summary>
/// 检查用户的猜测与目标单词的匹配情况，捕获两者之间匹配的任何字母以及正确的特定字母
/// </summary>
/// <param name="guess">用户的猜测</param>
/// <param name="target">“获胜”的单词</param>
/// <param name="progress">显示已经猜过的特定字母的字符串</param>
/// <returns>显示猜测与目标之间字符匹配数量的整数值</returns>
private int check_guess(string guess, string target, StringBuilder progress)
            // 遍历猜测的每个字母，看哪些字母与目标单词匹配。
            // 对于每个匹配的位置，更新进度以反映猜测
            int matches = 0; // 匹配的数量
            string common_letters = ""; // 匹配的字母

            for (int ctr = 0; ctr < 5; ctr++) // 循环5次，检查猜测的每个位置
            {
                // 首先查看这个字母是否出现在目标单词中，如果是，则添加到common_letters列表中
                if (target.Contains(guess[ctr]))
                {
                    common_letters.Append(guess[ctr]);
                }
                // 然后查看这个特定的字母是否与目标中的相同位置匹配，如果是，则更新进度追踪器
                if (guess[ctr].Equals(target[ctr]))
                {
                    progress[ctr] = guess[ctr];  // 将猜测的单词中与目标单词相同位置的字母赋值给进度数组
                    matches++;  // 匹配数加一
                }
            }

            Console.WriteLine($"There were {matches} matches and the common letters were... {common_letters}");  // 输出匹配数和共同的字母
            Console.WriteLine($"From the exact letter matches, you know......... {progress}");  // 输出根据完全匹配的字母得到的进度
            return matches;  // 返回匹配数
        }

        /// <summary>
        /// This plays one full game.
        /// </summary>
        private void play_game()
        {
            string guess_word, target_word;  // 定义猜测单词和目标单词
            StringBuilder guess_progress = new StringBuilder("-----");  // 初始化猜测进度为"-----"
            Random rand = new Random();  // 创建随机数生成器
            int count = 0;  // 初始化计数器为0
            Console.WriteLine("You are starting a new game..."); // 输出消息，表示开始新游戏

            // 从单词列表中随机选择一个单词
            target_word = words[rand.Next(words.Length)];

            // 无限循环，直到满足结束游戏的条件之一
            while (true)
            {
                // 请求用户输入猜测的单词
                guess_word = get_guess();
                count++; // 计数加一，记录猜测次数

                // 如果用户输入问号，则告诉他们答案并结束游戏
                if (guess_word.Equals("?"))
                {
                    Console.WriteLine($"The secret word is {target_word}"); // 输出答案
                    return; // 结束游戏
                }
                // 否则，检查猜测的单词是否与目标单词匹配，并记录进度
                if (check_guess(guess_word, target_word, guess_progress) == 0)
                {
                    Console.WriteLine("如果你放弃了，下一次猜测输入'?'。");
                }

                // 一旦他们猜对了单词，结束游戏。
                if (guess_progress.Equals(guess_word))
                {
                    Console.WriteLine($"你猜对了单词。用了 {count} 次猜测！");
                    return;
                }
            }
        }

        /// <summary>
        /// 类的主要入口点 - 保持游戏进行，直到用户决定退出。
        /// </summary>
        public void play()
        {
            // 调用intro()方法，显示游戏介绍
            intro();

            // 设置一个布尔变量，用于控制是否继续游戏
            bool keep_playing = true;

            // 当keep_playing为true时，循环进行游戏
            while (keep_playing)
            {
                // 调用play_game()方法，开始游戏
                play_game();
                // 提示用户是否想再玩一次
                Console.WriteLine($"{Environment.NewLine}Want to play again? ");
                // 读取用户输入，判断是否想再玩一次，并更新keep_playing变量
                keep_playing = Console.ReadLine().StartsWith("y", StringComparison.CurrentCultureIgnoreCase);
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 创建Word类的实例，并调用play()方法开始游戏
            new Word().play();
抱歉，给定的代码片段不完整，无法为其添加注释。
```