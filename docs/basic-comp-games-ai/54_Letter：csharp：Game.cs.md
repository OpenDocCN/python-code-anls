# `54_Letter\csharp\Game.cs`

```
        /// <summary>
        /// Maximum number of guesses.
        /// Note the program doesn't enforce this - it just displays a message if this is exceeded.
        /// </summary>
        private const int MaximumGuesses = 5;
        // 定义最大猜测次数为5

        /// <summary>
        /// Main game loop.
        /// </summary>
        public static void Play()
        {
            DisplayIntroductionText();
            // 调用显示游戏介绍的函数

            // Keep playing forever, or until the user quits.
            while (true)
            {
                // 游戏主循环，一直进行直到用户退出
                PlayRound();
            }
        }

        /// <summary>
        /// Play a single round.
        /// </summary>
        internal static void PlayRound()
        {
            // 创建一个新的游戏状态对象
            var gameState = new GameState();
            // 显示游戏回合介绍
            DisplayRoundIntroduction();

            // 初始化输入字符为一个不在A-Z范围内的字符
            char letterInput = '\0'; 
            // 当输入字符不等于游戏状态对象的字母时，循环执行猜字母的操作
            while (letterInput != gameState.Letter)
            {
                // 从键盘获取输入的字符
                letterInput = GetCharacterFromKeyboard();
                // 增加猜测次数
                gameState.GuessesSoFar++;
                // 显示猜测结果
                DisplayGuessResult(gameState.Letter, letterInput);
            }
            // 显示成功消息
            DisplaySuccessMessage(gameState);
        }

        /// <summary>
        /// Display an introduction when the game loads.
        /// </summary>
        internal static void DisplayIntroductionText()
        {
            // 设置控制台前景色为黄色
            Console.ForegroundColor = ConsoleColor.Yellow;
            // 打印游戏介绍信息
            Console.WriteLine("LETTER");
            Console.WriteLine("Creative Computing, Morristown, New Jersey.");
            Console.WriteLine("");

            // 设置控制台前景色为深绿色
            Console.ForegroundColor = ConsoleColor.DarkGreen;
            // 打印游戏介绍信息
            Console.WriteLine("Letter Guessing Game");
            Console.WriteLine("I'll think of a letter of the alphabet, A to Z.");
            Console.WriteLine("Try to guess my letter and I'll give you clues");
            Console.WriteLine("as to how close you're getting to my letter.");
            Console.WriteLine("");

            // 重置控制台颜色
            Console.ResetColor();
        }

        /// <summary>
        /// Display introductionary text for each round.
        /// </summary>
        internal static void DisplayRoundIntroduction()
        {
            # 设置控制台前景色为黄色
            Console.ForegroundColor = ConsoleColor.Yellow;
            # 打印提示文本
            Console.WriteLine("O.K., I have a letter. Start guessing.");

            # 重置控制台颜色
            Console.ResetColor();
        }

        /// <summary>
        /// Display text depending whether the guess is lower or higher.
        /// </summary>
        internal static void DisplayGuessResult(char letterToGuess, char letterInput)
        {
            # 设置控制台背景色为白色，前景色为黑色
            Console.BackgroundColor = ConsoleColor.White;
            Console.ForegroundColor = ConsoleColor.Black;
            Console.Write(" " + letterInput + " ");  // 在控制台输出用户输入的字母

            Console.ResetColor();  // 重置控制台前景色和背景色
            Console.ForegroundColor = ConsoleColor.Gray;  // 设置控制台前景色为灰色
            Console.Write(" ");  // 在控制台输出空格
            if (letterInput != letterToGuess)  // 如果用户输入的字母不等于要猜的字母
            {
                if (letterInput > letterToGuess)  // 如果用户输入的字母大于要猜的字母
                {
                    Console.WriteLine("Too high. Try a lower letter");  // 在控制台输出提示信息，要求用户尝试更小的字母
                }
                else  // 如果用户输入的字母小于要猜的字母
                {
                    Console.WriteLine("Too low. Try a higher letter");  // 在控制台输出提示信息，要求用户尝试更大的字母
                }
            }
            Console.ResetColor();  // 重置控制台前景色和背景色
        }

        /// <summary>  // 方法的注释，描述方法的作用和用法
        /// <summary>
        /// 显示成功消息，以及猜测的次数。
        /// </summary>
        internal static void DisplaySuccessMessage(GameState gameState)
        {
            // 设置控制台前景色为绿色
            Console.ForegroundColor = ConsoleColor.Green;
            // 打印猜测次数的消息
            Console.WriteLine($"You got it in {gameState.GuessesSoFar} guesses!!");
            // 如果猜测次数超过最大次数，则打印警告消息
            if (gameState.GuessesSoFar > MaximumGuesses)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"But it shouldn't take more than {MaximumGuesses} guesses!");
            }
            // 如果猜测次数未超过最大次数，则打印祝贺消息
            else
            {
                Console.WriteLine("Good job !!!!!");
            }
            // 设置控制台前景色为黄色
            Console.ForegroundColor = ConsoleColor.Yellow;
            // 打印空行
            Console.WriteLine("");
            // 打印再玩一次的提示消息
            Console.WriteLine("Let's play again.....");

            // 重置控制台颜色
            Console.ResetColor();
/// <summary>
/// 从键盘获取有效输入：必须是一个字母字符。如果需要，转换为大写。
/// </summary>
internal static char GetCharacterFromKeyboard()
{
    char letterInput; // 声明一个字符变量 letterInput
    do
    {
        var keyPressed = Console.ReadKey(true); // 从控制台获取用户按下的键
        letterInput = Char.ToUpper(keyPressed.KeyChar); // 立即转换为大写
    } while (!Char.IsLetter(letterInput)); // 如果输入不是字母，则等待用户按下另一个字母键
    return letterInput; // 返回获取到的字母字符
}
```