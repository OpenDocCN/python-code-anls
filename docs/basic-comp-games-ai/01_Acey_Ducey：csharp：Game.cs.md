# `d:/src/tocomm/basic-computer-games\01_Acey_Ducey\csharp\Game.cs`

```
// 命名空间 AceyDucey
namespace AceyDucey
{
    /// <summary>
    /// 实现游戏逻辑的主要类
    /// </summary>
    internal class Game
    {
        /// <summary>
        /// 我们的随机数生成器对象
        /// </summary>
        private Random Rnd { get; } = new Random();

        /// <summary>
        /// 一行下划线，我们将在每个回合之间打印，以在屏幕上将它们分开
        /// </summary>
        private string SeparatorLine { get; } = new string('_', 70);
```
```csharp
        /// <summary>
        /// A list of all the cards in the game
        /// </summary>
        private List<Card> Deck { get; set; }

        /// <summary>
        /// The player's hand
        /// </summary>
        private List<Card> PlayerHand { get; set; }

        /// <summary>
        /// The computer's hand
        /// </summary>
        private List<Card> ComputerHand { get; set; }

        /// <summary>
        /// The player's current bet
        /// </summary>
        private int Bet { get; set; }

        /// <summary>
        /// The player's current money
        /// </summary>
        private int Money { get; set; }

        /// <summary>
        /// The player's current score
        /// </summary>
        private int Score { get; set; }

        /// <summary>
        /// The computer's current score
        /// </summary>
        private int ComputerScore { get; set; }

        /// <summary>
        /// The constructor for the Game class
        /// </summary>
        public Game()
        {
            // Initialize the deck of cards
            Deck = new List<Card>();
            foreach (Suit s in Enum.GetValues(typeof(Suit)))
            {
                foreach (Rank r in Enum.GetValues(typeof(Rank)))
                {
                    Deck.Add(new Card(s, r));
                }
            }
        }
```
```csharp
        /// <summary>
        /// Shuffles the deck of cards
        /// </summary>
        private void ShuffleDeck()
        {
            for (int i = 0; i < Deck.Count; i++)
            {
                int r = i + Rnd.Next(Deck.Count - i);
                Card temp = Deck[r];
                Deck[r] = Deck[i];
                Deck[i] = temp;
            }
        }

        /// <summary>
        /// Deals two cards to the player and the computer
        /// </summary>
        private void DealCards()
        {
            PlayerHand = new List<Card> { Deck[0], Deck[1] };
            ComputerHand = new List<Card> { Deck[2], Deck[3] };
            Deck.RemoveRange(0, 4);
        }

        /// <summary>
        /// Prints the player's hand to the console
        /// </summary>
        private void PrintPlayerHand()
        {
            Console.WriteLine("Your hand:");
            foreach (Card c in PlayerHand)
            {
                Console.WriteLine(c);
            }
        }
```
```csharp
        /// <summary>
        /// Prints the computer's hand to the console, but only shows the first card
        /// </summary>
        private void PrintComputerHand()
        {
            Console.WriteLine("Computer's hand:");
            Console.WriteLine(ComputerHand[0]);
            Console.WriteLine("One card face down");
        }

        /// <summary>
        /// Asks the player for their bet
        /// </summary>
        private void GetPlayerBet()
        {
            Console.WriteLine($"You have {Money} dollars.");
            Console.Write("How much would you like to bet? ");
            Bet = int.Parse(Console.ReadLine());
        }

        /// <summary>
        /// Checks if the player has enough money to place the bet
        /// </summary>
        /// <returns>True if the player has enough money, false otherwise</returns>
        private bool EnoughMoney()
        {
            return Bet <= Money;
        }
```
```csharp
        /// <summary>
        /// Calculates the player's score
        /// </summary>
        private void CalculatePlayerScore()
        {
            Score = PlayerHand[0].Value + PlayerHand[1].Value;
        }

        /// <summary>
        /// Calculates the computer's score
        /// </summary>
        private void CalculateComputerScore()
        {
            ComputerScore = ComputerHand[0].Value + ComputerHand[1].Value;
        }

        /// <summary>
        /// Determines the winner of the game
        /// </summary>
        private void DetermineWinner()
        {
            if (Score > ComputerScore)
            {
                Console.WriteLine("You win!");
                Money += Bet;
            }
            else if (Score < ComputerScore)
            {
                Console.WriteLine("You lose!");
                Money -= Bet;
            }
            else
            {
                Console.WriteLine("It's a tie!");
            }
        }
    }
}
        /// <summary>
        /// Main game loop function. This will play the game endlessly until the player chooses to quit.
        /// </summary>
        internal void GameLoop()
        {
            // 首先向玩家显示游戏说明
            DisplayIntroText();

            // 我们将循环进行每一场游戏，直到玩家决定不再继续
            do
            {
                // 进行一场游戏！
                PlayGame();

                // 再玩一次？
            } while (TryAgain());
        }
/// <summary>
/// Play the game
/// </summary>
private void PlayGame()
{
    // 创建一个新的游戏状态对象
    GameState state = new GameState();

    // 清空控制台显示
    Console.Clear();
    // 循环直到玩家没有钱了
    do
    {
        // 进行下一轮游戏。传入状态对象，以便下一轮可以查看可用的钱，可以在玩家下注后更新它，并可以更新轮数。
        PlayTurn(state);

        // 继续循环直到玩家没有钱了
    } while (state.Money > 0);

    // 看起来玩家破产了，让他们知道他们的表现如何。
}
            // 设置控制台前景色为白色
            Console.ForegroundColor = ConsoleColor.White;
            // 输出空行
            Console.WriteLine("");
        }


        /// <summary>
        /// 执行一轮游戏
        /// </summary>
        /// <param name="state">当前游戏状态</param>
        private void PlayTurn(GameState state)
        {
            // 让玩家知道发生了什么
            Console.WriteLine("");
            Console.WriteLine(SeparatorLine);
            Console.WriteLine("");
            // 设置控制台前景色为白色
            Console.ForegroundColor = ConsoleColor.White;
            // 输出空行
            Console.WriteLine("");
            // 输出"这是你接下来的两张牌："
            Console.WriteLine("Here are your next two cards:");

            // 生成两张随机牌
            # 从牌堆中抽取第一张牌
            firstCard = GetCard()
            # 从牌堆中抽取第二张牌
            secondCard = GetCard()

            # 如果第二张牌比第一张牌小，交换它们的位置
            if (secondCard < firstCard):
                (firstCard, secondCard) = (secondCard, firstCard)

            # 显示抽取的两张牌
            DisplayCard(firstCard)
            DisplayCard(secondCard)

            # 询问玩家接下来的操作
            Console.ForegroundColor = ConsoleColor.White
            Console.WriteLine("")
            Console.Write("You currently have ")
            Console.ForegroundColor = ConsoleColor.Yellow
            Console.Write($"${state.Money}")
            Console.ForegroundColor = ConsoleColor.White
            // 提示玩家下注金额
            Console.WriteLine(". How much would you like to bet?");

            // 读取下注金额
            int betAmount = PlayTurn_GetBetAmount(state.Money);

            // 显示玩家输入的摘要
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("");
            Console.WriteLine($"You choose to {(betAmount == 0 ? "pass" : $"bet {betAmount}")}.");

            // 生成并显示最终的牌
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("");
            Console.WriteLine("The next card is:");

            int thirdCard = GetCard();
            DisplayCard(thirdCard);
            Console.WriteLine("");

            // 第三张牌是否在前两张牌之间？
            # 如果第三张牌大于第一张牌并且小于第二张牌
            if (thirdCard > firstCard && thirdCard < secondCard)
            {
                # 是的！通知玩家并且增加他们的赌注
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("You win!");
                # 如果赌注为0
                if (betAmount == 0)
                {
                    Console.WriteLine("(It's just a shame you chose not to bet!)")
                }
                else:
                    state.Money += betAmount
                    # 如果他们的钱超过了最大金额，也要更新最大金额
                    state.MaxMoney = Math.Max(state.Money, state.MaxMoney)
            }
            else:
                # 哦，玩家输了。让他们知道这个坏消息并且从他们的钱中扣除赌注
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("You lose!");  // 打印“你输了！”
                if (betAmount == 0)  // 如果下注金额为0
                {
                    Console.WriteLine("(It's lucky you chose not to bet!)");  // 打印“(幸运的是你选择不下注！)”
                }
                else
                {
                    state.Money -= betAmount;  // 从玩家的钱中减去下注金额
                }
            }

            Console.ForegroundColor = ConsoleColor.White;  // 设置控制台前景色为白色
            Console.Write("You now have ");  // 打印“你现在有”
            Console.ForegroundColor = ConsoleColor.Yellow;  // 设置控制台前景色为黄色
            Console.Write($"${state.Money}");  // 打印玩家当前的钱数
            Console.ForegroundColor = ConsoleColor.White;  // 设置控制台前景色为白色
            Console.WriteLine(".");  // 打印句号

            // Update the turn count now that another turn has been played
            state.TurnCount += 1;  // 更新游戏回合数，因为又玩了一回合
            // 准备下一轮...
            Console.ForegroundColor = ConsoleColor.DarkGreen; // 设置控制台前景色为深绿色
            Console.WriteLine(""); // 输出空行
            Console.WriteLine("Press any key to continue..."); // 输出提示信息
            Console.ReadKey(true); // 等待用户按下任意键继续
        }

        /// <summary>
        /// 提示用户输入他们的赌注金额并验证他们的输入
        /// </summary>
        /// <param name="currentMoney">玩家当前的资金</param>
        /// <returns>返回玩家选择下注的金额</returns>
        private int PlayTurn_GetBetAmount(int currentMoney)
        {
            int betAmount; // 声明赌注金额变量
            // 循环直到用户输入有效值
            do
            {
                // 将此移到一个单独的函数中...
                // 设置控制台前景色为黄色
                Console.ForegroundColor = ConsoleColor.Yellow;
                // 输出提示符
                Console.Write("> $");
                // 读取用户输入
                string input = Console.ReadLine();

                // 检查用户输入是否为有效的数字
                if (!int.TryParse(input, out betAmount))
                {
                    // 不是有效数字
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Sorry, I didn't understand. Please enter how much you would like to bet.");
                    // 继续循环
                    continue;
                }

                // 检查下注金额是否在0到可用资金之间
                if (betAmount < 0 || betAmount > currentMoney)
                {
                    // 不在范围内
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"Please enter a bet amount between $0 and ${currentMoney}.");
                    // 继续循环
                    continue;
                }

                // 我们有一个有效的赌注，停止循环
                break;
            } while (true);

            // 返回玩家输入的赌注
            return betAmount;
        }

        /// <summary>
        /// 生成一个新的随机卡片。
        /// </summary>
        /// <returns>将返回一个介于2和14之间的值，包括2和14。</returns>
        /// <remarks>值2到10是它们的面值。11代表一个J，12是一个Q，13是一个K，14是一个A。
        /// 尽管这是一个稍微偏移的序列，但它允许我们对卡片值进行简单的大于/小于比较，将A视为高牌。</remarks>
        private int GetCard()
        {
            return Rnd.Next(2, 15);  # 生成一个介于2到14之间的随机数，包括2但不包括15
        }

        /// <summary>
        /// 在屏幕上显示卡片号码，将值从11到14翻译成它们的名称等价物。
        /// </summary>
        /// <param name="card"></param>
        private void DisplayCard(int card)
        {
            string cardText;  # 声明一个字符串变量用于存储卡片的文本值
            switch (card)  # 开始一个switch语句，根据card的值进行不同的处理
            {
                case 11:  # 如果card的值为11
                    cardText = "Jack";  # 将cardText赋值为"Jack"
                    break;  # 结束当前case的处理
                case 12:  # 如果card的值为12
                    cardText = "Queen";  # 将cardText赋值为"Queen"
                    break;  # 结束当前case的处理
                case 13:  # 如果card的值为13
                    cardText = "King";  # 如果卡片的值是13，将cardText设置为"King"
                    break;  # 跳出switch语句
                case 14:  # 如果卡片的值是14
                    cardText = "Ace";  # 将cardText设置为"Ace"
                    break;  # 跳出switch语句
                default:  # 如果卡片的值不是13或14
                    cardText = card.ToString();  # 将cardText设置为卡片的字符串表示
                    break;  # 跳出switch语句
            }

            // Format as black text on a white background  # 设置为白色背景上的黑色文本格式
            Console.Write("   ");  # 输出三个空格
            Console.BackgroundColor = ConsoleColor.White;  # 设置控制台背景颜色为白色
            Console.ForegroundColor = ConsoleColor.Black;  # 设置控制台前景颜色为黑色
            Console.Write($"  {cardText}  ");  # 输出卡片文本
            Console.BackgroundColor = ConsoleColor.Black;  # 设置控制台背景颜色为黑色
            Console.ForegroundColor = ConsoleColor.White;  # 设置控制台前景颜色为白色
            Console.WriteLine("");  # 输出一个空行
        }
        /// <summary>
        /// 显示游戏玩法说明，并等待玩家按键。
        /// </summary>
        private void DisplayIntroText()
        {
            // 设置控制台前景色为黄色
            Console.ForegroundColor = ConsoleColor.Yellow;
            // 打印游戏名称和地点
            Console.WriteLine("Acey Ducey Gard Game.");
            Console.WriteLine("Creating Computing, Morristown, New Jersey.");
            Console.WriteLine("");

            // 设置控制台前景色为深绿色
            Console.ForegroundColor = ConsoleColor.DarkGreen;
            // 打印游戏的出版信息
            Console.WriteLine("Originally published in 1978 in the book 'Basic Computer Games' by David Ahl.");
            Console.WriteLine("Modernised and converted to C# in 2021 by Adam Dawes (@AdamDawes575).");
            Console.WriteLine("");

            // 设置控制台前景色为灰色
            Console.ForegroundColor = ConsoleColor.Gray;
            // 打印游戏玩法说明
            Console.WriteLine("Acey Ducey is played in the following manner:");
            Console.WriteLine("");
            Console.WriteLine("The dealer (computer) deals two cards, face up.");
            Console.WriteLine("");
            // 输出提示信息，告诉玩家可以选择下注或放弃，取决于他们是否认为下一张牌的值会在前两张牌之间
            Console.WriteLine("You have an option to bet or pass, depending on whether or not you feel the next card will have a value between the");
            Console.WriteLine("first two.");
            Console.WriteLine("");
            Console.WriteLine("If the card is between, you will win your stake, otherwise you will lose it. Ace is 'high' (higher than a King).");
            Console.WriteLine("");
            Console.WriteLine("If you want to pass, enter a bet amount of $0.");
            Console.WriteLine("");

            // 设置控制台前景色为黄色
            Console.ForegroundColor = ConsoleColor.Yellow;
            // 输出提示信息，告诉玩家按任意键开始游戏
            Console.WriteLine("Press any key start the game.");
            // 等待玩家按下任意键
            Console.ReadKey(true);

        }

        /// <summary>
        /// 提示玩家再试一次，并等待他们按下 Y 或 N
        /// </summary>
        /// <returns>如果玩家想再试一次，则返回true，如果他们已经玩够了，则返回false。</returns>
        private bool TryAgain()
        {
            // 设置控制台前景色为白色
            Console.ForegroundColor = ConsoleColor.White;
            // 打印消息询问用户是否要再次尝试（按 'Y' 表示是，按 'N' 表示否）
            Console.WriteLine("Would you like to try again? (Press 'Y' for yes or 'N' for no)");

            // 设置控制台前景色为黄色
            Console.ForegroundColor = ConsoleColor.Yellow;
            // 打印提示符
            Console.Write("> ");

            // 定义变量保存用户按下的键
            char pressedKey;
            // 循环直到获取到一个被识别的输入
            do
            {
                // 读取一个键，不在屏幕上显示
                ConsoleKeyInfo key = Console.ReadKey(true);
                // 转换为大写，这样我们就不需要关心大小写
                pressedKey = Char.ToUpper(key.KeyChar);
                // 如果这是一个我们识别的键吗？如果不是，继续循环
            } while (pressedKey != 'Y' && pressedKey != 'N');
            // 在屏幕上显示结果
            Console.WriteLine(pressedKey);

            // 如果玩家按下 'Y'，返回 true，否则返回 false
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 以二进制只读模式打开文件，读取文件内容并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象的文件名列表，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```