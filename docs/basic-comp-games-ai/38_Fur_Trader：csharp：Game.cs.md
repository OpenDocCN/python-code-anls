# `38_Fur_Trader\csharp\Game.cs`

```
        /// <summary>
        /// random number generator; no seed to be faithful to original implementation
        /// </summary>
        private Random Rnd { get; } = new Random();  // 创建一个随机数生成器对象，没有种子以保持原始实现的一致性

        /// <summary>
        /// Generate a price for pelts based off a factor and baseline value
        /// </summary>
        /// <param name="factor">Multiplier for the price</param>
        /// <param name="baseline">The baseline price</param>
        /// <returns>A randomised price for pelts</returns>
        internal double RandomPriceGenerator(double factor, double baseline)
        {
            var price = (Convert.ToInt32((factor * Rnd.NextDouble() + baseline) * 100d) + 5) / 100d;  // 根据给定的因子和基准值生成皮毛的价格
            return price;
        }
        # 返回价格

        /// <summary>
        /// Main game loop function. This will play the game endlessly until the player chooses to quit or a GameOver event occurs
        /// </summary>
        /// <remarks>
        /// General structure followed from Adam Dawes (@AdamDawes575) implementation of Acey Ducey.");
        /// </remarks>
        # 主游戏循环函数。这将无休止地玩游戏，直到玩家选择退出或发生游戏结束事件
        # 一般结构遵循了Adam Dawes (@AdamDawes575)对Acey Ducey的实现。

        internal void GameLoop()
        {
            // display instructions to the player
            # 向玩家显示说明文本
            DisplayIntroText();

            var state = new GameState();

            // loop for each turn until the player decides not to continue (or has a Game Over event)
            # 循环每一轮，直到玩家决定不继续（或发生游戏结束事件）
            while ((!state.GameOver) && ContinueGame())
            {
                // clear display at start of each turn
                # 每一轮开始时清除显示
                Console.Clear();  // 清空控制台屏幕

                // play the next turn; pass game state for details and updates from the turn
                PlayTurn(state);  // 执行下一个回合，传递游戏状态以获取回合的详细信息和更新

            }

            // end screen; show some statistics to the player
            Console.Clear();  // 清空控制台屏幕
            Console.WriteLine("Thanks for playing!");  // 显示感谢玩家的消息
            Console.WriteLine("");  // 输出空行
            Console.WriteLine($"Total Expeditions: {state.ExpeditionCount}");  // 显示总远征次数
            Console.WriteLine($"Final Amount:      {state.Savings.ToString("c")}");  // 显示最终金额

        }

        /// <summary>
        /// Display instructions on how to play the game and wait for the player to press a key.
        /// </summary>
        private void DisplayIntroText()
        {
            Console.ForegroundColor = ConsoleColor.Yellow;  // 设置控制台前景色为黄色
            # 输出游戏的标题和作者信息
            Console.WriteLine("Fur Trader.")
            Console.WriteLine("Creating Computing, Morristown, New Jersey.")
            Console.WriteLine("")

            # 设置控制台文本颜色为深绿色，输出游戏的原始出版信息
            Console.ForegroundColor = ConsoleColor.DarkGreen
            Console.WriteLine("Originally published in 1978 in the book 'Basic Computer Games' by David Ahl.")
            Console.WriteLine("")

            # 设置控制台文本颜色为灰色，输出游戏背景信息
            Console.ForegroundColor = ConsoleColor.Gray
            Console.WriteLine("You are the leader of a French fur trading expedition in 1776 leaving the Lake Ontario area to sell furs and get supplies for the next year.")
            Console.WriteLine("")
            Console.WriteLine("You have a choice of three forts at which you may trade. The cost of supplies and the amount you receive for your furs will depend on the fort that you choose.")
            Console.WriteLine("")

            # 设置控制台文本颜色为黄色，提示玩家按任意键开始游戏
            Console.ForegroundColor = ConsoleColor.Yellow
            Console.WriteLine("Press any key start the game.")
            Console.ReadKey(true)
        /// <summary>
        /// Prompt the player to try again, and wait for them to press Y or N.
        /// </summary>
        /// <returns>Returns true if the player wants to try again, false if they have finished playing.</returns>
        private bool ContinueGame()
        {
            // 输出空行
            Console.WriteLine("");
            // 设置控制台前景色为白色
            Console.ForegroundColor = ConsoleColor.White;
            // 输出提示信息
            Console.WriteLine("Do you wish to trade furs? ");
            // 输出提示信息
            Console.Write("Answer (Y)es or (N)o ");
            // 设置控制台前景色为黄色
            Console.ForegroundColor = ConsoleColor.Yellow;
            // 输出提示信息
            Console.Write("> ");

            char pressedKey;
            // 保持循环直到获得一个被识别的输入
            do
            {
                // 读取一个键，不在屏幕上显示
                ConsoleKeyInfo key = Console.ReadKey(true);
                // 转换为大写，这样我们就不需要关心大小写问题
                pressedKey = Char.ToUpper(key.KeyChar); // 将输入的字符转换为大写

                // 如果输入的字符不是 'Y' 或 'N'，则继续循环
            } while (pressedKey != 'Y' && pressedKey != 'N');

            // 在屏幕上显示结果
            Console.WriteLine(pressedKey);

            // 如果玩家按下 'Y'，则返回 true，否则返回 false
            return (pressedKey == 'Y');
        }

        /// <summary>
        /// Play a turn
        /// </summary>
        /// <param name="state">The current game state</param>
        private void PlayTurn(GameState state)
        {
            state.UnasignedFurCount = 190;      /// 每轮开始时有 190 个毛皮

            // 向用户提供当前状态
// 打印一条由 70 个下划线组成的分隔线
Console.WriteLine(new string('_', 70));
// 打印空行
Console.WriteLine("");
// 设置控制台前景色为白色
Console.ForegroundColor = ConsoleColor.White;
// 打印账户储蓄金额和未分配的毛皮数量
Console.WriteLine($"You have {state.Savings.ToString("c")} savings and {state.UnasignedFurCount} furs to begin the expedition.");
// 打印空行
Console.WriteLine("");
// 打印未分配的毛皮数量以及各种皮毛的种类
Console.WriteLine($"Your {state.UnasignedFurCount} furs are distributed among the following kinds of pelts: Mink, Beaver, Ermine, and Fox");
// 打印空行
Console.WriteLine("");

// 获取用户输入的水貂皮数量
Console.ForegroundColor = ConsoleColor.White;
Console.Write("How many Mink pelts do you have? ");
state.MinkPelts = GetPelts(state.UnasignedFurCount);
// 减去已分配的水貂皮数量
state.UnasignedFurCount -= state.MinkPelts;
// 打印空行
Console.WriteLine("");
// 设置控制台前景色为白色
Console.ForegroundColor = ConsoleColor.White;
// 打印剩余未分配的毛皮数量
Console.WriteLine($"You have {state.UnasignedFurCount} furs remaining for distribution");
// 获取用户输入的海狸皮数量
Console.Write("How many Beaver pelts do you have? ");
state.BeaverPelts = GetPelts(state.UnasignedFurCount);
// 减去已分配的海狸皮数量
state.UnasignedFurCount -= state.BeaverPelts;
            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.White;  // 设置控制台输出的前景色为白色
            Console.WriteLine($"You have {state.UnasignedFurCount} furs remaining for distribution");  // 输出剩余未分配的毛皮数量
            Console.Write("How many Ermine pelts do you have? ");  // 提示用户输入鼬皮的数量
            state.ErminePelts = GetPelts(state.UnasignedFurCount);  // 获取用户输入的鼬皮数量
            state.UnasignedFurCount -= state.ErminePelts;  // 更新剩余未分配的毛皮数量
            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.White;  // 设置控制台输出的前景色为白色
            Console.WriteLine($"You have {state.UnasignedFurCount} furs remaining for distribution");  // 输出剩余未分配的毛皮数量
            Console.Write("How many Fox pelts do you have? ");  // 提示用户输入狐狸皮的数量
            state.FoxPelts = GetPelts(state.UnasignedFurCount);  // 获取用户输入的狐狸皮数量
            state.UnasignedFurCount -= state.FoxPelts;  // 更新剩余未分配的毛皮数量

            // 获取用户选择要交易的堡垒；用户在选择后有机会评估并重新选择堡垒，直到用户确认选择
            do
            {
                Console.ForegroundColor = ConsoleColor.White;  // 设置控制台输出的前景色为白色
                Console.WriteLine("");
                Console.WriteLine("Do you want to trade your furs at Fort 1, Fort 2, or Fort 3");  // 提示用户选择要在哪个堡垒交易毛皮
                Console.WriteLine("Fort 1 is Fort Hochelaga (Montreal) and is under the protection of the French army.");  // 输出堡垒1的信息
                // 输出关于Fort 2的信息，包括位置和所需行动
                Console.WriteLine("Fort 2 is Fort Stadacona (Quebec) and is under the protection of the French army. However, you must make a portage and cross the Lachine rapids.");
                // 输出关于Fort 3的信息，包括位置和所需行动
                Console.WriteLine("Fort 3 is Fort New York and is under Dutch control. You must cross through Iroquois land.");
                // 输出空行
                Console.WriteLine("");
                // 获取用户选择的要前往的堡垒
                state.SelectedFort = GetSelectedFort();

                // 显示用户选择的堡垒的信息
                DisplaySelectedFortInformation(state.SelectedFort);

            } while (TradeAtAnotherFort());

            // 处理前往堡垒的旅行
            ProcessExpeditionOutcome(state);

            // 显示远征结果（储蓄变化）给用户
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("You now have ");
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write($"{state.Savings.ToString("c")}");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine(" including your previous savings.");
            // 更新回合计数，因为又进行了一次回合
            state.ExpeditionCount += 1;
        }

        /// <summary>
        /// 以一些标准格式显示远征成本的方法
        /// </summary>
        /// <param name="fortname">与之交易的堡垒的名称</param>
        /// <param name="supplies">堡垒的供应品成本</param>
        /// <param name="expenses">远征的旅行费用</param>
        internal void DisplayCosts(string fortname, double supplies, double expenses)
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Supplies at {fortname} cost".PadLeft(55));  // 在控制台上以特定格式显示堡垒供应品的成本
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"{supplies.ToString("c").PadLeft(10)}");  // 以货币格式显示供应品成本
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Your travel expenses to {fortname} were".PadLeft(55));  // 在控制台上以特定格式显示到堡垒的旅行费用
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"{expenses.ToString("c").PadLeft(10)}");  // 以货币格式显示旅行费用
            Console.ForegroundColor = ConsoleColor.White; // 设置控制台前景色为白色

        }

        /// <summary>
        /// Process the results of the expedition
        /// </summary>
        /// <param name="state">the game state</param>
        private void ProcessExpeditionOutcome(GameState state)
        {
            var beaverPrice = RandomPriceGenerator(0.25d, 1.00d); // 生成一个介于0.25和1.00之间的随机价格作为海狸的价格
            var foxPrice =    RandomPriceGenerator(0.2d , 0.80d);  // 生成一个介于0.2和0.80之间的随机价格作为狐狸的价格
            var erminePrice = RandomPriceGenerator(0.15d, 0.95d); // 生成一个介于0.15和0.95之间的随机价格作为貂的价格
            var minkPrice =   RandomPriceGenerator(0.2d , 0.70d);  // 生成一个介于0.2和0.70之间的随机价格作为水貂的价格

            var fortname = String.Empty; // 初始化一个空字符串作为要访问的堡垒的名称
            var suppliesCost = 0.0d;     // 初始化供应品成本为0.0
            var travelExpenses = 0.0d;   // 初始化旅行费用为0.0

            // create a random value 1 to 10 for the different outcomes at each fort
            var p = ((int)(10 * Rnd.NextDouble())) + 1; // 生成一个介于1和10之间的随机整数，用于表示每个堡垒的不同结果
            Console.WriteLine("");  // 打印空行

            switch (state.SelectedFort) {  // 根据 state.SelectedFort 的值进行不同的操作
                case 1:  // 如果 state.SelectedFort 的值为 1
                    beaverPrice = RandomPriceGenerator(0.2d, 0.75d);  // 生成一个介于 0.2 和 0.75 之间的随机价格，并赋值给 beaverPrice
                    foxPrice =    RandomPriceGenerator(0.2d, 0.80d);  // 生成一个介于 0.2 和 0.80 之间的随机价格，并赋值给 foxPrice
                    erminePrice = RandomPriceGenerator(0.2d, 0.65d);  // 生成一个介于 0.2 和 0.65 之间的随机价格，并赋值给 erminePrice
                    minkPrice =   RandomPriceGenerator(0.2d, 0.70d);  // 生成一个介于 0.2 和 0.70 之间的随机价格，并赋值给 minkPrice
                    fortname = "Fort Hochelaga";  // 将 fortname 设置为 "Fort Hochelaga"
                    suppliesCost = 150.0d;  // 将 suppliesCost 设置为 150.0
                    travelExpenses = 10.0d;  // 将 travelExpenses 设置为 10.0
                    break;  // 结束当前 case
                case 2:  // 如果 state.SelectedFort 的值为 2
                    beaverPrice = RandomPriceGenerator(0.2d , 0.90d);  // 生成一个介于 0.2 和 0.90 之间的随机价格，并赋值给 beaverPrice
                    foxPrice =    RandomPriceGenerator(0.2d , 0.80d);  // 生成一个介于 0.2 和 0.80 之间的随机价格，并赋值给 foxPrice
                    erminePrice = RandomPriceGenerator(0.15d, 0.80d);  // 生成一个介于 0.15 和 0.80 之间的随机价格，并赋值给 erminePrice
                    minkPrice =   RandomPriceGenerator(0.3d , 0.85d);  // 生成一个介于 0.3 和 0.85 之间的随机价格，并赋值给 minkPrice
                    fortname = "Fort Stadacona";  // 将 fortname 设置为 "Fort Stadacona"
                    suppliesCost = 125.0d;  // 将 suppliesCost 设置为 125.0
                    # 设置旅行费用为15.0
                    travelExpenses = 15.0d;
                    # 如果旅行天数小于等于2
                    if (p <= 2)
                    {
                        # 将state.BeaverPelts设为0
                        state.BeaverPelts = 0;
                        # 输出信息：你的海狸皮太重了，无法携带过渡地。
                        Console.WriteLine("Your beaver were to heavy to carry across the portage.");
                        # 输出信息：你不得不留下皮毛，但当你回来时发现它们被偷了。
                        Console.WriteLine("You had to leave the pelts but found them stolen when you returned");
                    }
                    # 如果旅行天数小于等于6
                    else if (p <= 6)
                    {
                        # 输出信息：你安全抵达Stadacona砦。
                        Console.WriteLine("You arrived safely at Fort Stadacona");
                    }
                    # 如果旅行天数小于等于8
                    else if (p <= 8)
                    {
                        # 将state.BeaverPelts、state.FoxPelts、state.ErminePelts、state.MinkPelts都设为0
                        state.BeaverPelts = 0;
                        state.FoxPelts = 0;
                        state.ErminePelts = 0;
                        state.MinkPelts = 0;
                        # 输出信息：你的独木舟在拉钦急流中翻了。
                        Console.WriteLine("Your canoe upset in the Lachine Rapids.");
                        # 输出信息：你失去了所有的毛皮。
                        Console.WriteLine("Your lost all your furs");
                    }
                    else if (p <= 10)  # 如果 p 小于等于 10
                    {
                        state.FoxPelts = 0;  # 将 state 对象中的 FoxPelts 属性设置为 0
                        Console.WriteLine("Your fox pelts were not cured properly.");  # 在控制台打印消息，指出狐狸皮未被正确处理
                        Console.WriteLine("No one will buy them.");  # 在控制台打印消息，指出没有人会购买它们
                    }
                    else  # 否则
                    {
                        throw new Exception($"Unexpected Outcome p = {p}");  # 抛出异常，指出意外的结果，包括 p 的值
                    }
                    break;  # 跳出 switch 语句
                case 3:     // outcome for expedition to Fort New York  # 对前往纽约新堡的远征结果
                    beaverPrice = RandomPriceGenerator(0.2d , 1.00d);  # 使用 RandomPriceGenerator 函数生成海狸皮价格
                    foxPrice =    RandomPriceGenerator(0.25d, 1.10d);  # 使用 RandomPriceGenerator 函数生成狐狸皮价格
                    erminePrice = RandomPriceGenerator(0.2d , 0.95d);  # 使用 RandomPriceGenerator 函数生成貂皮价格
                    minkPrice =   RandomPriceGenerator(0.15d, 1.05d);  # 使用 RandomPriceGenerator 函数生成水貂皮价格
                    fortname = "Fort New York";  # 将 fortname 变量设置为 "Fort New York"
                    suppliesCost = 80.0d;  # 将 suppliesCost 变量设置为 80.0
                    travelExpenses = 25.0d;  # 将 travelExpenses 变量设置为 25.0
                    if (p <= 2)  # 如果 p 小于等于 2
                    {
                        // 初始化状态变量
                        state.BeaverPelts = 0;
                        state.FoxPelts = 0;
                        state.ErminePelts = 0;
                        state.MinkPelts = 0;
                        suppliesCost = 0.0d;
                        travelExpenses = 0.0d;
                        // 输出信息：被易洛魁人袭击
                        Console.WriteLine("You were attacked by a party of Iroquois.");
                        Console.WriteLine("All people in your trading group were killed.");
                        Console.WriteLine("This ends the game (press any key).");
                        // 等待用户按下任意键
                        Console.ReadKey(true);
                        // 设置游戏结束状态为真
                        state.GameOver = true;
                    }
                    // 如果概率小于等于6
                    else if (p <= 6)
                    {
                        // 输出信息：安全到达纽约堡
                        Console.WriteLine("You were lucky. You arrived safely at Fort New York.");
                    }
                    // 如果概率小于等于8
                    else if (p <= 8)
                    {
                        // 重置状态变量
                        state.BeaverPelts = 0;
                        state.FoxPelts = 0;  // 将 state 对象中的 FoxPelts 属性设置为 0
                        state.ErminePelts = 0;  // 将 state 对象中的 ErminePelts 属性设置为 0
                        state.MinkPelts = 0;  // 将 state 对象中的 MinkPelts 属性设置为 0
                        Console.WriteLine("You narrowly escaped an Iroquois raiding party.");  // 在控制台打印消息，表示你勉强逃脱了伊罗quois的袭击
                        Console.WriteLine("However, you had to leave all your furs behind.");  // 在控制台打印消息，表示你不得不把所有的毛皮留下
                    }
                    else if (p <= 10)  // 如果 p 小于等于 10
                    {
                        beaverPrice = beaverPrice / 2;  // 将 beaverPrice 除以 2
                        minkPrice = minkPrice / 2;  // 将 minkPrice 除以 2
                        Console.WriteLine("Your mink and beaver were damaged on your trip.");  // 在控制台打印消息，表示你的水貂和海狸在旅途中受损
                        Console.WriteLine("You receive only half the current price for these furs.");  // 在控制台打印消息，表示你只能得到这些毛皮当前价格的一半
                    }
                    else
                    {
                        throw new Exception($"Unexpected Outcome p = {p}");  // 抛出异常，表示意外的结果，p 的值为 {p}
                    }
                    break;  // 结束 switch 语句
                default:  // 默认情况
                    break;  // 结束 switch 语句
            }

            # 计算河狸皮、狐狸皮、貂皮和水貂皮的销售额
            var beaverSale = state.BeaverPelts * beaverPrice;
            var foxSale = state.FoxPelts * foxPrice;
            var ermineSale = state.ErminePelts * erminePrice;
            var minkSale = state.MinkPelts * minkPrice;
            # 计算总利润
            var profit = beaverSale + foxSale + ermineSale + minkSale - suppliesCost - travelExpenses;
            # 将利润加到储蓄中
            state.Savings += profit;

            # 打印销售信息
            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Your {state.BeaverPelts.ToString().PadLeft(3, ' ')} beaver pelts sold @ {beaverPrice.ToString("c")} per pelt for a total");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"{beaverSale.ToString("c").PadLeft(10)}");
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Your {state.FoxPelts.ToString().PadLeft(3, ' ')} fox    pelts sold @ {foxPrice.ToString("c")} per pelt for a total");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"{foxSale.ToString("c").PadLeft(10)}");
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Your {state.ErminePelts.ToString().PadLeft(3, ' ')} ermine pelts sold @ {erminePrice.ToString("c")} per pelt for a total");
```
这段代码是用来计算不同类型皮毛的销售额和总利润，并将利润加到储蓄中。然后通过控制台打印出每种皮毛的销售信息。
            // 设置控制台前景色为绿色
            Console.ForegroundColor = ConsoleColor.Green;
            // 在控制台中打印销售额，并使用货币格式进行左对齐填充
            Console.WriteLine($"{ermineSale.ToString("c").PadLeft(10)}");
            // 设置控制台前景色为白色
            Console.ForegroundColor = ConsoleColor.White;
            // 在控制台中打印销售的水貂皮数量和价格信息
            Console.Write($"Your {state.MinkPelts.ToString().PadLeft(3, ' ')} mink   pelts sold @ {minkPrice.ToString("c")} per pelt for a total");
            // 设置控制台前景色为绿色
            Console.ForegroundColor = ConsoleColor.Green;
            // 在控制台中打印水貂皮销售额，并使用货币格式进行左对齐填充
            Console.WriteLine($"{minkSale.ToString("c").PadLeft(10)}");
            // 在控制台中打印空行
            Console.WriteLine("");
            // 调用DisplayCosts方法，显示要显示的成本信息
            DisplayCosts(fortname, suppliesCost, travelExpenses);
            // 在控制台中打印空行
            Console.WriteLine("");
            // 在控制台中打印利润/损失信息，并根据利润的正负设置控制台前景色为绿色或红色
            Console.Write($"Profit / Loss".PadLeft(55));
            Console.ForegroundColor = profit >= 0.0d ? ConsoleColor.Green : ConsoleColor.Red;
            // 在控制台中打印利润，并使用货币格式进行左对齐填充
            Console.WriteLine($"{profit.ToString("c").PadLeft(10)}");
            // 设置控制台前景色为白色
            Console.ForegroundColor = ConsoleColor.White;
            // 在控制台中打印空行
            Console.WriteLine("");
        }

        // 显示所选要塞的信息
        private void DisplaySelectedFortInformation(int selectedFort)
        {
            // 在控制台中打印空行
            Console.WriteLine("");
            // 设置控制台前景色为白色
            Console.ForegroundColor = ConsoleColor.White;
            switch (selectedFort)
            {
                case 1:    // 选择了Hochelaga要塞的详细信息
                    Console.WriteLine("You have chosen the easiest route.");  // 输出选择了最容易的路线
                    Console.WriteLine("However, the fort is far from any seaport.");  // 输出然而，这个要塞离海港很远
                    Console.WriteLine("The value you receive for your furs will be low.");  // 输出你的毛皮的价值会很低
                    Console.WriteLine("The cost of supplies will be higher than at Forts Stadacona or New York");  // 输出供应品的成本会比Stadacona或New York要塞高
                    break;
                case 2:    // 选择了Stadacona要塞的详细信息
                    Console.WriteLine("You have chosen a hard route.");  // 输出你选择了一条困难的路线
                    Console.WriteLine("It is, in comparsion, harder than the route to Hochelaga but easier than the route to New York.");  // 输出相比之下，它比去Hochelaga要塞的路线更困难，但比去New York要塞的路线更容易
                    Console.WriteLine("You will receive an average value for your furs.");  // 输出你的毛皮会得到平均价值
                    Console.WriteLine("The cost of your supplies will be average.");  // 输出你的供应品成本会是平均水平
                    break;
                case 3:    // 选择了New York要塞的详细信息
                    Console.WriteLine("You have chosen the most difficult route.");  // 输出你选择了最困难的路线
                    Console.WriteLine("At Fort New York you will receive the higest value for your furs.");  // 输出在New York要塞你的毛皮会得到最高价值
                    Console.WriteLine("The cost of your supplies will be lower than at all the other forts.");  // 输出你的供应品成本会比其他所有要塞都要低
                    break;
                default:
                    break;
            }
        }
        // 定义一个名为TradeAtAnotherFort的私有方法，返回布尔值
        private bool TradeAtAnotherFort()
        {
            // 设置控制台前景色为白色，打印空行和提示信息
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("");
            Console.WriteLine("Do you want to trade at another fort?");
            Console.Write("Answer (Y)es or (N)o ");
            // 设置控制台前景色为黄色，打印提示符
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("> ");

            char pressedKey;
            // 循环直到获取到一个被识别的输入
            do
            {
                // 读取一个按键，不在屏幕上显示
                ConsoleKeyInfo key = Console.ReadKey(true);
                // 将按键转换为大写，这样我们就不需要关心大小写
                pressedKey = Char.ToUpper(key.KeyChar);
                // 这是我们认识的按键吗？如果不是，就继续循环
            } while (pressedKey != 'Y' && pressedKey != 'N');

            // 在屏幕上显示结果
            Console.WriteLine(pressedKey);

            // 如果玩家按下'Y'，则返回true，否则返回false
            return (pressedKey == 'Y');
        }

        /// <summary>
        /// 从用户获取一定数量的皮毛
        /// </summary>
        /// <param name="currentMoney">当前可用的皮毛总数</param>
        /// <returns>返回玩家选择的数量</returns>
        private int GetPelts(int furCount)
        {
            int peltCount;
// 循环直到用户输入有效值
do
{
    // 设置控制台前景色为黄色，并提示用户输入
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.Write("> ");
    string input = Console.ReadLine();

    // 解析用户输入，检查是否为有效的数字
    if (!int.TryParse(input, out peltCount))
    {
        // 无效的输入；向用户发送消息
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine("Sorry, I didn't understand. Please enter the number of pelts.");

        // 继续循环
        continue;
    }

    // 检查皮毛数量是否超过可用的皮毛数量
                if (peltCount > furCount)
                {
                    // 如果选择的皮毛数量超过了毛皮数量
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"You may not have that many furs. Do not try to cheat. I can add.");

                    // 继续循环
                    continue;
                }

                // 输入了有效的皮毛数量
                break;
            } while (true);

            // 将皮毛数量返回给用户
            return peltCount;
        }

        /// <summary>
        /// 提示用户选择他们的要建造的堡垒
        /// </summary>
        /// <returns>returns the fort the user has selected</returns>
        private int GetSelectedFort()
        {
            int selectedFort;  // 声明一个整型变量 selectedFort

            // loop until the user enters a valid value
            do
            {
                Console.ForegroundColor = ConsoleColor.White;  // 设置控制台前景色为白色
                Console.Write("Answer 1, 2, or 3 ");  // 在控制台上输出提示信息
                Console.ForegroundColor = ConsoleColor.Yellow;  // 设置控制台前景色为黄色
                Console.Write("> ");  // 在控制台上输出提示信息
                string input = Console.ReadLine();  // 从控制台读取用户输入并存储在 input 变量中

                // is the user entry valid
                if (!int.TryParse(input, out selectedFort))  // 判断用户输入是否为整数，如果是则将其存储在 selectedFort 变量中
                {
                    // no, invalid data
                    Console.ForegroundColor = ConsoleColor.Red;  // 设置控制台前景色为红色
// 输出提示信息
Console.WriteLine("Sorry, I didn't understand. Please answer 1, 2, or 3.");

// 继续循环
continue;
```

```
// 检查选择的要塞是否是选项（只能是1、2或3）
if (selectedFort != 1 && selectedFort != 2 && selectedFort != 3)
{
    // 无效的选择
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"Please answer 1, 2, or 3.");

    // 继续循环
    continue;
}

// 选择了有效的要塞，停止循环
break;
// 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流对象
    使用字节流里面内容创建 ZIP 对象  # 使用字节流内容创建一个 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流创建一个 ZIP 对象，以只读模式打开
    遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典  # 遍历 ZIP 对象中的文件名，读取文件数据，将文件名和数据组成字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 使用字典推导式将文件名和数据组成字典
    # 关闭 ZIP 对象  # 关闭 ZIP 对象，释放资源
    zip.close()
    # 返回结果字典  # 返回文件名到数据的字典
    return fdict
```