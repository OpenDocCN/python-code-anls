# `69_Pizza\csharp\PizzaGame.cs`

```
// 命名空间 Pizza
namespace Pizza
{
    // PizzaGame 类
    internal class PizzaGame
    {
        // 常量 CustomerMapSize，值为 4
        private const int CustomerMapSize = 4;
        // 创建 CustomerMap 对象 _customerMap，传入 CustomerMapSize 作为参数
        private readonly CustomerMap _customerMap = new CustomerMap(CustomerMapSize);

        /// <summary>
        /// 开始游戏。Pizza 游戏的主协调器。
        /// 负责显示信息，从用户获取数据，并开始派送披萨。
        /// </summary>
        public void Play()
        {
            // 显示游戏标题
            ShowHeader();

            // 获取玩家姓名
            string playerName = GetPlayerName();

            // 显示游戏介绍，传入玩家姓名作为参数
            ShowIntroduction(playerName);
            // 显示地图
            ShowMap();
            # 如果需要更多的指示
            if (AskForMoreDirections())
            {
                # 显示更多的指示给玩家
                ShowMoreDirections(playerName);

                # 询问玩家是否理解
                var playerUnderstands = AskIfPlayerUnderstand();
                # 如果玩家不理解，则返回
                if (!playerUnderstands)
                {
                    return;
                }
            }

            # 开始配送
            StartDelivery(playerName);
            # 结束配送
            EndDelivery(playerName);
        }

        /// <summary>
        /// 从开始向顾客配送披萨。
        /// 每5次配送询问用户是否想继续配送。
        /// </summary>
        /// <param name="playerName">由用户填写的玩家姓名。</param>
                // 开始配送，初始化已配送比萨数量
                private void StartDelivery(string playerName)
                {
                    var numberOfDeliveredPizzas = 0;
                    // 进入无限循环，不断配送比萨
                    while (true)
                    {
                        numberOfDeliveredPizzas++;
                        // 从顾客列表中随机选择一个顾客
                        string deliverPizzaToCustomer = GetRandomCustomer();

                        // 输出配送信息
                        WriteEmptyLine();
                        Console.WriteLine($"HELLO {playerName}'S PIZZA.  THIS IS {deliverPizzaToCustomer}.");
                        Console.WriteLine("\tPLEASE SEND A PIZZA.");

                        // 玩家配送比萨给顾客
                        DeliverPizzaByPlayer(playerName, deliverPizzaToCustomer);

                        // 每配送5个比萨询问玩家是否继续配送
                        if (numberOfDeliveredPizzas % 5 == 0)
                        {
                            bool playerWantToDeliveryMorePizzas = AskQuestionWithYesNoResponse("DO YOU WANT TO DELIVER MORE PIZZAS?");
                            if (!playerWantToDeliveryMorePizzas)
                            {
                                WriteEmptyLine();
/// <summary>
/// 获取应该送披萨的随机顾客。
/// </summary>
/// <returns>顾客姓名。</returns>
private string GetRandomCustomer()
{
    // 生成随机的 X 坐标位置
    int randomPositionOnX = Random.Shared.Next(0, CustomerMapSize);
    // 生成随机的 Y 坐标位置
    int randomPositionOnY = Random.Shared.Next(0, CustomerMapSize);

    // 根据随机位置获取顾客姓名
    return _customerMap.GetCustomerOnPosition(randomPositionOnX, randomPositionOnY);
}

/// <summary>
/// 由玩家将披萨送到顾客那里。它验证玩家是否将披萨送到了正确的顾客那里。
        /// <summary>
        /// 通过玩家输入的信息将披萨送到顾客处
        /// </summary>
        /// <param name="playerName">玩家填写的玩家名称</param>
        /// <param name="deliverPizzaToCustomer">订购披萨的顾客名称</param>
        private void DeliverPizzaByPlayer(string playerName, string deliverPizzaToCustomer)
        {
            while (true)
            {
                // 获取玩家输入的信息
                string userInput = GetPlayerInput($"\tDRIVER TO {playerName}:  WHERE DOES {deliverPizzaToCustomer} LIVE?");
                // 从玩家输入的信息中获取顾客名称
                var deliveredToCustomer = GetCustomerFromPlayerInput(userInput);
                // 如果顾客名称为空，则设置为"UNKNOWN CUSTOMER"
                if (string.IsNullOrEmpty(deliveredToCustomer))
                {
                    deliveredToCustomer = "UNKNOWN CUSTOMER";
                }

                // 如果送达的顾客名称与订购披萨的顾客名称相同，则打印消息并结束循环
                if (deliveredToCustomer.Equals(deliverPizzaToCustomer))
                {
                    Console.WriteLine($"HELLO {playerName}.  THIS IS {deliverPizzaToCustomer}, THANKS FOR THE PIZZA.");
                    break;
                }
                // 打印出顾客收到的消息，消息中包含顾客的姓名和地址
                Console.WriteLine($"THIS IS {deliveredToCustomer}.  I DID NOT ORDER A PIZZA.");
                // 打印出顾客的地址
                Console.WriteLine($"I LIVE AT {userInput}");
            }
        }

        /// <summary>
        /// 通过用户输入获取顾客的姓名和坐标
        /// </summary>
        /// <param name="userInput">用户输入的内容，应该包含用逗号分隔的顾客坐标</param>
        /// <returns>如果坐标正确且顾客存在，则返回 true，否则返回 false</returns>
        private string GetCustomerFromPlayerInput(string userInput)
        {
            // 将用户输入的内容按逗号分隔，并转换为整数数组
            var pizzaIsDeliveredToPosition = userInput?
                .Split(',')
                .Select(i => int.TryParse(i, out var customerPosition) ? (customerPosition - 1) : -1)
                .Where(i => i != -1)
                .ToArray() ?? Array.Empty<int>();
            // 如果坐标数组的长度不为2，则返回空字符串
            if (pizzaIsDeliveredToPosition.Length != 2)
            {
                return string.Empty;
            }

            return _customerMap.GetCustomerOnPosition(pizzaIsDeliveredToPosition[0], pizzaIsDeliveredToPosition[1]);
        }

        /// <summary>
        /// Shows game header in console.
        /// </summary>
        private void ShowHeader()
        {
            // 在控制台显示游戏标题
            Console.WriteLine("PIZZA".PadLeft(22));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            WriteEmptyLine(3);
            Console.WriteLine("PIZZA DELIVERY GAME");
            WriteEmptyLine();
        }

        /// <summary>
        /// Asks user for name which will be used in game.
        /// </summary>
        /// <returns>Player name.</returns>
        private string GetPlayerName()
        {
            // 返回玩家输入的名字
            return GetPlayerInput("WHAT IS YOUR FIRST NAME:");
        }

        /// <summary>
        /// Shows game introduction in console
        /// </summary>
        /// <param name="playerName">Player name which was filled by user.</param>
        private void ShowIntroduction(string playerName)
        {
            // 在控制台中显示游戏介绍，包括玩家的名字
            Console.WriteLine($"HI, {playerName}.  IN THIS GAME YOU ARE TO TAKE ORDERS");
            Console.WriteLine("FOR PIZZAS.  THEN YOU ARE TO TELL A DELIVERY BOY");
            Console.WriteLine("WHERE TO DELIVER THE ORDERED PIZZAS.");
            WriteEmptyLine(2); // 调用自定义的方法，在控制台中输出两行空行
        }

        /// <summary>
        /// Shows customers map in console. In this method is used overridden method 'ToString' for getting text representation of customers map.
        /// </summary>
        // 显示城市海茨维尔的地图
        private void ShowMap()
        {
            Console.WriteLine("MAP OF THE CITY OF HYATTSVILLE"); // 打印城市海茨维尔的地图
            WriteEmptyLine(); // 调用自定义函数，在控制台输出空行

            Console.WriteLine(_customerMap.ToString()); // 打印客户地图信息

            Console.WriteLine("THE OUTPUT IS A MAP OF THE HOMES WHERE"); // 打印输出的是客户家的地图
            Console.WriteLine("YOU ARE TO SEND PIZZAS."); // 打印你需要送披萨的地方
            WriteEmptyLine(); // 调用自定义函数，在控制台输出空行
            Console.WriteLine("YOUR JOB IS TO GIVE A TRUCK DRIVER"); // 打印你的工作是给卡车司机
            Console.WriteLine("THE LOCATION OR COORDINATES OF THE"); // 打印客户家的位置或坐标
            Console.WriteLine("HOME ORDERING THE PIZZA."); // 打印订购披萨的家庭
            WriteEmptyLine(); // 调用自定义函数，在控制台输出空行
        }

        /// <summary>
        /// Asks user if needs more directions.
        /// </summary>
        /// <returns>True if user need more directions otherwise false.</returns>
        private bool AskForMoreDirections()
        {
            // 调用AskQuestionWithYesNoResponse方法询问用户是否需要更多指引
            var playerNeedsMoreDirections = AskQuestionWithYesNoResponse("DO YOU NEED MORE DIRECTIONS?");
            // 调用WriteEmptyLine方法输出空行
            WriteEmptyLine();

            // 返回用户是否需要更多指引的结果
            return playerNeedsMoreDirections;
        }

        /// <summary>
        /// Shows more directions.
        /// </summary>
        /// <param name="playerName">Player name which was filled by user.</param>
        private void ShowMoreDirections(string playerName)
        {
            // 输出更多指引的内容
            Console.WriteLine("SOMEBODY WILL ASK FOR A PIZZA TO BE");
            Console.WriteLine("DELIVERED.  THEN A DELIVERY BOY WILL");
            Console.WriteLine("ASK YOU FOR THE LOCATION.");
            Console.WriteLine("\tEXAMPLE:");
            Console.WriteLine("THIS IS J.  PLEASE SEND A PIZZA.");
        }
            Console.WriteLine($"DRIVER TO {playerName}.  WHERE DOES J LIVE?");  // 打印出司机向玩家询问J的住址
            Console.WriteLine("YOUR ANSWER WOULD BE 2,3");  // 打印出提示玩家应该回答的内容为2,3
        }

        /// <summary>
        /// Asks user if understands to instructions.
        /// </summary>
        /// <returns>True if user understand otherwise false.</returns>
        private bool AskIfPlayerUnderstand()
        {
            var playerUnderstands = AskQuestionWithYesNoResponse("UNDERSTAND?");  // 调用AskQuestionWithYesNoResponse方法询问玩家是否理解了指示
            if (!playerUnderstands)  // 如果玩家不理解
            {
                Console.WriteLine("THIS JOB IS DEFINITELY TOO DIFFICULT FOR YOU. THANKS ANYWAY");  // 打印出提示玩家这项工作对他来说太难了
                return false;  // 返回false
            }

            WriteEmptyLine();  // 调用WriteEmptyLine方法打印空行
            Console.WriteLine("GOOD.  YOU ARE NOW READY TO START TAKING ORDERS.");  // 打印出提示玩家现在已经准备好开始接受订单了
            WriteEmptyLine();  // 调用WriteEmptyLine方法打印空行
            Console.WriteLine("GOOD LUCK!!");
            WriteEmptyLine();  // 调用自定义函数，向控制台输出空行

            return true;  // 返回 true，表示成功执行该函数
        }

        /// <summary>
        /// Shows message about ending delivery in console.
        /// </summary>
        /// <param name="playerName">Player name which was filled by user.</param>
        private void EndDelivery(string playerName)
        {
            Console.WriteLine($"O.K. {playerName}, SEE YOU LATER!");  // 在控制台显示结束交付的消息，包括玩家的名字
            WriteEmptyLine();  // 调用自定义函数，向控制台输出空行
        }

        /// <summary>
        /// Gets input from user.
        /// </summary>
        /// <param name="question">Question which is displayed in console.</param>
        /// <returns>User input.</returns>
        private string GetPlayerInput(string question)
        {
            // 输出问题到控制台
            Console.Write($"{question} ");

            // 循环等待用户输入
            while (true)
            {
                // 读取用户输入
                var userInput = Console.ReadLine();
                // 如果用户输入不为空或空格
                if (!string.IsNullOrWhiteSpace(userInput))
                {
                    // 返回用户输入
                    return userInput;
                }
            }
        }

        /// <summary>
        /// Asks user with required resposne 'YES', 'Y, 'NO', 'N'.
        /// </summary>
        /// <param name="question">Question which is displayed in console.</param>
        /// <returns>True if user write 'YES', 'Y'. False if user write 'NO', 'N'.</returns>
        # 定义一个函数，用于向用户提出问题并接收 yes 或 no 的回答
        def AskQuestionWithYesNoResponse(question):
            # 定义用户可能的肯定回答
            possitiveResponse = ["Y", "YES"]
            # 定义用户可能的否定回答
            negativeResponse = ["N", "NO"]
            # 将肯定回答和否定回答合并成一个有效的用户输入列表
            validUserInputs = possitiveResponse + negativeResponse

            # 在控制台上打印问题
            print(f"{question} ", end="")

            # 循环等待用户输入
            while True:
                userInput = input()
                # 如果用户输入不为空且在有效的用户输入列表中
                if userInput.strip().upper() in validUserInputs:
                    # 退出循环
                    break

                # 如果用户输入不合法，提示用户重新输入
                print(f"'YES' OR 'NO' PLEASE, NOW THEN, {question} ", end="")
        return possitiveResponse.Contains(userInput.ToUpper());
        // 检查 positiveResponse 是否包含 userInput 转换为大写后的内容，返回布尔值

        /// <summary>
        /// Writes empty line in console.
        /// </summary>
        /// <param name="numberOfEmptyLines">Number of empty lines which will be written in console. Parameter is optional and default value is 1.</param>
        private void WriteEmptyLine(int numberOfEmptyLines = 1)
        {
            for (int i = 0; i < numberOfEmptyLines; i++)
            {
                Console.WriteLine();
            }
        }
        // 在控制台中写入空行，可以指定写入的空行数量，默认为1行
```