# `d:/src/tocomm/basic-computer-games\76_Russian_Roulette\csharp\Program.cs`

```
// 打印游戏标题
void PrintTitle()
{
    Console.WriteLine("Russian Roulette");
}

// 主程序入口
public static void Main(string[] args)
{
    // 打印游戏标题
    PrintTitle();

    // 初始化是否包含左轮手枪的变量
    var includeRevolver = true;
    // 游戏循环
    while (true)
    {
        // 打印游戏说明
        PrintInstructions(includeRevolver);
        // 开始游戏，并根据游戏结果进行处理
        switch (PlayGame())
        {
            // 赢得游戏
            case GameResult.Win:
                includeRevolver = true;
                break;
            // 逃跑
            case GameResult.Chicken:
        private static void PrintTitle()
        {
            // 打印游戏标题
            Console.WriteLine("           Russian Roulette");
            // 打印游戏创意来源地点
            Console.WriteLine("Creative Computing  Morristown, New Jersey");
            // 打印空行
            Console.WriteLine();
            // 打印空行
            Console.WriteLine();
            // 打印空行
            Console.WriteLine();
            // 打印游戏说明
            Console.WriteLine("This is a game of >>>>>>>>>>Russian Roulette.");
        }

        private static void PrintInstructions(bool includeRevolver)
        {
            // 打印游戏说明
            Console.WriteLine();
            if (includeRevolver)  # 如果包含左轮手枪
            {
                Console.WriteLine("Here is a revolver.");  # 输出提示信息：这里有一把左轮手枪
            }
            else  # 否则
            {
                Console.WriteLine();  # 输出空行
                Console.WriteLine();  # 输出空行
                Console.WriteLine("...Next Victim...");  # 输出提示信息：...下一个受害者...
            }
            Console.WriteLine("Type '1' to spin chamber and pull trigger.");  # 输出提示信息：输入'1'来旋转弹舱并扣动扳机
            Console.WriteLine("Type '2' to give up.");  # 输出提示信息：输入'2'来放弃
        }

        private static GameResult PlayGame()  # 定义一个名为PlayGame的静态方法，返回类型为GameResult
        {
            var rnd = new Random();  # 创建一个Random对象rnd
            var round = 0;  # 初始化一个名为round的变量，赋值为0
            while (true)  # 进入无限循环
            {
                round++;  // 增加游戏轮数
                Console.Write("Go: ");  // 在控制台输出提示信息
                var input = Console.ReadKey().KeyChar;  // 从控制台读取用户输入的字符
                Console.WriteLine();  // 在控制台输出空行
                if (input != '2')  // 如果用户输入的字符不是'2'
                {
                    // Random.Next将返回一个大于等于最小值且小于最大值的随机数
                    // 一个左轮手枪有6发子弹
                    if (rnd.Next(1, 7) == 6)  // 如果随机数等于6
                    {
                        Console.WriteLine("     Bang!!!!!   You're dead!");  // 在控制台输出玩家被击中的信息
                        Console.WriteLine("Condolences will be sent to your relatives.");  // 在控制台输出慰问信息
                        return GameResult.Dead;  // 返回游戏结果为死亡
                    }
                    else  // 如果随机数不等于6
                    {
                        if (round > 10)  // 如果游戏轮数大于10
                        {
                            Console.WriteLine("You win!!!!!");  // 在控制台输出玩家获胜的信息
                            Console.WriteLine("Let someone else blow their brains out.");  # 打印消息，表示游戏结果为胜利
                            return GameResult.Win;  # 返回游戏结果为胜利
                        }
                        else
                        {
                            Console.WriteLine("- CLICK -");  # 打印消息，表示扳机被扣动但没有发射子弹
                            Console.WriteLine();  # 打印空行
                        }
                    }
                }
                else
                {
                    Console.WriteLine("     CHICKEN!!!!!");  # 打印消息，表示玩家胆小
                    return GameResult.Chicken;  # 返回游戏结果为胆小
                }
            }
        }

        private enum GameResult  # 定义枚举类型 GameResult
抱歉，你提供的代码片段似乎不完整或与要求不符，我无法为其添加注释。请提供正确的代码片段，以便我能够帮助你添加注释。
```