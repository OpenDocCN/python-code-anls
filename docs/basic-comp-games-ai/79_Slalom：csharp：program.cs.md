# `79_Slalom\csharp\program.cs`

```
using System.Text;  // 导入 System.Text 命名空间，以便在代码中使用其中的类和方法

namespace Slalom  // 定义名为 Slalom 的命名空间
{
    class Slalom  // 定义名为 Slalom 的类
    {
        private int[] GateMaxSpeed = { 14,18,26,29,18,25,28,32,29,20,29,29,25,21,26,29,20,21,20,  // 声明并初始化名为 GateMaxSpeed 的整型数组
                                       18,26,25,33,31,22 };

        private int GoldMedals = 0;  // 声明并初始化名为 GoldMedals 的整型变量
        private int SilverMedals = 0;  // 声明并初始化名为 SilverMedals 的整型变量
        private int BronzeMedals = 0;  // 声明并初始化名为 BronzeMedals 的整型变量
        private void DisplayIntro()  // 定义名为 DisplayIntro 的私有方法
        {
            Console.WriteLine("");  // 在控制台输出空行
            Console.WriteLine("SLALOM".PadLeft(23));  // 在控制台输出经过左对齐后的字符串 "SLALOM"
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 在控制台输出字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
            Console.WriteLine("");  // 在控制台输出空行
        }
# 显示游戏指令
private void DisplayInstructions()
{
    # 打印游戏介绍
    Console.WriteLine();
    Console.WriteLine("*** Slalom: This is the 1976 Winter Olympic Giant Slalom.  You are");
    Console.WriteLine("            the American team's only hope of a gold medal.");
    Console.WriteLine();
    # 打印游戏操作选项
    Console.WriteLine("     0 -- Type this if you want to see how long you've taken.");
    Console.WriteLine("     1 -- Type this if you want to speed up a lot.");
    Console.WriteLine("     2 -- Type this if you want to speed up a little.");
    Console.WriteLine("     3 -- Type this if you want to speed up a teensy.");
    Console.WriteLine("     4 -- Type this if you want to keep going the same speed.");
    Console.WriteLine("     5 -- Type this if you want to check a teensy.");
    Console.WriteLine("     6 -- Type this if you want to check a litte.");
    Console.WriteLine("     7 -- Type this if you want to check a lot.");
    Console.WriteLine("     8 -- Type this if you want to cheat and try to skip a gate.");
    Console.WriteLine();
    # 打印游戏提示
    Console.WriteLine(" The place to use these options is when the computer asks:");
    Console.WriteLine();
    Console.WriteLine("Option?");
    Console.WriteLine();
}
            Console.WriteLine("               Good Luck!");  # 输出"Good Luck!"到控制台
            Console.WriteLine();  # 输出空行到控制台
        }

        private bool PromptYesNo(string Prompt)  # 定义名为PromptYesNo的函数，接受一个字符串参数Prompt，返回布尔值
        {
            bool Success = false;  # 定义一个布尔变量Success并初始化为false

            while (!Success):  # 进入循环，条件为Success为false
                Console.Write(Prompt);  # 输出Prompt到控制台
                string LineInput = Console.ReadLine().Trim().ToLower();  # 从控制台读取输入并去除首尾空格，转换为小写存储在变量LineInput中

                if (LineInput.Equals("yes"))  # 如果LineInput等于"yes"
                    return true;  # 返回true
                else if (LineInput.Equals("no"))  # 如果LineInput等于"no"
                    return false;  # 返回false
                else  # 如果LineInput既不是"yes"也不是"no"
                    Console.WriteLine("Please type 'YES' or 'NO'");  # 输出提示到控制台
            }
            else
            {
                Console.WriteLine("Invalid input. Please enter a number between 1 and 25.");
            }
        }
        return NumberOfGates;
    }
```

注释：

1. 定义一个私有方法PromptForGates，用于提示用户输入课程的门数并返回门数值
2. 声明一个布尔类型的变量Success，用于标记用户输入是否成功
3. 声明一个整型变量NumberOfGates，用于存储用户输入的门数值
4. 使用while循环，当用户输入不成功时一直提示用户输入
5. 使用Console.Write方法提示用户输入课程的门数范围
6. 使用Console.ReadLine方法获取用户输入的字符串，并去除首尾空格并转换为小写
7. 使用int.TryParse方法将用户输入的字符串转换为整型，并将转换结果存储在NumberOfGates变量中
8. 判断用户输入的门数值是否在1到25之间
9. 如果用户输入的门数值在1到25之间，则将Success设置为true
10. 如果用户输入的门数值不在1到25之间，则提示用户重新输入
11. 返回用户输入的门数值
                    else if (NumberOfGates < 1)
                    {
                        Console.WriteLine("Try again,");
                    }
```
这段代码是一个条件语句，如果NumberOfGates小于1，则打印"Try again,"。

```python
                    else // greater than 25
                    {
                        Console.WriteLine("25 is the limit.");
                        NumberOfGates = 25;
                        Success = true;
                    }
```
这是另一个条件语句，如果NumberOfGates大于25，则打印"25 is the limit."，将NumberOfGates设置为25，并将Success设置为true。

```python
                else
                {
                    Console.WriteLine("Try again,");
                }
```
这是另一个条件语句的else分支，如果NumberOfGates不满足前面的条件，则打印"Try again,"。

```python
            }

            return NumberOfGates;
        }
```
这段代码结束了前面的条件语句块，并且返回NumberOfGates的值。
        # 定义一个私有方法，用于提示用户输入滑雪水平评分
        private int PromptForRate()
        {
            # 初始化一个布尔变量，用于标记用户输入是否成功
            bool Success = false;
            # 初始化一个整型变量，用于存储用户输入的评分
            int Rating = 0;

            # 循环直到用户输入成功
            while (!Success)
            {
                # 提示用户输入滑雪水平评分
                Console.Write("Rate yourself as a skier, (1=worst, 3=best) ");
                # 读取用户输入并去除首尾空格，并转换为小写
                string LineInput = Console.ReadLine().Trim().ToLower();

                # 判断用户输入是否为整数
                if (int.TryParse(LineInput, out Rating))
                {
                    # 判断用户输入的评分是否在1到3之间
                    if (Rating >= 1 && Rating <= 3)
                    {
                        # 如果评分在合理范围内，标记输入成功
                        Success = true;
                    }
                    else
                    {
                        # 如果评分不在合理范围内，提示用户评分范围
                        Console.WriteLine("The bounds are 1-3");
                    }
                }
                else
                {
                    Console.WriteLine("The bounds are 1-3");
                }
            }
```
这段代码是一个if-else语句，用于检查用户输入的值是否在1到3之间的范围内。如果是，则返回用户输入的值；如果不是，则打印错误信息。

```
            return Rating;
        }
```
这段代码是一个函数的结束标志，表示函数返回变量Rating的值。

```
        private int PromptForOption()
        {
            bool Success = false;
            int Option = 0;

            while (!Success)
            {
                Console.Write("Option? ");
                string LineInput = Console.ReadLine().Trim().ToLower();
```
这段代码是一个私有函数PromptForOption()的开始标志，该函数返回一个整数值。在函数内部，定义了一个布尔类型的变量Success和一个整数类型的变量Option，并且使用while循环来不断提示用户输入选项，直到用户输入合法的选项。
                # 如果输入的内容可以转换为整数，则将转换后的整数赋值给Option
                if (int.TryParse(LineInput, out Option))
                {
                    # 如果Option在0到8之间，则将Success设置为true
                    if (Option >= 0 && Option <= 8)
                    {
                        Success = true;
                    }
                    # 如果Option大于8，则输出"What?"
                    else if (Option > 8)
                    {
                        Console.WriteLine("What?");
                    }
                }
                # 如果输入的内容无法转换为整数，则输出"What?"
                else
                {
                    Console.WriteLine("What?");
                }
            }

            # 返回Option的值
            return Option;
        }
        # 定义一个名为 PromptForCommand 的私有方法，返回一个字符串类型的值
        private string PromptForCommand()
        {
            # 定义一个布尔类型的变量 Success，并初始化为 False
            bool Success = false;
            # 定义一个字符串类型的变量 Result，并初始化为空字符串
            string Result = "";

            # 在控制台输出空行
            Console.WriteLine();
            # 在控制台输出提示信息
            Console.WriteLine("Type \"INS\" for intructions");
            Console.WriteLine("Type \"MAX\" for approximate maximum speeds");
            Console.WriteLine("Type \"RUN\" for the beginning of the race");

            # 进入循环，直到 Success 变为 True
            while (!Success)
            {
                # 在控制台输出提示信息，并等待用户输入，去除首尾空格并转换为小写
                Console.Write("Command--? ");
                string LineInput = Console.ReadLine().Trim().ToLower();

                # 如果用户输入的命令是 "ins"、"max" 或 "run" 中的一个
                if (LineInput.Equals("ins") || LineInput.Equals("max") || LineInput.Equals("run"))
                {
                    # 将用户输入的命令赋值给 Result 变量
                    Result = LineInput;
                    # 将 Success 变量设置为 True，结束循环
                    Success = true;
                }
                else
                {
                    Console.WriteLine();
                    Console.WriteLine();
                    Console.WriteLine("\"{0}\" is an illegal command--retry", LineInput);
                }
            }
```
这部分代码是一个条件语句的结尾，用于判断条件成立时和不成立时的操作。

```
            return Result;
        }
```
这部分代码是一个函数的结尾，用于返回函数的结果。

```
        private bool ExceedGateSpeed(double MaxGateSpeed, double MPH, double Time)
        {
            Random rand = new Random();

            Console.WriteLine("{0:N0} M.P.H.", MPH);
```
这部分代码是一个函数的定义，用于判断车辆是否超过了最大门限速度，并且打印出当前的车速。

```
            if (MPH > MaxGateSpeed)
            {
                Console.Write("You went over the maximum speed ");
```
这部分代码是一个条件语句，用于判断车速是否超过了最大门限速度，如果超过则打印一条警告信息。
                # 如果随机生成的一个小数小于 ((MPH - MaxGateSpeed) * 0.1) + 0.2
                if (rand.NextDouble() < ((MPH - (double)MaxGateSpeed) * 0.1) + 0.2)
                {
                    # 打印 "and made it!"
                    Console.WriteLine("and made it!");
                }
                # 如果上述条件不满足
                else
                {
                    # 如果随机生成的一个小数小于 0.5
                    if (rand.NextDouble() < 0.5)
                    {
                        # 打印 "snagged a flag!"
                        Console.WriteLine("snagged a flag!");
                    }
                    # 如果上述条件不满足
                    else
                    {
                        # 打印 "wiped out!"
                        Console.WriteLine("wiped out!");
                    }

                    # 打印 "You took {0:N2} seconds"，其中 {0:N2} 会被替换为 rand.NextDouble() + Time 的值
                    Console.WriteLine("You took {0:N2} seconds", rand.NextDouble() + Time);

                    # 返回 false
                    return false;
                }
            }
            else if (MPH > (MaxGateSpeed - 1))  # 如果当前速度大于最大门速减1
            {
                Console.WriteLine("Close one!");  # 输出 "Close one!"
            }

            return true;  # 返回 true
        }
        private void DoARun(int NumberOfGates, int Rating)  # 定义一个名为 DoARun 的函数，接受 NumberOfGates 和 Rating 两个参数
        {
            Random rand = new Random();  # 创建一个 Random 类的实例 rand
            double MPH = 0;  # 初始化变量 MPH 为 0
            double Time = 0;  # 初始化变量 Time 为 0
            int Option = 0;  # 初始化变量 Option 为 0
            double MaxGateSpeed = 0; // Q  # 初始化变量 MaxGateSpeed 为 0
            double PreviousMPH = 0;  # 初始化变量 PreviousMPH 为 0
            double Medals = 0;  # 初始化变量 Medals 为 0

            Console.WriteLine("The starter counts down...5...4...3...2...1...GO!");  # 输出 "The starter counts down...5...4...3...2...1...GO!"

            MPH = rand.NextDouble() * (18-9)+9;  # 生成一个 9 到 18 之间的随机数，并赋值给 MPH
            // 输出空行
            Console.WriteLine();
            // 输出提示信息
            Console.WriteLine("You're off!");

            // 遍历每个门
            for (int GateNumber = 1; GateNumber <= NumberOfGates; GateNumber++)
            {
                // 获取当前门的最大速度
                MaxGateSpeed = GateMaxSpeed[GateNumber-1];

                // 输出空行
                Console.WriteLine();
                // 输出当前门的信息
                Console.WriteLine("Here comes Gate # {0}:", GateNumber);
                // 输出当前速度
                Console.WriteLine("{0:N0} M.P.H.", MPH);

                // 保存当前速度
                PreviousMPH = MPH;

                // 提示用户选择选项
                Option = PromptForOption();
                // 当用户选择为0时，循环直到用户选择其他选项
                while (Option == 0)
                {
                    // 输出已经花费的时间
                    Console.WriteLine("You've taken {0:N2} seconds.", Time);
                    // 提示用户选择选项
                    Option = PromptForOption();
                }
# 根据不同的选项进行不同的操作
switch (Option)
{
    # 如果选项为1，增加速度并检查是否超过最大门限速度，如果超过则跳出循环，否则返回
    case 1:
        MPH = MPH + (rand.NextDouble() * (10-5)+5);
        if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
            break;
        else
            return;
    # 如果选项为2，增加速度并检查是否超过最大门限速度，如果超过则跳出循环，否则返回
    case 2:
        MPH = MPH + (rand.NextDouble() * (5-3)+3);
        if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
            break;
        else
            return;
    # 如果选项为3，增加速度并检查是否超过最大门限速度，如果超过则跳出循环，否则...
    case 3:
        MPH = MPH + (rand.NextDouble() * (4-1)+1);
        if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
            break;
        else
# 返回空值
return;

# 情况4：如果超过了最大门速度，则跳出当前情况，否则返回
case 4:
    if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
        break;
    else
        return;

# 情况5：减去一个随机数后，如果超过了最大门速度，则跳出当前情况，否则返回
case 5:
    MPH = MPH - (rand.NextDouble() * (4-1)+1);
    if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
        break;
    else
        return;

# 情况6：减去一个随机数后，如果超过了最大门速度，则跳出当前情况，否则返回
case 6:
    MPH = MPH - (rand.NextDouble() * (5-3)+3);
    if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
        break;
    else
        return;

# 情况7：减去一个随机数后，如果超过了最大门速度，则跳出当前情况
case 7:
    MPH = MPH - (rand.NextDouble() * (10-5)+5);
                        if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))  # 如果超过了最大门速度
                            break;  # 结束当前循环
                        else
                            return;  # 返回当前函数
                    case 8:  # 情况8：作弊！
                        Console.WriteLine("***Cheat");  # 输出作弊信息
                        if (rand.NextDouble() < 0.7)  # 如果随机数小于0.7
                        {
                            Console.WriteLine("An official caught you!");  # 输出官员抓到你的信息
                            Console.WriteLine("You took {0:N2} seconds.", Time);  # 输出你花了多少秒

                            return;  # 返回当前函数
                        }
                        else
                        {
                            Console.WriteLine("You made it!");  # 输出你成功了
                            Time = Time + 1.5;  # 时间加上1.5秒
                        }
                        break;  # 结束当前情况
                }
                if (MPH < 7)  # 如果速度小于7
                {
                    Console.WriteLine("Let's be realistic, OK?  Let's go back and try again...");  # 输出提示信息
                    MPH = PreviousMPH;  # 将速度设置为之前的速度
                }
                else  # 否则
                {
                    Time = Time + (MaxGateSpeed - MPH + 1);  # 时间加上（最大门速度 - 速度 + 1）
                    if (MPH > MaxGateSpeed)  # 如果速度大于最大门速度
                    {
                        Time = Time + 0.5;  # 时间再加上0.5
                    }
                }
            }

            Console.WriteLine();  # 输出空行
            Console.WriteLine("You took {0:N2} seconds.", Time);  # 输出所花费的时间
            Medals = Time;  // 将时间赋值给奖牌变量
            Medals = Medals / NumberOfGates;  // 将奖牌数量除以门的数量，得到每个门的平均时间

            if (Medals < (1.5 - (Rating * 0.1)))  // 如果平均时间小于（1.5 - 评分 * 0.1），则赢得金牌
            {
                Console.WriteLine("You won a gold medal!");  // 输出赢得金牌的消息
                GoldMedals++;  // 金牌数量加一
            }
            else if (Medals < (2.9 - (Rating * 0.1)))  // 如果平均时间小于（2.9 - 评分 * 0.1），则赢得银牌
            {
                Console.WriteLine("You won a silver medal!");  // 输出赢得银牌的消息
                SilverMedals++;  // 银牌数量加一
            }
            else if (Medals < (4.4 - (Rating * 0.01)))  // 如果平均时间小于（4.4 - 评分 * 0.01），则赢得铜牌
            {
                Console.WriteLine("You won a bronze medal!");  // 输出赢得铜牌的消息
                BronzeMedals++;  // 铜牌数量加一
            }
        }
        private void PlayOneRound()
        {
            int NumberOfGates = 0;  // 初始化变量 NumberOfGates 为 0
            string Command = "first";  // 初始化变量 Command 为 "first"
            bool KeepPlaying = false;  // 初始化变量 KeepPlaying 为 false
            int Rating = 0;  // 初始化变量 Rating 为 0

            Console.WriteLine("");  // 在控制台输出空行

            NumberOfGates = PromptForGates();  // 调用 PromptForGates() 方法，将返回值赋给 NumberOfGates

            while (!Command.Equals(""))  // 当 Command 不等于空字符串时执行循环
            {
                Command = PromptForCommand();  // 调用 PromptForCommand() 方法，将返回值赋给 Command

                // Display instructions
                if (Command.Equals("ins"))  // 如果 Command 等于 "ins"，则执行以下代码
                {
                    DisplayInstructions();  // 调用 DisplayInstructions() 方法
                }
                else if (Command.Equals("max")) // 如果命令是"max"
                {
                    Console.WriteLine("Gate Max"); // 输出"Gate Max"
                    Console.WriteLine(" #  M.P.H."); // 输出" #  M.P.H."
                    Console.WriteLine("----------"); // 输出"----------"
                    for (int i = 0; i < NumberOfGates; i++) // 循环遍历每个门的最大速度
                    {
                        Console.WriteLine(" {0}     {1}", i+1, GateMaxSpeed[i]); // 输出门的编号和最大速度
                    }
                }
                else // 如果命令不是"max"，则进行比赛
                {
                    Rating = PromptForRate(); // 获取用户的评分

                    do
                    {
                        DoARun(NumberOfGates, Rating); // 进行一次比赛

                        KeepPlaying = PromptYesNo("Do you want to race again? "); // 提示用户是否要再次比赛
                    }
                    while (KeepPlaying);  // 当 KeepPlaying 为真时，执行循环体内的代码

                    Console.WriteLine("Thanks for the race");  // 打印感谢信息

                    if (GoldMedals > 0)  // 如果 GoldMedals 大于 0
                        Console.WriteLine("Gold Medals: {0}", GoldMedals);  // 打印金牌数量
                    if (SilverMedals > 0)  // 如果 SilverMedals 大于 0
                        Console.WriteLine("Silver Medals: {0}", SilverMedals);  // 打印银牌数量
                    if (BronzeMedals > 0)  // 如果 BronzeMedals 大于 0
                        Console.WriteLine("Bronze Medals: {0}", BronzeMedals);  // 打印铜牌数量

                    return;  // 返回
                }
            }
        }

        public void PlayTheGame()  // 定义 PlayTheGame 方法
        {
            DisplayIntro();  // 调用 DisplayIntro 方法
            PlayOneRound();  # 调用PlayOneRound()函数，执行一轮游戏
        }
    }
    class Program
    {
        static void Main(string[] args)
        {

            new Slalom().PlayTheGame();  # 创建Slalom对象并调用PlayTheGame()方法开始游戏

        }
    }
}
```