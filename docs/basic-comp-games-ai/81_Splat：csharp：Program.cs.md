# `d:/src/tocomm/basic-computer-games\81_Splat\csharp\Program.cs`

```
// 命名空间 Splat
namespace Splat
{
    // 定义 Splat 类
    class Splat
    {
        // 创建一个 ArrayList 类型的 DistanceLog 变量
        private ArrayList DistanceLog = new ArrayList();

        // 创建一个二维字符串数组 AccelerationData
        private string[][] AccelerationData =
        {
            // 初始化 AccelerationData 数组的第一个元素
            new string[] {"Fine. You're on Mercury. Acceleration={0} ft/sec/sec", "12.2"},
            // 初始化 AccelerationData 数组的第二个元素
            new string[] {"All right.  You're on Venus. Acceleration={0} ft/sec/sec", "28.3"},
            // 初始化 AccelerationData 数组的第三个元素
            new string[] {"Then you're on Earth. Acceleration={0} ft/sec/sec", "32.16"},
            // 初始化 AccelerationData 数组的第四个元素
            new string[] {"Fine. You're on the Moon. Acceleration={0} ft/sec/sec", "5.15"},
            // 初始化 AccelerationData 数组的第五个元素
            new string[] {"All right. You're on Mars. Acceleration={0} ft/sec/sec", "12.5"},
            // 初始化 AccelerationData 数组的第六个元素
            new string[] {"Then you're on Jupiter. Acceleration={0} ft/sec/sec", "85.2"},
            // 初始化 AccelerationData 数组的第七个元素
            new string[] {"Fine. You're on Saturn. Acceleration={0} ft/sec/sec", "37.6"},
            // 初始化 AccelerationData 数组的第八个元素
            new string[] {"All right. You're on Uranus. Acceleration={0} ft/sec/sec", "33.8"},
            // 初始化 AccelerationData 数组的第九个元素
            new string[] {"Then you're on Neptune. Acceleration={0} ft/sec/sec", "39.6"},
            new string[] {"Fine. You're on the Sun. Acceleration={0} ft/sec/sec", "896"}
        };
```
这段代码定义了一个名为`new`的字符串数组，其中包含了两个字符串元素。

```
        private void DisplayIntro()
        {
            Console.WriteLine("");
            Console.WriteLine("SPLAT".PadLeft(23));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine("");
            Console.WriteLine("Welcome to 'Splat' -- the game that simulates a parachute");
            Console.WriteLine("jump.  Try to open your chute at the last possible");
            Console.WriteLine("moment without going splat.");
            Console.WriteLine("");
        }
```
这段代码定义了一个名为`DisplayIntro`的方法，用于在控制台上显示游戏的介绍信息。

```
        private bool PromptYesNo(string Prompt)
        {
            bool Success = false;

            while (!Success)
```
这段代码定义了一个名为`PromptYesNo`的方法，用于在控制台上提示用户输入yes或no，并返回一个布尔值表示用户的选择。同时定义了一个布尔变量`Success`并初始化为`false`。
            {
                // 输出提示信息
                Console.Write(Prompt);
                // 读取用户输入并去除首尾空格，转换为小写
                string LineInput = Console.ReadLine().Trim().ToLower();

                // 如果用户输入为"yes"，返回true
                if (LineInput.Equals("yes"))
                    return true;
                // 如果用户输入为"no"，返回false
                else if (LineInput.Equals("no"))
                    return false;
                // 如果用户输入既不是"yes"也不是"no"，输出提示信息
                else
                    Console.WriteLine("Yes or No");
            }

            // 默认返回false
            return false;
        }

        // 写入随机的错误结果
        private void WriteRandomBadResult()
        {
           // 定义错误结果的字符串数组
           string[] BadResults = {"Requiescat in pace.","May the Angel of Heaven lead you into paradise.",
                "Rest in peace.","Son-of-a-gun.","#$%&&%!$","A kick in the pants is a boost if you're headed right.",
                "Hmmm. Should have picked a shorter time.","Mutter. Mutter. Mutter.","Pushing up daisies.",
                "Easy come, easy go."};  // 创建包含字符串的字符串数组

            Random rand = new Random();  // 创建一个随机数生成器对象

            Console.WriteLine(BadResults[rand.Next(BadResults.Length)]);  // 从字符串数组中随机选择一个字符串并打印出来
        }

        private void WriteColumnOutput(double Column1, double Column2)
        {

            Console.WriteLine("{0,-11:N3}    {1,-17:N2}", Column1, Column2);  // 格式化输出两个双精度数值

        }

        private void WriteColumnOutput(double Column1, string Column2)
        {

            Console.WriteLine("{0,-11:N3}    {1,-17}", Column1, Column2);  // 格式化输出一个双精度数值和一个字符串
        }
        private void WriteColumnOutput(string Column1, string Column2)
        {
            // 输出两列数据，第一列左对齐，第二列左对齐
            Console.WriteLine("{0,-11}    {1,-17}", Column1, Column2);
        }

        private void WriteSuccessfulResults(double Distance)
        {
            // 添加新的距离结果
            DistanceLog.Add(Distance);

            // 按距离排序
            DistanceLog.Sort();

            int ArrayLength = DistanceLog.Count;

            // 如果是第1、2或3次跳跃，则写入特殊消息
            if (ArrayLength <= 3)
            {
                // 打印消息，根据跳跃次数不同选择不同的序数词
                Console.Write("Amazing!!! Not bad for your ");
                if (ArrayLength == 1)
                    Console.Write("1st ");
                else if (ArrayLength == 2)
                    Console.Write("2nd ");
                else
                    Console.Write("3rd ");
                Console.WriteLine("successful jump!!!");
            }
            // 否则根据跳跃在列表中的位置写一条消息
            else
            {
                // 获取跳跃距离在列表中的位置
                int JumpPosition = DistanceLog.IndexOf(Distance);

                // 如果跳跃位置在列表末尾的10%以内
                if (ArrayLength - JumpPosition <= .1 * ArrayLength)
                {
                    Console.WriteLine("Wow! That's some jumping. Of the {0} successful jumps", ArrayLength);
                    Console.WriteLine("before yours, only {0} opened their chutes lower than", (ArrayLength - JumpPosition));
                    Console.WriteLine("you did.");
```
这是一个条件语句的分支，如果条件满足，则打印"you did."。

```python
                else if (ArrayLength - JumpPosition <= .25 * ArrayLength)
```
这是一个条件语句的分支，如果条件满足，则执行下面的代码块。

```python
                    Console.WriteLine("Pretty good! {0} successful jumps preceded yours and only", ArrayLength - 1);
                    Console.WriteLine("{0} of them got lower than you did before their chutes", (ArrayLength - 1 - JumpPosition));
                    Console.WriteLine("opened.");
```
在条件满足时，打印一系列描述成功跳伞次数和排名的信息。

```python
                else if (ArrayLength - JumpPosition <= .5 * ArrayLength)
```
这是一个条件语句的分支，如果条件满足，则执行下面的代码块。

```python
                    Console.WriteLine("Not bad. There have been  {0} successful jumps before yours.", ArrayLength - 1);
                    Console.WriteLine("You were beaten out by {0} of them.", (ArrayLength - 1 - JumpPosition));
```
在条件满足时，打印一系列描述成功跳伞次数和排名的信息。

```python
                else if (ArrayLength - JumpPosition <= .75 * ArrayLength)
```
这是一个条件语句的分支，如果条件满足，则执行下面的代码块。

```python
                    Console.WriteLine("Conservative aren't you? You ranked only {0} in the", (ArrayLength - JumpPosition));
                    Console.WriteLine("{0} successful jumps before yours.", ArrayLength - 1);
```
在条件满足时，打印一系列描述成功跳伞次数和排名的信息。

```python
                else if (ArrayLength - JumpPosition <= .9 * ArrayLength)
```
这是一个条件语句的分支，如果条件满足，则执行下面的代码块。
                    Console.WriteLine("Humph! Don't you have any sporting blood? There were");
                    Console.WriteLine("{0} successful jumps before yours and you came in {1} jumps", ArrayLength - 1, JumpPosition);
                    Console.WriteLine("better than the worst. Shape up!!!");
                }
                else
                {
                    Console.WriteLine("Hey! You pulled the rip cord much too soon. {0} successful", ArrayLength - 1);
                    Console.WriteLine("jumps before yours and you came in number {0}! Get with it!", (ArrayLength - JumpPosition));
                }
            }
```
这部分代码是一个条件语句，根据条件输出不同的提示信息。

```
        }

        private void PlayOneRound()
        {
            bool InputSuccess = false;
            Random rand = new Random();
            double Velocity = 0;
            double TerminalVelocity = 0;
            double Acceleration = 0;
```
这部分代码定义了一个名为PlayOneRound的方法，方法内部定义了几个变量，包括InputSuccess、rand、Velocity、TerminalVelocity和Acceleration。
            // 初始化加速度输入为0
            double AccelerationInput = 0;
            // 初始化高度为随机生成的1000到9001之间的值
            double Altitude = ((9001 * rand.NextDouble()) + 1000);
            // 初始化计时器为0
            double SecondsTimer = 0;
            // 初始化距离为0
            double Distance = 0;
            // 初始化是否达到最终速度为false
            bool TerminalVelocityReached = false;

            // 输出空行
            Console.WriteLine("");

            // 确定最终速度（用户或系统）
            if (PromptYesNo("选择自己的最终速度（是或否）？ "))
            {
                // 提示用户输入他们选择的最终速度
                while (!InputSuccess)
                {
                    Console.Write("什么是最终速度（mi/hr）？ ");
                    string Input = Console.ReadLine().Trim();
                    // 尝试将输入转换为double类型的最终速度
                    InputSuccess = double.TryParse(Input, out TerminalVelocity);
                    if (!InputSuccess)
                        Console.WriteLine("*** 请输入有效的数字 ***");
                 }
            }
            else
            {
                // 生成一个随机的终端速度
                TerminalVelocity = rand.NextDouble() * 1000;
                // 打印终端速度
                Console.WriteLine("OK.  Terminal Velocity = {0:N0} mi/hr", (TerminalVelocity));
            }

            // 将终端速度转换为英尺/秒
            TerminalVelocity = TerminalVelocity * 5280 / 3600;

            // 不确定这个计算是什么意思
            Velocity = TerminalVelocity + ((TerminalVelocity * rand.NextDouble()) / 20) - ((TerminalVelocity * rand.NextDouble()) / 20);

            // 确定由于重力引起的加速度（用户或系统）
            if (PromptYesNo("Want to select acceleration due to gravity (yes or no)? "))
            {
                 // 提示用户输入他们选择的加速度
                InputSuccess = false;
                while (!InputSuccess)
                {
                    Console.Write("What acceleration (ft/sec/sec)? ");  // 提示用户输入加速度信息
                    string Input = Console.ReadLine().Trim();  // 读取用户输入的加速度信息并去除首尾空格
                    InputSuccess = double.TryParse(Input, out AccelerationInput);  // 尝试将用户输入的加速度信息转换为 double 类型，并将结果存储在 AccelerationInput 中
                    if (!InputSuccess)  // 如果转换不成功
                        Console.WriteLine("*** Please enter a valid number ***");  // 提示用户输入有效的数字
                 }
            }
            else
            {
                // 从数据数组中随机选择一个加速度条目
                int Index = rand.Next(0, AccelerationData.Length);  // 生成一个随机数作为索引
                Double.TryParse(AccelerationData[Index][1], out AccelerationInput);  // 尝试将选定的加速度信息转换为 double 类型，并将结果存储在 AccelerationInput 中

                // 显示该加速度所在的行星以及数值
                Console.WriteLine(AccelerationData[Index][0], AccelerationInput.ToString());  // 输出选定的加速度所在的行星和对应的数值
            }

            Acceleration = AccelerationInput + ((AccelerationInput * rand.NextDouble()) / 20) - ((AccelerationInput * rand.NextDouble()) / 20);  // 根据用户输入的加速度信息和随机数计算最终的加速度值

            Console.WriteLine("");  // 输出空行
            // 打印高度
            Console.WriteLine("    Altitude         = {0:N0} ft", Altitude);
            // 打印终端速度
            Console.WriteLine("    Term. Velocity   = {0:N3} ft/sec +/-5%", TerminalVelocity);
            // 打印加速度
            Console.WriteLine("    Acceleration     = {0:N2} ft/sec/sec +/-5%", AccelerationInput);
            // 设置倒计时器
            Console.WriteLine("Set the timer for your freefall.");

            // 提示用户在打开降落伞之前自由落体的时间
            InputSuccess = false;
            while (!InputSuccess)
            {
                Console.Write("How many seconds? ");
                string Input = Console.ReadLine().Trim();
                InputSuccess = double.TryParse(Input, out SecondsTimer);
                if (!InputSuccess)
                    Console.WriteLine("*** Please enter a valid number ***");
            }

            // 开始自由落体
            Console.WriteLine("Here we go.");
            Console.WriteLine("");
            // 输出表头
            WriteColumnOutput("Time (sec)", "Dist to Fall (ft)");
            WriteColumnOutput("==========", "=================");

            // 循环遍历秒数，每次增加8个间隔
            for (double i = 0; i < SecondsTimer; i+=(SecondsTimer/8))
            {
                // 如果时间超过了达到终端速度所需的时间
                if (i > (Velocity / Acceleration))
                {
                    // 终端速度已达到，只打印警告一次
                    if (TerminalVelocityReached == false)
                        Console.WriteLine("Terminal velocity reached at T plus {0:N4} seconds.", (Velocity / Acceleration));

                    TerminalVelocityReached = true;
                }

                // 根据是否达到终端速度来计算距离
                if (TerminalVelocityReached)
                {
                    Distance = Altitude - ((Math.Pow(Velocity,2) / (2 * Acceleration)) + (Velocity * (i - (Velocity / Acceleration))));
                }
                else
                {
                    // 根据公式计算物体的下落距离
                    Distance = Altitude - ((Acceleration / 2) * Math.Pow(i,2));
                }

                // 判断是否触地，如果是，则输出"SPLAT!"
                if (Distance <= 0)
                {
                    // 如果达到了最终速度，则根据公式输出结果，否则根据另一个公式输出结果
                    if (TerminalVelocityReached)
                    {
                        WriteColumnOutput((Velocity / Acceleration) + ((Altitude - (Math.Pow(Velocity,2) / (2 * Acceleration))) / Velocity).ToString(), "SPLAT");
                    }
                    else
                    {
                        WriteColumnOutput(Math.Sqrt(2 * Altitude / Acceleration), "SPLAT");
                    }

                    // 输出随机的坏结果
                    WriteRandomBadResult();

                    // 输出"I'll give you another chance."
                    Console.WriteLine("I'll give you another chance.");
                break;  // 跳出循环，结束当前循环的执行
            }
            else  // 如果条件不满足
            {
                WriteColumnOutput(i, Distance);  // 调用函数，输出当前跳跃的信息
            }
        }

        // 如果跳跃结束时还在空中，则跳伞成功
        if (Distance > 0)  // 如果距离大于0
        {
            // 成功跳伞！打开降落伞！
            Console.WriteLine("Chute Open");  // 在控制台输出信息

            // 存储成功的跳伞记录，并输出一条有趣的消息
            WriteSuccessfulResults(Distance);  // 调用函数，输出成功跳伞的信息
        }

    }
# 定义一个名为 PlayTheGame 的方法，用于控制游戏的进行
public void PlayTheGame()
{
    # 定义一个布尔变量 ContinuePlay，用于控制游戏是否继续进行
    bool ContinuePlay = false;

    # 显示游戏的介绍
    DisplayIntro();

    # 使用 do-while 循环来控制游戏的进行
    do
    {
        # 调用 PlayOneRound 方法，进行一轮游戏
        PlayOneRound();

        # 提示玩家是否想要再玩一次，并将结果存储在 ContinuePlay 变量中
        ContinuePlay = PromptYesNo("Do you want to play again? ");
        
        # 如果玩家不想再玩一次，则再次提示玩家是否愿意再玩一次，并将结果存储在 ContinuePlay 变量中
        if (!ContinuePlay)
            ContinuePlay = PromptYesNo("Please? ");
    }
    # 当 ContinuePlay 变量为 true 时继续循环
    while (ContinuePlay);

    # 打印游戏结束的消息
    Console.WriteLine("SSSSSSSSSS.");
}
    # 定义一个类 Program
    class Program
    {
        # 定义一个静态方法 Main，接受参数 args
        static void Main(string[] args)
        {
            # 创建一个 Splat 类的实例，并调用其 PlayTheGame 方法
            new Splat().PlayTheGame();
        }
    }
}
```

这段代码是一个 C# 程序的入口点，定义了一个类 Program 和一个静态方法 Main，其中创建了一个 Splat 类的实例并调用了其 PlayTheGame 方法。
```