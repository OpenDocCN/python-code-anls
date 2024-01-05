# `d:/src/tocomm/basic-computer-games\64_Nicomachus\csharp\program.cs`

```
            # 显示提示信息
            Console.WriteLine(Prompt)
            # 读取用户输入的信息
            string input = Console.ReadLine();
            # 将用户输入的信息转换为小写
            input = input.ToLower();
            # 如果用户输入的信息包含"y"，返回True；否则返回False
            return input.Contains("y");
        }
    }
}
            bool Success = false;  # 定义一个布尔变量 Success，初始值为 false

            while (!Success)  # 当 Success 为 false 时执行循环
            {
                Console.Write(Prompt);  # 在控制台输出提示信息
                string LineInput = Console.ReadLine().Trim().ToLower();  # 从控制台读取用户输入并去除首尾空格，转换为小写

                if (LineInput.Equals("yes"))  # 如果用户输入为 "yes"
                    return true;  # 返回 true
                else if (LineInput.Equals("no"))  # 如果用户输入为 "no"
                    return false;  # 返回 false
                else  # 如果用户输入既不是 "yes" 也不是 "no"
                    Console.WriteLine("Eh?  I don't understand '{0}'  Try 'Yes' or 'No'.", LineInput);  # 在控制台输出错误信息
            }

            return false;  # 返回 false
        }

        private int PromptForNumber(string Prompt)  # 定义一个私有方法 PromptForNumber，接受一个字符串参数 Prompt
            # 声明一个布尔变量，用于标记输入是否成功
            bool InputSuccess = false;
            # 声明一个整型变量，用于存储返回结果
            int ReturnResult = 0;

            # 循环直到输入成功
            while (!InputSuccess)
            {
                # 提示用户输入
                Console.Write(Prompt);
                # 读取用户输入并去除两端空格
                string Input = Console.ReadLine().Trim();
                # 尝试将输入转换为整型，如果成功则将结果存入ReturnResult，并将InputSuccess标记为true
                InputSuccess = int.TryParse(Input, out ReturnResult);
                # 如果输入不成功，则提示用户重新输入
                if (!InputSuccess)
                    Console.WriteLine("*** Please enter a valid number ***");
            }   

            # 返回输入的结果
            return ReturnResult;
        }

        # 定义一个方法用于进行一轮游戏
        private void PlayOneRound()
        {
            # 创建一个随机数生成器对象
            Random rand = new Random();
            # 声明两个整型变量，用于存储A和B的数字
            int A_Number = 0;
            int B_Number = 0;
            int C_Number = 0;  // 初始化变量C_Number为0
            int D_Number = 0;  // 初始化变量D_Number为0

            Console.WriteLine();  // 输出空行
            Console.WriteLine("Please think of a number between 1 and 100.");  // 输出提示信息

            A_Number = PromptForNumber("Your number divided by 3 has a remainder of? ");  // 调用PromptForNumber函数，获取用户输入的A_Number
            B_Number = PromptForNumber("Your number divided by 5 has a remainder of? ");  // 调用PromptForNumber函数，获取用户输入的B_Number
            C_Number = PromptForNumber("Your number divided by 7 has a remainder of? ");  // 调用PromptForNumber函数，获取用户输入的C_Number

            Console.WriteLine();  // 输出空行
            Console.WriteLine("Let me think a moment...");  // 输出提示信息

            Thread.Sleep(2000);  // 程序暂停2000毫秒

            D_Number = 70 * A_Number + 21 * B_Number + 15 * C_Number;  // 计算D_Number的值

            while (D_Number > 105)  // 当D_Number大于105时执行循环
            {
                D_Number -= 105;  // D_Number减去105
            }

            if (PromptYesNo("Your number was " + D_Number.ToString() + ", right? "))
            {
                Console.WriteLine();
                Console.WriteLine("How about that!!");
            }
            else
            {
                Console.WriteLine();
                Console.WriteLine("I feel your arithmetic is in error.");
            }

            Console.WriteLine();

       }
```
这部分代码是一个方法的结束和另一个方法的开始。在这之前可能有一些其他的代码，但是在这里被省略了。

```
        public void Play()
        {
            bool ContinuePlay = true;
```
这部分代码是一个名为Play的公共方法的开始，该方法没有返回值(void)。它创建了一个名为ContinuePlay的布尔变量，并将其初始化为true。
            DisplayIntro();  // 调用 DisplayIntro() 函数，显示游戏介绍

            do 
            {
                PlayOneRound();  // 调用 PlayOneRound() 函数，进行一轮游戏

                ContinuePlay = PromptYesNo("Let's try another? ");  // 调用 PromptYesNo() 函数，询问是否继续游戏，并将结果赋给 ContinuePlay
            }
            while (ContinuePlay);  // 当 ContinuePlay 为 true 时继续循环

        }
    }
    class Program
    {
        static void Main(string[] args)
        {

            new Nicomachus().Play();  // 创建 Nicomachus 对象并调用其 Play() 方法

        }
这部分代码是一个函数的结束标志，表示函数的定义结束。在Python中，函数的定义是通过缩进来表示的，当缩进结束时，表示函数定义结束。
```