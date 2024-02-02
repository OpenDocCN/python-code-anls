# `basic-computer-games\92_Trap\csharp\Program.cs`

```py
// 命名空间声明
namespace trap_cs
{
  // 程序类声明
  class Program
  {
    // 常量声明，最大猜测次数
    const int maxGuesses = 6;
    // 常量声明，最大数字
    const int maxNumber = 100;
    // 主函数
    static void Main(string[] args)
    {
      // 初始化低猜测值
      int lowGuess  = 0;
      // 初始化高猜测值
      int highGuess = 0;

      // 创建随机数生成器对象
      Random randomNumberGenerator = new ();

      // 打印字符串 "TRAP"
      Print("TRAP");
      // 打印字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
      Print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
      // 打印空行
      Print();
      // 打印空行
      Print();
      // 打印空行
      Print();

      // 打印游戏说明
      PrintInstructions();

      // 生成要猜测的随机数
      int numberToGuess = randomNumberGenerator.Next(1, maxNumber);

      // 循环猜测
      for (int nGuess = 1; nGuess <= maxGuesses + 1; nGuess++)
      {
        // 如果猜测次数超过最大次数
        if (nGuess > maxGuesses)
        {
          // 打印字符串，显示猜测次数和正确数字
          Print(string.Format("SORRY, THAT'S {0} GUESSES. THE NUMBER WAS {1}", maxGuesses, numberToGuess));
          // 打印空行
          Print();
          // 退出循环
          break;
        }

        // 获取玩家猜测的数字范围
        GetGuesses(nGuess, ref lowGuess, ref highGuess);

        // 如果玩家猜中了
        if(lowGuess == highGuess && lowGuess == numberToGuess)
        {
          // 打印字符串 "YOU GOT IT!!!"
          Print("YOU GOT IT!!!");
          // 打印空行
          Print();
          // 打印字符串 "TRY AGAIN."
          Print("TRY AGAIN.");
          // 打印空行
          Print();
          // 退出循环
          break;
        }
        // 如果玩家猜测的范围小于正确数字
        if (highGuess < numberToGuess)
        {
          // 打印字符串 "MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS."
          Print("MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS.");
        }
        // 如果玩家猜测的范围大于正确数字
        else if (lowGuess > numberToGuess)
        {
          // 打印字符串 "MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS."
          Print("MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS.");
        }
        // 如果玩家猜中了
        else
        {
          // 打印字符串 "YOU HAVE TRAPPED MY NUMBER."
          Print("YOU HAVE TRAPPED MY NUMBER.");
        }
      }
    }

    // 打印游戏说明
    static void PrintInstructions()
    {
      # 打印提示信息
      Print("INSTRUCTIONS ?");
    
      # 从控制台读取用户输入的字符
      char response = Console.ReadKey().KeyChar;
      # 如果用户输入的字符是 'Y'
      if (response == 'Y')
      {
        # 打印游戏规则和提示信息
        Print(string.Format("I AM THINKING OF A NUMBER BETWEEN 1 AND {0}", maxNumber));
        Print("TRY TO GUESS MY NUMBER. ON EACH GUESS,");
        Print("YOU ARE TO ENTER 2 NUMBERS, TRYING TO TRAP");
        Print("MY NUMBER BETWEEN THE TWO NUMBERS. I WILL");
        Print("TELL YOU IF YOU HAVE TRAPPED MY NUMBER, IF MY");
        Print("NUMBER IS LARGER THAN YOUR TWO NUMBERS, OR IF");
        Print("MY NUMBER IS SMALLER THAN YOUR TWO NUMBERS.");
        Print("IF YOU WANT TO GUESS ONE SINGLE NUMBER, TYPE");
        Print("YOUR GUESS FOR BOTH YOUR TRAP NUMBERS.");
        Print(string.Format("YOU GET {0} GUESSES TO GET MY NUMBER.", maxGuesses));
      }
    }
    # 定义一个打印字符串的函数
    static void Print(string stringToPrint)
    {
      Console.WriteLine(stringToPrint);
    }
    # 定义一个打印空行的函数
    static void Print()
    {
      Console.WriteLine();
    }
    # 定义一个获取用户猜测的函数
    static void GetGuesses(int nGuess, ref int lowGuess, ref int highGuess)
    {
      # 打印猜测的次数
      Print();
      Print(string.Format("GUESS #{0}", nGuess));
    
      # 从控制台获取用户输入的两个猜测数字
      lowGuess  = GetIntFromConsole("Type low guess");
      highGuess = GetIntFromConsole("Type high guess");
    
      # 如果低猜测值大于高猜测值，则交换它们的值
      if(lowGuess > highGuess)
      {
        int tempGuess = lowGuess;
    
        lowGuess = highGuess;
        highGuess = tempGuess;
      }
    }
    # 从控制台获取整数输入的函数
    static int GetIntFromConsole(string prompt)
    {
    
      Console.Write( prompt + " > ");
      string intAsString = Console.ReadLine();
    
      # 如果无法将输入的字符串转换为整数，则默认值为1
      if(int.TryParse(intAsString, out int intValue) ==false)
      {
        intValue = 1;
      }
    
      return intValue;
    }
    }
# 闭合前面的函数定义
```