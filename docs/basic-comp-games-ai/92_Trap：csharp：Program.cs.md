# `92_Trap\csharp\Program.cs`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
      # 打印游戏说明
      PrintInstructions();

      # 生成一个1到maxNumber之间的随机数作为要猜的数字
      int numberToGuess = randomNumberGenerator.Next(1, maxNumber);

      # 循环进行猜数字的操作，最多允许猜maxGuesses次
      for (int nGuess = 1; nGuess <= maxGuesses + 1; nGuess++)
      {
        # 如果猜测次数超过了最大允许次数，则打印出错信息并结束游戏
        if (nGuess > maxGuesses)
        {
          Print(string.Format("SORRY, THAT'S {0} GUESSES. THE NUMBER WAS {1}", maxGuesses, numberToGuess));
          Print();
          break;
        }

        # 获取玩家的猜测范围
        GetGuesses(nGuess, ref lowGuess, ref highGuess);

        # 如果玩家猜中了数字，则打印成功信息并结束游戏
        if(lowGuess == highGuess && lowGuess == numberToGuess)
        {
          Print("YOU GOT IT!!!");
          Print();
          # 打印"TRY AGAIN."
          Print("TRY AGAIN.")
          # 打印空行
          Print()
          # 跳出循环
          break
        }
        # 如果猜的数比要猜的数小
        if (highGuess < numberToGuess)
        {
          # 打印"MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS."
          Print("MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS.")
        }
        # 如果猜的数比要猜的数大
        else if (lowGuess > numberToGuess)
        {
          # 打印"MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS."
          Print("MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS.")
        }
        # 如果猜对了
        else
        {
          # 打印"YOU HAVE TRAPPED MY NUMBER."
          Print("YOU HAVE TRAPPED MY NUMBER.")
        }
      }
    }

// TRAP
// 打印游戏说明
static void PrintInstructions()
{
  // 打印提示信息
  Print("INSTRUCTIONS ?");

  // 读取用户输入的响应
  char response = Console.ReadKey().KeyChar;
  // 如果用户输入的是 'Y'
  if (response == 'Y')
  {
    // 打印游戏规则
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
    # 定义一个静态方法，用于打印字符串
    static void Print(string stringToPrint)
    {
      Console.WriteLine(stringToPrint);
    }
    # 定义一个静态方法，用于打印空行
    static void Print()
    {
      Console.WriteLine();
    }
    # 定义一个静态方法，用于获取猜测的数字范围
    static void GetGuesses(int nGuess, ref int lowGuess, ref int highGuess)
    {
      # 打印空行
      Print()
      # 打印猜测的次数
      Print(string.Format("GUESS #{0}", nGuess));

      # 从控制台获取用户输入的低猜测值
      lowGuess  = GetIntFromConsole("Type low guess");
      # 从控制台获取用户输入的高猜测值
      highGuess = GetIntFromConsole("Type high guess");

      # 如果低猜测值大于高猜测值，则交换它们的值
      if(lowGuess > highGuess)
      {
        int tempGuess = lowGuess;
        lowGuess = highGuess;  # 将变量 highGuess 的值赋给变量 lowGuess
        highGuess = tempGuess;  # 将变量 tempGuess 的值赋给变量 highGuess
      }  # 结束 if 语句的代码块
    }  # 结束 while 循环的代码块
    static int GetIntFromConsole(string prompt)  # 定义一个静态方法，用于从控制台获取整数输入
    {

      Console.Write( prompt + " > ");  # 在控制台打印提示信息
      string intAsString = Console.ReadLine();  # 从控制台读取用户输入的字符串

      if(int.TryParse(intAsString, out int intValue) ==false)  # 判断用户输入的字符串是否可以转换为整数
      {
        intValue = 1;  # 如果无法转换为整数，则将 intValue 的值设为 1
      }

      return intValue;  # 返回获取到的整数值
    }
  }
}
```