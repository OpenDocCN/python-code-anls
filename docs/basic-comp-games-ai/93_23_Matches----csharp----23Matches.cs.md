# `93_23_Matches\csharp\23Matches.cs`

```
# Initialize 3 public variables so that they can be accessed anywhere in the code
numberOfMatches = 0
numberOfMatchesRemovedByPlayer = 0
playerGoesFirst = False  # a flag to show if the player won the coin toss

def Main(args):
    # Print introduction text

    # Prints the title with 31 spaces placed in front of the text using the PadLeft() string function
    print("23 MATCHES".rjust(31))
    print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".rjust(15))
    
    # Print 3 blank lines with \n escape sequence
      Console.Write("\n\n\n");  # 输出三个换行符
      Console.WriteLine(" THIS IS A GAME CALLED '23 MATCHES'.");  # 输出游戏名称
      Console.Write("\n");  # 输出一个换行符

      Console.WriteLine("WHEN IT IS YOUR TURN, YOU MAY TAKE ONE, TWO, OR THREE");  # 输出游戏规则
      Console.WriteLine("MATCHES. THE OBJECT OF THE GAME IS NOT TO HAVE TO TAKE");  # 输出游戏规则
      Console.WriteLine("THE LAST MATCH.");  # 输出游戏规则
      Console.Write("\n");  # 输出一个换行符
      Console.WriteLine("LET'S FLIP A COIN TO SEE WHO GOES FIRST.");  # 输出翻硬币决定先手
      Console.WriteLine("IF IT COMES UP HEADS, I WILL WIN THE TOSS.");  # 输出翻硬币结果
      Console.Write("\n");  # 输出一个换行符

      // Set the number of matches to 23
      numberOfMatches = 23;  # 设置比赛的火柴数量为23根

      // Create a random class object to generate the coin toss
      Random random = new Random();  # 创建一个随机数生成器对象
      // Generates a random number between 0.0 and 1.0
      // Multiplies that number by 2 and then
      // Converts it into an integer giving either a 0 or a 1
      # 生成一个随机数，模拟抛硬币的结果，0表示正面，1表示反面
      int coinTossResult = (int)(2 * random.NextDouble()); 

      # 如果抛硬币结果为1，表示反面，玩家先行
      if (coinTossResult == 1)
      {
        Console.WriteLine("TAILS! YOU GO FIRST. ");
        # 设置玩家先行的标志为true
        playerGoesFirst = true;
        # 玩家进行操作
        PlayerTurn();
      }
      # 如果抛硬币结果为0，表示正面，计算机先行
      else
      {
        Console.WriteLine("HEADS! I WIN! HA! HA!");
        Console.WriteLine("PREPARE TO LOSE, MEATBALL-NOSE!!");
        Console.Write("\n");
        Console.WriteLine("I TAKE 2 MATCHES");
        # 减去2根火柴
        numberOfMatches = numberOfMatches - 2;
      }

      # 循环直到火柴数量为1或更少
      do
      {
        // 检查玩家是否已经进行了游戏
        // 因为他们赢得了抛硬币
        // 如果他们还没有进行游戏，则玩家可以进行游戏
        if (playerGoesFirst == false)
        {
          Console.Write("THE NUMBER OF MATCHES IS NOW " + numberOfMatches);
          PlayerTurn();
        }
        // 将抛硬币标志设置为false，因为
        // 这只在代码的第一个循环中需要
        playerGoesFirst = false;
        ComputerTurn();        
      } while (numberOfMatches > 1);

    }

    static void PlayerTurn()
    {
      Console.WriteLine("\n");
      // 输出提示信息，告诉玩家轮到他们了，可以拿走1、2或3根火柴棍
      Console.WriteLine("YOUR TURN -- YOU MAY TAKE 1, 2, OR 3 MATCHES.");
      // 输出提示信息，询问玩家想要拿走多少根火柴棍
      Console.Write("HOW MANY DO YOU WISH TO REMOVE ?? ");
      // 获取玩家输入的数字
      numberOfMatchesRemovedByPlayer = ReadPlayerInput();
      // 如果输入的数字无效（不是1、2或3），则要求玩家重新输入
      while (numberOfMatchesRemovedByPlayer > 3 || numberOfMatchesRemovedByPlayer <= 0)
      {
        Console.WriteLine("VERY FUNNY! DUMMY!");
        Console.WriteLine("DO YOU WANT TO PLAY OR GOOF AROUND?");
        Console.Write("NOW, HOW MANY MATCHES DO YOU WANT                 ?? ");
        numberOfMatchesRemovedByPlayer = ReadPlayerInput();
      }

      // 减去玩家指定数量的火柴棍
      numberOfMatches = numberOfMatches - numberOfMatchesRemovedByPlayer;

      // 输出剩余的火柴棍数量
      Console.WriteLine("THE ARE NOW " + numberOfMatches + " MATCHES REMAINING");      
    }
    static void ComputerTurn()
      // 初始化计算机移除的火柴数量
      int numberOfMatchesRemovedByComputer = 0;
      // 根据火柴数量进行不同的情况处理
      switch (numberOfMatches)
      {
        case 4:
          numberOfMatchesRemovedByComputer = 3;
          break;
        case 3:
          numberOfMatchesRemovedByComputer = 2;
          break;
        case 2:
          numberOfMatchesRemovedByComputer = 1;
          break;
        case 1: case 0: // 如果计算机输了，则执行这个情况
          Console.WriteLine("YOU WON, FLOPPY EARS !");
          Console.WriteLine("THING YOU'RE PRETTY SMART !");
          Console.WriteLine("LETS PLAY AGAIN AND I'LL BLOW YOUR SHOES OFF !!");
          break;
        default: // 如果火柴数量大于4，则执行这个情况
          numberOfMatchesRemovedByComputer = 4 - numberOfMatchesRemovedByPlayer;  # 计算电脑移除的火柴数量，使得剩余火柴数量为4
          break;  # 跳出循环
      }
      // 如果 numberOfMatchesRemovedByComputer 已经被更新，则运行这段代码，
      // 如果没有被更新，那么电脑已经输了
      if (numberOfMatchesRemovedByComputer != 0)
      {
        Console.WriteLine("MY TURN ! I REMOVE " + numberOfMatchesRemovedByComputer + " MATCHES");  # 打印电脑移除的火柴数量
        numberOfMatches = numberOfMatches - numberOfMatchesRemovedByComputer;  # 更新剩余火柴数量
        // 如果剩余火柴数量少于或等于1
        // 那么玩家已经输了
        if (numberOfMatches <= 1)
        {
          Console.Write("\n");
          Console.WriteLine("YOU POOR BOOB! YOU TOOK THE LAST MATCH! I GOTCHA!!");  # 打印玩家输了的消息
          Console.WriteLine("HA ! HA ! I BEAT YOU !!!");  # 打印电脑赢了的消息
          Console.Write("\n");
          Console.WriteLine("GOOD BYE LOSER!");  # 打印结束游戏的消息
        }
      }
    // This method handles the player input 
    // and will handle inncorrect input
    static int ReadPlayerInput()
    {
      // Read user input and convert to integer
      int playerInput = 0;
      // Try to read player input
      try
      {
        playerInput = Convert.ToInt32(Console.ReadLine());  // 从控制台读取用户输入并转换为整数
      }
      // If there is an error in the player input
      catch (System.Exception)
      {
        Console.WriteLine("?REENTER");  // 如果玩家输入错误，提示重新输入
        Console.Write("?? ");  // 询问玩家重新输入
        // Ask the player to reenter their input
        playerInput = ReadPlayerInput();  # 从玩家输入中读取数据并存储在变量playerInput中
      }  # 结束if语句的代码块
      return playerInput;  # 返回存储在playerInput中的数据
    }  # 结束readPlayerInput方法的代码块

  }  # 结束Player类的代码块
}  # 结束命名空间的代码块
```