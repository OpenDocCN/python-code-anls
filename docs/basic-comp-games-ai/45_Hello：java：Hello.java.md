# `d:/src/tocomm/basic-computer-games\45_Hello\java\Hello.java`

```
import java.util.Scanner;  # 导入 java.util.Scanner 包，用于接收用户输入

/**
 * Game of Hello
 * <p>
 * Based on the BASIC game of Hello here
 * https://github.com/coding-horror/basic-computer-games/blob/main/45%20Hello/hello.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Hello {

  private static final int MONEY_WAIT_MS = 3000;  # 定义一个静态常量 MONEY_WAIT_MS，值为 3000

  private final boolean goodEnding = false;  # 定义一个私有的布尔类型变量 goodEnding，初始值为 false
  private final Scanner scan;  // For user input  // 声明一个私有的Scanner对象scan，用于用户输入

  public Hello() {

    scan = new Scanner(System.in);  // 在构造函数中初始化Scanner对象scan，使其可以从标准输入中读取用户输入

  }  // End of constructor Hello

  public void play() {

    showIntro();  // 调用showIntro方法，显示游戏介绍
    startGame();  // 调用startGame方法，开始游戏

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(32) + "HELLO");  // 打印"HELLO"，并在前面添加32个空格
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面添加14个空格
    System.out.println("\n\n");  // 打印两个换行符
  }  // End of method showIntro

  private void startGame() {

    boolean moreProblems = true;  // 声明并初始化一个布尔变量，用于控制游戏循环

    String userCategory = "";  // 声明并初始化一个空字符串变量，用于存储用户选择的类别
    String userName = "";  // 声明并初始化一个空字符串变量，用于存储用户输入的名字
    String userResponse = "";  // 声明并初始化一个空字符串变量，用于存储用户的回答

    // Name question
    System.out.println("HELLO.  MY NAME IS CREATIVE COMPUTER.\n\n");  // 打印欢迎信息
    System.out.print("WHAT'S YOUR NAME? ");  // 提示用户输入名字
    userName = scan.nextLine();  // 从用户输入中读取名字并存储
    System.out.println("");  // 打印空行

    // Enjoyment question
    System.out.print("HI THERE, " + userName + ", ARE YOU ENJOYING YOURSELF HERE? ");  // 打印问候信息和询问用户是否在这里玩得开心
    while (true):  # 进入循环，等待用户输入
      userResponse = scan.nextLine();  # 从用户输入中获取一行文本
      System.out.println("");  # 打印空行

      if (userResponse.toUpperCase().equals("YES")):  # 如果用户输入转换为大写后等于"YES"
        System.out.println("I'M GLAD TO HEAR THAT, " + userName + ".\n");  # 打印肯定回答的消息
        break;  # 退出循环
      elif (userResponse.toUpperCase().equals("NO")):  # 如果用户输入转换为大写后等于"NO"
        System.out.println("OH, I'M SORRY TO HEAR THAT, " + userName + ". MAYBE WE CAN");  # 打印否定回答的消息
        System.out.println("BRIGHTEN UP YOUR VISIT A BIT.");
        break;  # 退出循环
      else:  # 如果用户输入不是"YES"也不是"NO"
        System.out.println(userName + ", I DON'T UNDERSTAND YOUR ANSWER OF '" + userResponse + "'.");  # 打印无法理解用户回答的消息
        System.out.print("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE? ");  # 提示用户只能回答"YES"或"NO"
    // Category question  # 注释：类别问题
    // 输出空行
    System.out.println("");
    // 输出问候语和询问用户问题类型
    System.out.println("SAY, " + userName + ", I CAN SOLVE ALL KINDS OF PROBLEMS EXCEPT");
    System.out.println("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO");
    System.out.print("YOU HAVE (ANSWER SEX, HEALTH, MONEY, OR JOB)? ");

    // 循环处理用户问题
    while (moreProblems) {
      // 读取用户输入的问题类型
      userCategory = scan.nextLine();
      // 输出空行
      System.out.println("");

      // 处理性健康问题
      if (userCategory.toUpperCase().equals("SEX")) {
        // 询问用户问题的具体情况
        System.out.print("IS YOUR PROBLEM TOO MUCH OR TOO LITTLE? ");
        userResponse = scan.nextLine();
        System.out.println("");

        // 处理用户具体情况
        while (true) {
          if (userResponse.toUpperCase().equals("TOO MUCH")) {
            // 输出建议
            System.out.println("YOU CALL THAT A PROBLEM?!!  I SHOULD HAVE SUCH PROBLEMS!");
            System.out.println("IF IT BOTHERS YOU, " + userName + ", TAKE A COLD SHOWER.");
            // 结束内部循环
            break;
          }
          else if (userResponse.toUpperCase().equals("TOO LITTLE")) { // 如果用户输入的响应是"TOO LITTLE"
            System.out.println("WHY ARE YOU HERE IN SUFFERN, " + userName + "?  YOU SHOULD BE"); // 输出提示信息
            System.out.println("IN TOKYO OR NEW YORK OR AMSTERDAM OR SOMEPLACE WITH SOME");
            System.out.println("REAL ACTION.");
            break; // 跳出循环
          }
          else { // 如果用户输入的响应不是"TOO MUCH"也不是"TOO LITTLE"
            System.out.println("DON'T GET ALL SHOOK, " + userName + ", JUST ANSWER THE QUESTION"); // 输出提示信息
            System.out.print("WITH 'TOO MUCH' OR 'TOO LITTLE'.  WHICH IS IT? "); // 输出提示信息
            userResponse = scan.nextLine(); // 获取用户输入的响应
          }
        }
      }
      // Health advice
      else if (userCategory.toUpperCase().equals("HEALTH")) { // 如果用户选择的类别是"HEALTH"
        System.out.println("MY ADVICE TO YOU " + userName + " IS:"); // 输出提示信息
        System.out.println("     1.  TAKE TWO ASPRIN"); // 输出健康建议
        System.out.println("     2.  DRINK PLENTY OF FLUIDS (ORANGE JUICE, NOT BEER!)"); // 输出健康建议
        System.out.println("     3.  GO TO BED (ALONE)"); // 输出健康建议
      }
      // Money advice
      else if (userCategory.toUpperCase().equals("MONEY")) {
        // 打印给用户的建议
        System.out.println("SORRY, " + userName + ", I'M BROKE TOO.  WHY DON'T YOU SELL");
        System.out.println("ENCYCLOPEADIAS OR MARRY SOMEONE RICH OR STOP EATING");
        System.out.println("SO YOU WON'T NEED SO MUCH MONEY?");
      }
      // Job advice
      else if (userCategory.toUpperCase().equals("JOB")) {
        // 打印给用户的建议
        System.out.println("I CAN SYMPATHIZE WITH YOU " + userName + ".  I HAVE TO WORK");
        System.out.println("VERY LONG HOURS FOR NO PAY -- AND SOME OF MY BOSSES");
        System.out.println("REALLY BEAT ON MY KEYBOARD.  MY ADVICE TO YOU, " + userName + ",");
        System.out.println("IS TO OPEN A RETAIL COMPUTER STORE.  IT'S GREAT FUN.");
      }
      else {
        // 打印用户输入无法识别的提示
        System.out.println("OH, " + userName + ", YOUR ANSWER OF " + userCategory + " IS GREEK TO ME.");
      }

      // More problems question
      // 无限循环，直到用户输入有效的问题类别
      while (true) {
        System.out.println(""); // 打印空行
        System.out.print("ANY MORE PROBLEMS YOU WANT SOLVED, " + userName + "? "); // 打印提示信息，询问用户是否还有需要解决的问题
        userResponse = scan.nextLine(); // 从用户输入中获取回答
        System.out.println(""); // 打印空行

        if (userResponse.toUpperCase().equals("YES")) { // 如果用户回答是"YES"
          System.out.print("WHAT KIND (SEX, MONEY, HEALTH, JOB)? "); // 打印提示信息，询问用户需要解决的问题类型
          break; // 跳出循环
        }
        else if (userResponse.toUpperCase().equals("NO")) { // 如果用户回答是"NO"
          moreProblems = false; // 将moreProblems变量设为false
          break; // 跳出循环
        }
        else { // 如果用户回答既不是"YES"也不是"NO"
          System.out.println("JUST A SIMPLE 'YES' OR 'NO' PLEASE, " + userName + "."); // 打印提示信息，要求用户只能回答"YES"或"NO"
        }
      }
    }

    // Payment question
    # 输出空行
    System.out.println("");
    # 输出一条消息，包含用户名
    System.out.println("THAT WILL BE $5.00 FOR THE ADVICE, " + userName + ".");
    # 输出一条消息
    System.out.println("PLEASE LEAVE THE MONEY ON THE TERMINAL.");

    # 暂停一段时间
    try:
      Thread.sleep(MONEY_WAIT_MS)
    # 捕获异常并输出异常信息
    except (Exception e):
      System.out.println("Caught Exception: " + e.getMessage())

    # 输出两个空行
    System.out.println("\n\n")

    # 进入无限循环
    while (true):
      # 提示用户是否留下了钱，并获取用户输入
      System.out.print("DID YOU LEAVE THE MONEY? ")
      userResponse = scan.nextLine()
      System.out.println("")

      # 如果用户输入的是"YES"（不区分大小写），则输出一条消息
      if (userResponse.toUpperCase().equals("YES")):
        System.out.println("HEY, " + userName + "??? YOU LEFT NO MONEY AT ALL!")
        // 打印警告信息，指示用户作弊
        System.out.println("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.");
        // 打印空行
        System.out.println("");
        // 打印警告信息，指示用户欺骗
        System.out.println("WHAT A RIP OFF, " + userName + "!!!\n");
        // 跳出循环
        break;
      }
      // 如果用户回答是"NO"
      else if (userResponse.toUpperCase().equals("NO")) {
        // 打印信息，指示用户诚实
        System.out.println("THAT'S HONEST, " + userName + ", BUT HOW DO YOU EXPECT");
        System.out.println("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENTS");
        System.out.println("DON'T PAY THEIR BILLS?");
        // 跳出循环
        break;
      }
      // 如果用户回答不是"YES"或"NO"
      else {
        // 打印信息，指示用户回答不清晰
        System.out.println("YOUR ANSWER OF '" + userResponse + "' CONFUSES ME, " + userName + ".");
        System.out.println("PLEASE RESPOND WITH 'YES' OR 'NO'.");
      }
    }

    // 遗留的不可达代码
    if (goodEnding) {
      // 打印友好的结束信息
      System.out.println("NICE MEETING YOU, " + userName + ", HAVE A NICE DAY.");
    }
    else {
      // 打印空行
      System.out.println("");
      // 打印带有玩家用户名的消息
      System.out.println("TAKE A WALK, " + userName + ".\n");
    }

  }  // End of method startGame

  public static void main(String[] args) {
    // 创建 Hello 对象
    Hello hello = new Hello();
    // 调用 play 方法开始游戏
    hello.play();

  }  // End of method main

}  // End of class Hello
```