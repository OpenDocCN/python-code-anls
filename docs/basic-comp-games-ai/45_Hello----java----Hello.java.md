# `basic-computer-games\45_Hello\java\Hello.java`

```py
import java.util.Scanner;

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

  private static final int MONEY_WAIT_MS = 3000;

  private final boolean goodEnding = false;

  private final Scanner scan;  // For user input

  public Hello() {

    scan = new Scanner(System.in);

  }  // End of constructor Hello

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private static void showIntro() {

    // 打印游戏介绍
    System.out.println(" ".repeat(32) + "HELLO");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro

  private void startGame() {

    boolean moreProblems = true;

    String userCategory = "";
    String userName = "";
    String userResponse = "";

    // Name question
    // 打印问候语并获取用户输入的名字
    System.out.println("HELLO.  MY NAME IS CREATIVE COMPUTER.\n\n");
    System.out.print("WHAT'S YOUR NAME? ");
    userName = scan.nextLine();
    System.out.println("");

    // Enjoyment question
    // 打印问候语并获取用户对游戏的喜好
    System.out.print("HI THERE, " + userName + ", ARE YOU ENJOYING YOURSELF HERE? ");
    // 循环直到用户输入为"YES"或"NO"
    while (true) {
      // 读取用户输入
      userResponse = scan.nextLine();
      // 输出空行
      System.out.println("");

      // 如果用户输入为"YES"，输出对应消息并结束循环
      if (userResponse.toUpperCase().equals("YES")) {
        System.out.println("I'M GLAD TO HEAR THAT, " + userName + ".\n");
        break;
      }
      // 如果用户输入为"NO"，输出对应消息并结束循环
      else if (userResponse.toUpperCase().equals("NO")) {
        System.out.println("OH, I'M SORRY TO HEAR THAT, " + userName + ". MAYBE WE CAN");
        System.out.println("BRIGHTEN UP YOUR VISIT A BIT.");
        break;
      }
      // 如果用户输入既不是"YES"也不是"NO"，输出提示信息
      else {
        System.out.println(userName + ", I DON'T UNDERSTAND YOUR ANSWER OF '" + userResponse + "'.");
        System.out.print("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE? ");
      }
    }

    // 输出类别问题
    System.out.println("");
    System.out.println("SAY, " + userName + ", I CAN SOLVE ALL KINDS OF PROBLEMS EXCEPT");
    System.out.println("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO");
    System.out.print("YOU HAVE (ANSWER SEX, HEALTH, MONEY, OR JOB)? ");

    // 输出付款问题
    System.out.println("");
    System.out.println("THAT WILL BE $5.00 FOR THE ADVICE, " + userName + ".");
    System.out.println("PLEASE LEAVE THE MONEY ON THE TERMINAL.");

    // 暂停一段时间
    try {
      Thread.sleep(MONEY_WAIT_MS);
    } catch (Exception e) {
      System.out.println("Caught Exception: " + e.getMessage());
    }

    // 输出空行
    System.out.println("\n\n");
    // 进入循环，询问用户是否留下了钱
    while (true) {
      // 打印提示信息，询问用户是否留下了钱
      System.out.print("DID YOU LEAVE THE MONEY? ");
      // 读取用户输入
      userResponse = scan.nextLine();
      System.out.println("");

      // 如果用户回答是，则输出相关信息并结束循环
      if (userResponse.toUpperCase().equals("YES")) {
        System.out.println("HEY, " + userName + "??? YOU LEFT NO MONEY AT ALL!");
        System.out.println("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.");
        System.out.println("");
        System.out.println("WHAT A RIP OFF, " + userName + "!!!\n");
        break;
      }
      // 如果用户回答否，则输出相关信息并结束循环
      else if (userResponse.toUpperCase().equals("NO")) {
        System.out.println("THAT'S HONEST, " + userName + ", BUT HOW DO YOU EXPECT");
        System.out.println("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENTS");
        System.out.println("DON'T PAY THEIR BILLS?");
        break;
      }
      // 如果用户回答既不是是也不是否，则输出提示信息
      else {
        System.out.println("YOUR ANSWER OF '" + userResponse + "' CONFUSES ME, " + userName + ".");
        System.out.println("PLEASE RESPOND WITH 'YES' OR 'NO'.");
      }
    }

    // 遗留的不可达代码
    if (goodEnding) {
      // 如果是好结局，则输出问候信息
      System.out.println("NICE MEETING YOU, " + userName + ", HAVE A NICE DAY.");
    }
    else {
      System.out.println("");
      // 如果不是好结局，则输出提示信息
      System.out.println("TAKE A WALK, " + userName + ".\n");
    }

  }  // 方法 startGame 的结束

  // 主方法
  public static void main(String[] args) {

    // 创建 Hello 对象
    Hello hello = new Hello();
    // 调用 play 方法
    hello.play();

  }  // 方法 main 的结束
}  // 类 Hello 的结束
```