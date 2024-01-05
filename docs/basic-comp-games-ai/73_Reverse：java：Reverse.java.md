# `73_Reverse\java\Reverse.java`

```
import java.util.Scanner;  # 导入 Scanner 类，用于用户输入
import java.lang.Math;  # 导入 Math 类，用于数学运算

/**
 * Game of Reverse
 * <p>
 * Based on the BASIC game of Reverse here
 * https://github.com/coding-horror/basic-computer-games/blob/main/73%20Reverse/reverse.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Reverse {

  private final int NUMBER_COUNT = 9;  # 定义常量 NUMBER_COUNT，表示数字的个数

  private final Scanner scan;  // For user input  # 创建 Scanner 对象，用于用户输入
  private enum Step {
    INITIALIZE, PERFORM_REVERSE, TRY_AGAIN, END_GAME  # 定义一个枚举类型 Step，包含四个枚举值
  }

  public Reverse() {
    scan = new Scanner(System.in);  # 创建一个 Scanner 对象，用于从控制台读取输入
  }  // End of constructor Reverse

  public void play() {
    showIntro();  # 调用 showIntro 方法显示游戏介绍
    startGame();  # 调用 startGame 方法开始游戏
  }  // End of method play

  private static void showIntro() {
    # 这里是 showIntro 方法的具体实现，用于显示游戏介绍
    System.out.println(" ".repeat(31) + "REVERSE");  // 打印字符串，使用空格重复31次，用于显示游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印字符串，使用空格重复14次，用于显示游戏信息
    System.out.println("\n\n");  // 打印两个换行符，用于格式化输出
    System.out.println("REVERSE -- A GAME OF SKILL");  // 打印游戏标题
    System.out.println("");  // 打印空行

  }  // End of method showIntro  // 方法结束注释

  private void startGame() {  // 定义私有方法startGame

    int index = 0;  // 初始化整型变量index为0
    int numMoves = 0;  // 初始化整型变量numMoves为0
    int numReverse = 0;  // 初始化整型变量numReverse为0
    int tempVal = 0;  // 初始化整型变量tempVal为0
    int[] numList = new int[NUMBER_COUNT + 1];  // 创建整型数组numList，长度为NUMBER_COUNT + 1

    Step nextStep = Step.INITIALIZE;  // 初始化枚举类型变量nextStep为Step.INITIALIZE

    String userResponse = "";  // 初始化字符串变量userResponse为空字符串

    System.out.print("DO YOU WANT THE RULES? ");  // 打印提示信息，询问用户是否需要规则
    userResponse = scan.nextLine();  // 从用户输入中获取响应

    if (!userResponse.toUpperCase().equals("NO")) {  // 如果用户响应不是“NO”（不区分大小写）
      this.printRules();  // 调用 printRules 方法打印规则
    }

    // Begin outer while loop  // 开始外部 while 循环
    while (true) {  // 当条件为真时执行循环

    // Begin outer switch  // 开始外部 switch
    switch (nextStep) {  // 根据 nextStep 的值执行不同的操作

      case INITIALIZE:  // 如果 nextStep 的值为 INITIALIZE

        // Make a random list of numbers  // 生成一个随机数列表
       numList[1] = (int)((NUMBER_COUNT - 1) * Math.random() + 2);  // 生成一个随机数并赋值给 numList[1]

         for (index = 2; index <= NUMBER_COUNT; index++) {  // 循环遍历 index 从 2 到 NUMBER_COUNT
          // Keep generating lists if there are duplicates
          // 如果存在重复的数字，就一直生成新的列表
          while (true) {
            // Generate a random number between 1 and NUMBER_COUNT
            // 生成一个介于1和NUMBER_COUNT之间的随机数
            numList[index] = (int)(NUMBER_COUNT * Math.random() + 1);

            // Search for duplicates
            // 检查是否存在重复的数字
            if (!this.findDuplicates(numList, index)) {
              break;
            }
          }
        }

        System.out.println("");
        System.out.println("HERE WE GO ... THE LIST IS:");
        // 输出提示信息

        numMoves = 0;
        // 初始化移动次数为0

        this.printBoard(numList);
        // 调用printBoard方法打印生成的列表
        # 设置下一步操作为执行反转
        nextStep = Step.PERFORM_REVERSE;
        # 跳出 switch 语句
        break;

      case PERFORM_REVERSE:

        # 打印提示信息，询问需要反转多少个数字
        System.out.print("HOW MANY SHALL I REVERSE? ");
        # 从用户输入中获取需要反转的数字数量
        numReverse = Integer.parseInt(scan.nextLine());

        # 如果需要反转的数量为 0
        if (numReverse == 0) {
          # 设置下一步操作为再次尝试
          nextStep = Step.TRY_AGAIN;

        # 如果需要反转的数量超过了可用数字的数量
        } else if (numReverse > NUMBER_COUNT) {
          # 打印错误信息
          System.out.println("OOPS! TOO MANY! I CAN REVERSE AT MOST " + NUMBER_COUNT);
          # 设置下一步操作为执行反转
          nextStep = Step.PERFORM_REVERSE;

        # 如果需要反转的数量合理
        } else {
          # 增加操作次数
          numMoves++;
          // 反转数字列表中的数字顺序
          for (index = 1; index <= (int)(numReverse / 2.0); index++) {
            tempVal = numList[index];
            numList[index] = numList[numReverse - index + 1];
            numList[numReverse - index + 1] = tempVal;
          }

          // 打印反转后的数字列表
          this.printBoard(numList);

          // 设置下一步操作为尝试再次反转
          nextStep = Step.TRY_AGAIN;

          // 检查是否获胜
          for (index = 1; index <= NUMBER_COUNT; index++) {
            // 如果数字列表中的数字不等于索引值，则设置下一步操作为执行反转
            if (numList[index] != index) {
              nextStep = Step.PERFORM_REVERSE;
            }
          }
          if (nextStep == Step.TRY_AGAIN) {  # 如果下一步是要再试一次
            System.out.println("YOU WON IT IN " + numMoves + " MOVES!!!");  # 打印出赢得游戏所用的步数
            System.out.println("");  # 打印空行
          }
        }
        break;  # 结束当前的 case

      case TRY_AGAIN:  # 如果当前步骤是再试一次

        System.out.println("");  # 打印空行
        System.out.print("TRY AGAIN (YES OR NO)? ");  # 打印提示信息，要求用户输入是否再试一次
        userResponse = scan.nextLine();  # 获取用户输入的响应

        if (userResponse.toUpperCase().equals("YES")) {  # 如果用户响应是“YES”
          nextStep = Step.INITIALIZE;  # 设置下一步为初始化
        } else {  # 如果用户响应不是“YES”
          nextStep = Step.END_GAME;  # 设置下一步为结束游戏
        }
        break;  # 结束当前的 case
      case END_GAME:  // 当游戏结束时
        System.out.println("");  // 打印空行
        System.out.println("O.K. HOPE YOU HAD FUN!!");  // 打印消息“好的，希望你玩得开心！”
        return;  // 返回

      default:  // 默认情况
        System.out.println("INVALID STEP");  // 打印消息“无效的步骤”
        break;  // 跳出当前循环

      }  // 结束外部 switch

    }  // 结束外部 while 循环

  }  // 方法 startGame 结束

  public boolean findDuplicates(int[] board, int length) {  // 定义方法 findDuplicates，接受整数数组 board 和长度 length 作为参数

    int index = 0;  // 初始化变量 index 为 0
    for (index = 1; index <= length - 1; index++) {
      // 使用循环遍历数组中的元素，从第二个元素开始到倒数第二个元素
      // 这里的 length 是数组的长度
      // index 是循环变量，用于遍历数组

      // Identify duplicates
      // 判断数组中是否有重复的元素
      if (board[length] == board[index]) {
        // 如果发现重复的元素
        return true;  // Found a duplicate
        // 返回 true，表示找到了重复的元素
      }
    }

    return false;  // No duplicates found
    // 如果循环结束后没有发现重复的元素，返回 false

  }  // End of method findDuplicates
  // findDuplicates 方法结束

  public void printBoard(int[] board) {
    // 定义一个名为 printBoard 的公共方法，接受一个整型数组作为参数

    int index = 0;
    // 定义一个整型变量 index，初始化为 0

    System.out.println("");
    // 打印空行
```

    for (index = 1; index <= NUMBER_COUNT; index++) {
      // 循环遍历数组中的元素，打印每个元素的值
      System.out.format("%2d", board[index]);
    }
    // 打印换行
    System.out.println("\n");
  }  // End of method printBoard

  public void printRules() {
    // 打印空行
    System.out.println("");
    // 打印游戏规则说明
    System.out.println("THIS IS THE GAME OF 'REVERSE'.  TO WIN, ALL YOU HAVE");
    System.out.println("TO DO IS ARRANGE A LIST OF NUMBERS (1 THROUGH " + NUMBER_COUNT + ")");
    System.out.println("IN NUMERICAL ORDER FROM LEFT TO RIGHT.  TO MOVE, YOU");
    System.out.println("TELL ME HOW MANY NUMBERS (COUNTING FROM THE LEFT) TO");
    System.out.println("REVERSE.  FOR EXAMPLE, IF THE CURRENT LIST IS:");
    System.out.println("");
    System.out.println("2 3 4 5 1 6 7 8 9");
    System.out.println("");
  }
    # 打印提示信息
    print("AND YOU REVERSE 4, THE RESULT WILL BE:")
    print("")
    print("5 4 3 2 1 6 7 8 9")
    print("")
    print("NOW IF YOU REVERSE 5, YOU WIN!")
    print("")
    print("1 2 3 4 5 6 7 8 9")
    print("")
    print("NO DOUBT YOU WILL LIKE THIS GAME, BUT")
    print("IF YOU WANT TO QUIT, REVERSE 0 (ZERO).")
    print("")

  # 结束方法 printRules

  def main():
    # 创建 Reverse 对象
    game = Reverse()
    # 调用 play 方法开始游戏
    game.play()

  # 结束方法 main
}  # 类 Reverse 的结束
```
这行代码是一个注释，用于说明该行是类 Reverse 的结束。在 Python 中，使用 # 符号可以添加注释，用于解释代码的作用。
```