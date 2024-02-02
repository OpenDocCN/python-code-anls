# `basic-computer-games\01_Acey_Ducey\java\src\AceyDucey17.java`

```py
import java.util.Random;
import java.util.Scanner;

/**
 * A modern version (JDK17) of ACEY DUCEY using post Java 8 features. Notes
 * regarding new java features or differences in the original basic
 * implementation are numbered and at the bottom of this code.
 * The goal is to recreate the exact look and feel of the original program
 * minus a large glaring bug in the original code that lets you cheat.
 */
public class AceyDucey17 {

  public static void main(String[] args) {
    // notes [1]
    // 打印游戏规则和介绍
    System.out.println("""
                                        ACEY DUCEY CARD GAME
                             CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY


              ACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER
              THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP
              YOU HAVE AN OPTION TO BET OR NOT BET DEPENDING
              ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE
              A VALUE BETWEEN THE FIRST TWO.
              IF YOU DO NOT WANT TO BET, INPUT A 0""");

    do {
      playGame();
    } while (stillInterested());
    // 打印游戏结束语
    System.out.println("O.K., HOPE YOU HAD FUN!");
  }

  public static void playGame() {
    int cashOnHand = 100; // our only mutable variable  note [11]
    // 打印玩家初始资金
    System.out.println("YOU NOW HAVE  "+ cashOnHand +"  DOLLARS.");// note [6]
    while (cashOnHand > 0) {
      System.out.println();
      System.out.println("HERE ARE YOUR NEXT TWO CARDS:");

      // 获取两张随机牌，第一张牌的最大值为2，第二张牌的最小值为第一张牌的值+1
      final Card lowCard = Card.getRandomCard(2, Card.KING); //note [3]
      System.out.println(lowCard);
      final Card highCard = Card.getRandomCard(lowCard.rank() + 1, Card.ACE);
      System.out.println(highCard);

      // 获取玩家下注金额
      final int bet = getBet(cashOnHand);
      // 确定玩家赢得的金额
      final int winnings = determineWinnings(lowCard,highCard,bet);
      cashOnHand += winnings;
      // 如果玩家赢得金额不为0或者玩家资金不为0，则打印玩家当前资金
      if(winnings != 0 || cashOnHand != 0){  //note [2]
        System.out.println("YOU NOW HAVE  "+ cashOnHand +"  DOLLARS.");//note [6]
      }
  }
}

// 确定赢得的赌注
public static int determineWinnings(Card lowCard, Card highCard, int bet){
  if (bet <= 0) {    // 如果赌注小于等于0
    System.out.println("CHICKEN!!");  // 输出"CHICKEN!!"
    return 0;  // 返回0
  }
  Card nextCard = Card.getRandomCard(2, Card.ACE);  // 获取一个随机的扑克牌
  System.out.println(nextCard);  // 输出随机扑克牌
  if(nextCard.between(lowCard,highCard)){  // 如果随机扑克牌在给定的两张牌之间
    System.out.println("YOU WIN!!!");  // 输出"YOU WIN!!!"
    return bet;  // 返回赌注
  }
  System.out.println("SORRY, YOU LOSE");  // 输出"SORRY, YOU LOSE"
  return -bet;  // 返回负赌注
}

// 是否还感兴趣
public static boolean stillInterested(){
  System.out.println();  // 输出空行
  System.out.println();  // 输出空行
  System.out.println("SORRY, FRIEND, BUT YOU BLEW YOUR WAD.");  // 输出"SORRY, FRIEND, BUT YOU BLEW YOUR WAD."
  System.out.println();  // 输出空行
  System.out.println();  // 输出空行
  System.out.print("TRY AGAIN (YES OR NO)? ");  // 输出"TRY AGAIN (YES OR NO)? "
  Scanner input = new Scanner(System.in);  // 创建一个Scanner对象
  boolean playAgain = input.nextLine()  // 获取用户输入的下一行
                           .toUpperCase()  // 转换为大写
                           .startsWith("Y"); // 检查是否以"Y"开头，返回布尔值
  System.out.println();  // 输出空行
  System.out.println();  // 输出空行
  return playAgain;  // 返回是否再次玩游戏的布尔值
}

// 获取赌注
public static int getBet(int cashOnHand){
  int bet;
  do{
    System.out.println();  // 输出空行
    System.out.print("WHAT IS YOUR BET? ");  // 输出"WHAT IS YOUR BET? "
    bet = inputNumber();  // 获取输入的赌注
    if (bet > cashOnHand) {  // 如果赌注大于手头现金
      System.out.println("SORRY, MY FRIEND, BUT YOU BET TOO MUCH.");  // 输出"SORRY, MY FRIEND, BUT YOU BET TOO MUCH."
      System.out.println("YOU HAVE ONLY  "+cashOnHand+"  DOLLARS TO BET.");  // 输出"YOU HAVE ONLY  "+cashOnHand+"  DOLLARS TO BET."
    }
  }while(bet > cashOnHand);  // 当赌注大于手头现金时重复
  return bet;  // 返回赌注
}

// 获取输入的数字
public static int inputNumber() {
  final Scanner input = new Scanner(System.in);  // 创建一个Scanner对象
  // 设置为负数以标记为尚未输入，以防输入错误。
  int number = -1;  // 初始化number为-1
    // 当输入的数字小于0时，进入循环
    while (number < 0) {
      // 尝试获取输入的整数
      try {
        number = input.nextInt();
      } catch(Exception ex) {   // note [7]
        // 捕获异常，输出错误信息并重新输入
        System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");
        System.out.print("? ");
        // 尝试清空输入行
        try{
          input.nextLine();
        }
        // 捕获异常，判断是否为输入结束
        catch(Exception ns_ex){ // received EOF (ctrl-d or ctrl-z if windows)
          System.out.println("END OF INPUT, STOPPING PROGRAM.");
          System.exit(1);
        }
      }
    }
    // 返回输入的数字
    return number;
  }

  // 定义卡片类，包含卡片等级的常量和随机生成卡片的方法
  record Card(int rank){
    // Some constants to describe face cards.
    // 定义描述花色卡片的常量
    public static final int JACK = 11, QUEEN = 12, KING = 13, ACE = 14;
    // 创建随机数生成器
    private static final Random random = new Random();

    // 静态方法，随机生成指定范围内的卡片
    public static Card getRandomCard(int from, int to){
      return new Card(random.nextInt(from, to+1));  // note [4]
    }

    // 判断当前卡片是否在两张给定卡片之间
    public boolean between(Card lower, Card higher){
      return lower.rank() < this.rank() && this.rank() < higher.rank();
    }

    // 重写toString方法，根据卡片等级返回对应的字符串
    @Override
    public String toString() { // note [13]
      return switch (rank) {
        case JACK -> "JACK";
        case QUEEN -> "QUEEN";
        case KING -> "KING";
        case ACE -> "ACE\n"; // note [10]
        default -> " "+rank+" "; // note [6]
      };
    }
  }

  /*
    Notes:
    1. Multiline strings, a.k.a. text blocks, were added in JDK15.
    2. The original game only displays the players balance if it changed,
       which it does not when the player chickens out and bets zero.
       It also doesn't display the balance when it becomes zero because it has
       a more appropriate message: Sorry, You Lose.
    3. 为了选择展示两张牌，原始的BASIC实现存在一个bug，可能会导致竞争条件，如果RND函数从未先选择较低的数字，然后选择较高的数字。它会无限循环重新选择随机数，直到满足第一个数字较低的条件。这里改变了逻辑，使得第一张选择的牌不是ACE，即最大的可能牌，然后第二张牌是在刚刚选择的第一张牌到ACE之间。
    4. Random.nextInt(origin, bound)在JDK17中添加，允许直接选择要生成的随机整数的范围。第二个参数是范围的上限，因此它们被陈述为+1，以适应面值卡。
    5. 原始的BASIC实现存在一个bug，允许负值下注。由于你不能下注比你拥有的现金更多，你总是可以下注更少，包括一个非常非常大的负值。当赢的机会很小或为零时，你会这样做，因为输掉一手牌会从你的现金中减去你的赌注；减去一个负数实际上会增加你的现金，可能让你成为即时亿万富翁。这个漏洞现在已经关闭。
    6. BASIC PRINT命令的微妙行为导致在所有正数之前打印一个空格，以及一个尾随空格。任何打印非面值卡或玩家余额的地方都有额外的空格，以模仿这种行为。
    # 7. 输入错误可能是特定于解释器的。该程序尝试匹配 Vintage Basic 解释器的错误消息。最终的 input.nextLine() 命令用于清除输入的非数字内容。但即使这样做也可能失败，如果用户输入 Ctrl-D（Windows 上的 Ctrl-Z），表示 EOF（文件结束），从而关闭 STDIN 通道。原始程序在收到 EOF 信号时打印 "END OF INPUT IN LINE 660"，因此我们以大致相同的方式处理它。所有这些都是为了避免在程序崩溃时打印混乱的堆栈跟踪。
    # 9. 原始游戏只接受全大写的 "YES" 来继续玩游戏，如果破产了。这个程序更宽容，将接受以字母 'y' 开头的任何输入，不区分大小写。
    # 10. 如果卡片是 ACE，原始游戏会打印一个额外的空行。似乎没有理由这样做。
    # 11. 现代 Java 最佳实践正在向更功能性的范式靠拢，因此不鼓励改变状态。除了 cashOnHand 之外，所有其他变量都是 final，并且只初始化一次。
    # 12. 通过记录对卡片的概念进行了重构。记录是在 JDK14 中引入的。卡片功能被封装在这个记录的示例中。枚举可能是一个更好的选择，因为从技术上讲只有 13 张卡片可能。
    # 13. Switch 表达式最早是在 JDK12 中引入的，但继续为了清晰性和完整性进行改进。截至 JDK17，可以通过启用预览功能来访问 switch 表达式的模式匹配。
# 闭合前面的代码块
```