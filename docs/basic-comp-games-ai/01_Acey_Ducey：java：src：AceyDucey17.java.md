# `d:/src/tocomm/basic-computer-games\01_Acey_Ducey\java\src\AceyDucey17.java`

```
import java.util.Random;  // 导入 Random 类，用于生成随机数
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

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
    System.out.println("""
                                        ACEY DUCEY CARD GAME
                             CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY


              ACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER
    // 创建一个名为playGame的公共静态方法
    public static void playGame() {
        // 初始化一个名为cashOnHand的整数变量，并赋值为100，表示手头上的现金
        int cashOnHand = 100; // our only mutable variable  note [11]
        // 打印出手头上的现金数目
        System.out.println("YOU NOW HAVE  "+ cashOnHand +"  DOLLARS.");// note [6]
        // 当手头上的现金大于0时，执行以下循环
        while (cashOnHand > 0) {
            // 打印空行
            System.out.println();
            // 打印出下两张牌的信息
            System.out.println("HERE ARE YOUR NEXT TWO CARDS:");
            // 创建一个名为lowCard的最终卡片对象，调用getRandomCard方法生成一个2到KING之间的随机卡片
            final Card lowCard = Card.getRandomCard(2, Card.KING); //note [3]
      System.out.println(lowCard); // 打印低牌的信息
      final Card highCard = Card.getRandomCard(lowCard.rank() + 1, Card.ACE); // 生成比低牌高一点的随机牌
      System.out.println(highCard); // 打印高牌的信息

      final int bet = getBet(cashOnHand); // 获取下注金额
      final int winnings = determineWinnings(lowCard,highCard,bet); // 确定赢得的金额
      cashOnHand += winnings; // 更新手头现金
      if(winnings != 0 || cashOnHand != 0){  // 如果赢得的金额不为0或者手头现金不为0
        System.out.println("YOU NOW HAVE  "+ cashOnHand +"  DOLLARS."); // 打印当前手头现金
      }
    }
  }

  public static int determineWinnings(Card lowCard, Card highCard, int bet){
    if (bet <= 0) {    // 如果下注金额小于等于0
      System.out.println("CHICKEN!!"); // 打印“CHICKEN!!”
      return 0; // 返回赢得的金额为0
    }
    Card nextCard = Card.getRandomCard(2, Card.ACE); // 生成一张随机牌
    System.out.println(nextCard); // 打印随机牌的信息
    if(nextCard.between(lowCard,highCard)){  // 如果下一张卡片的值在最小值和最大值之间
      System.out.println("YOU WIN!!!");  // 打印“你赢了！”
      return bet;  // 返回赌注
    }
    System.out.println("SORRY, YOU LOSE");  // 打印“对不起，你输了”
    return -bet;  // 返回负赌注
  }

  public static boolean stillInterested(){  // 定义一个名为stillInterested的公共静态布尔型方法
    System.out.println();  // 打印空行
    System.out.println();  // 再次打印空行
    System.out.println("SORRY, FRIEND, BUT YOU BLEW YOUR WAD.");  // 打印“对不起，朋友，但你输光了。”
    System.out.println();  // 打印空行
    System.out.println();  // 再次打印空行
    System.out.print("TRY AGAIN (YES OR NO)? ");  // 打印“再试一次（是或否）？”
    Scanner input = new Scanner(System.in);  // 创建一个Scanner对象，用于接收用户输入
    boolean playAgain = input.nextLine()  // 读取用户输入的下一行
                             .toUpperCase()  // 将输入转换为大写
                             .startsWith("Y"); // 检查输入是否以“Y”开头，返回布尔值 // 注释 [9]
    System.out.println();  // 打印空行
    System.out.println();  // 打印空行
    return playAgain;  // 返回 playAgain 变量的值
  }

  public static int getBet(int cashOnHand){
    int bet;  // 声明一个整型变量 bet
    do{
      System.out.println();  // 打印空行
      System.out.print("WHAT IS YOUR BET? ");  // 打印提示信息
      bet = inputNumber();  // 调用 inputNumber 方法，将返回值赋给 bet
      if (bet > cashOnHand) {  // 如果 bet 大于 cashOnHand
        System.out.println("SORRY, MY FRIEND, BUT YOU BET TOO MUCH.");  // 打印提示信息
        System.out.println("YOU HAVE ONLY  "+cashOnHand+"  DOLLARS TO BET.");  // 打印提示信息
      }
    }while(bet > cashOnHand);  // 当 bet 大于 cashOnHand 时循环
    return bet;  // 返回 bet 变量的值
  }

  public static int inputNumber() {  // 声明一个返回整型值的方法
    final Scanner input = new Scanner(System.in);  // 创建一个 Scanner 对象
    // 将数字设置为负数，以标记为尚未输入（在输入错误的情况下）
    int number = -1;
    // 当数字小于0时循环，直到输入正确的数字
    while (number < 0) {
      try {
        // 尝试从输入中获取下一个整数
        number = input.nextInt();
      } catch(Exception ex) {   // 注意[7]：捕获异常
        // 如果捕获到异常，打印错误消息并重新输入
        System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");
        System.out.print("? ");
        try{
          // 清空输入缓冲区
          input.nextLine();
        }
        catch(Exception ns_ex){ // 捕获异常：接收到EOF（如果是Windows，则为ctrl-d或ctrl-z）
          // 如果接收到EOF，打印消息并退出程序
          System.out.println("END OF INPUT, STOPPING PROGRAM.");
          System.exit(1);
        }
      }
    }
    // 返回输入的数字
    return number;
  }
  record Card(int rank){  // 定义一个名为Card的记录类型，包含一个整数类型的rank字段
    // 用于描述花色牌的一些常量
    public static final int JACK = 11, QUEEN = 12, KING = 13, ACE = 14;
    private static final Random random = new Random();  // 创建一个私有的Random对象

    public static Card getRandomCard(int from, int to){
      return new Card(random.nextInt(from, to+1));  // 返回一个随机生成的Card对象，rank值在from和to之间
    }

    public boolean between(Card lower, Card higher){
      return lower.rank() < this.rank() && this.rank() < higher.rank();  // 判断当前牌的rank值是否在lower和higher之间
    }

    @Override
    public String toString() {  // 重写toString方法
      return switch (rank) {  // 使用switch语句根据rank的值返回不同的字符串
        case JACK -> "JACK";
        case QUEEN -> "QUEEN";
        case KING -> "KING";
        case ACE -> "ACE\n";  // 如果rank是ACE，则返回"ACE\n"
        default -> " "+rank+" "; // note [6]
      };
    }
  }
```

这部分代码是一个 switch 语句的一部分，根据不同的情况返回不同的字符串。

```
  /*
    Notes:
    1. Multiline strings, a.k.a. text blocks, were added in JDK15.
    2. The original game only displays the players balance if it changed,
       which it does not when the player chickens out and bets zero.
       It also doesn't display the balance when it becomes zero because it has
       a more appropriate message: Sorry, You Lose.
    3. To pick two cards to show, the original BASIC implementation has a
       bug that could cause a race condition if the RND function never chose
       a lower number first and higher number second. It loops infinitely
       re-choosing random numbers until the condition is met of the first
       one being lower. The logic is changed a bit here so that the first
       card picked is anything but an ACE, the highest possible card,
       and then the second card is between the just picked first card upto
       and including the ACE.
```

这部分是一段注释，解释了一些代码的细节和历史背景。包括了多行字符串的添加、原始游戏的一些显示逻辑、以及原始 BASIC 实现中的一个 bug。
抱歉，给定的代码片段并不是一个完整的程序，而且也没有提供任何需要注释的代码。如果您有其他需要帮助的地方，请随时告诉我。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```