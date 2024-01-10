# `basic-computer-games\48_High_IQ\java\src\HighIQGame.java`

```
// 导入 Scanner 类
import java.util.Scanner;

// 定义 HighIQGame 类
public class HighIQGame {
    // 主函数
    public static void main(String[] args) {

        // 调用打印说明函数
        printInstructions();

        // 创建 Scanner 对象
        Scanner scanner = new Scanner(System.in);
        // 循环进行游戏
        do {
            // 创建 HighIQ 对象并开始游戏
            new HighIQ(scanner).play();
            // 打印提示信息
            System.out.println("PLAY AGAIN (YES OR NO)");
        } while(scanner.nextLine().equalsIgnoreCase("yes"));
    }

    // 打印游戏说明的函数
    public static void printInstructions() {
        // 打印游戏标题
        System.out.println("\t\t\t H-I-Q");
        // 打印游戏信息
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("HERE IS THE BOARD:");
        // 打印游戏棋盘
        System.out.println("          !    !    !");
        System.out.println("         13   14   15\n");
        System.out.println("          !    !    !");
        System.out.println("         22   23   24\n");
        System.out.println("!    !    !    !    !    !    !");
        System.out.println("29   30   31   32   33   34   35\n");
        System.out.println("!    !    !    !    !    !    !");
        System.out.println("38   39   40   41   42   43   44\n");
        System.out.println("!    !    !    !    !    !    !");
        System.out.println("47   48   49   50   51   52   53\n");
        System.out.println("          !    !    !");
        System.out.println("         58   59   60\n");
        System.out.println("          !    !    !");
        System.out.println("         67   68   69");
        // 打印游戏提示信息
        System.out.println("TO SAVE TYPING TIME, A COMPRESSED VERSION OF THE GAME BOARD");
        System.out.println("WILL BE USED DURING PLAY.  REFER TO THE ABOVE ONE FOR PEG");
        System.out.println("NUMBERS.  OK, LET'S BEGIN.");
    }
}
```