# `d:/src/tocomm/basic-computer-games\48_High_IQ\java\src\HighIQGame.java`

```
import java.util.Scanner;  // 导入 Scanner 类，用于从控制台读取输入

public class HighIQGame {
    public static void main(String[] args) {

        printInstructions();  // 调用 printInstructions 方法打印游戏说明

        Scanner scanner = new Scanner(System.in);  // 创建 Scanner 对象，用于接收控制台输入
        do {
            new HighIQ(scanner).play();  // 创建 HighIQ 对象并调用 play 方法开始游戏
            System.out.println("PLAY AGAIN (YES OR NO)");  // 提示用户是否再次玩游戏
        } while(scanner.nextLine().equalsIgnoreCase("yes"));  // 当用户输入为 "yes" 时继续循环
    }

    public static void printInstructions() {
        System.out.println("\t\t\t H-I-Q");  // 打印游戏标题
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印游戏信息
        System.out.println("HERE IS THE BOARD:");  // 打印游戏棋盘
        System.out.println("          !    !    !");
        System.out.println("         13   14   15\n");
    }
        System.out.println("          !    !    !"); // 打印字符串
        System.out.println("         22   23   24\n"); // 打印字符串并换行
        System.out.println("!    !    !    !    !    !    !"); // 打印字符串
        System.out.println("29   30   31   32   33   34   35\n"); // 打印字符串并换行
        System.out.println("!    !    !    !    !    !    !"); // 打印字符串
        System.out.println("38   39   40   41   42   43   44\n"); // 打印字符串并换行
        System.out.println("!    !    !    !    !    !    !"); // 打印字符串
        System.out.println("47   48   49   50   51   52   53\n"); // 打印字符串并换行
        System.out.println("          !    !    !"); // 打印字符串
        System.out.println("         58   59   60\n"); // 打印字符串并换行
        System.out.println("          !    !    !"); // 打印字符串
        System.out.println("         67   68   69"); // 打印字符串
        System.out.println("TO SAVE TYPING TIME, A COMPRESSED VERSION OF THE GAME BOARD"); // 打印字符串
        System.out.println("WILL BE USED DURING PLAY.  REFER TO THE ABOVE ONE FOR PEG"); // 打印字符串
        System.out.println("NUMBERS.  OK, LET'S BEGIN."); // 打印字符串
    }
}
```