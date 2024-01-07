# `basic-computer-games\48_High_IQ\java\src\HighIQGame.java`

```

# 导入 Scanner 类
import java.util.Scanner;

# 定义 HighIQGame 类
public class HighIQGame {
    # 主函数
    public static void main(String[] args) {
        # 调用 printInstructions 函数打印游戏说明
        printInstructions();

        # 创建 Scanner 对象
        Scanner scanner = new Scanner(System.in);
        # 循环进行游戏
        do {
            # 创建 HighIQ 对象并开始游戏
            new HighIQ(scanner).play();
            # 打印提示信息
            System.out.println("PLAY AGAIN (YES OR NO)");
        } while(scanner.nextLine().equalsIgnoreCase("yes"));  # 当用户输入为"yes"时继续游戏
    }

    # 打印游戏说明
    public static void printInstructions() {
        # 打印游戏标题
        System.out.println("\t\t\t H-I-Q");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        # 打印游戏棋盘
        System.out.println("HERE IS THE BOARD:");
        # ...（打印棋盘具体内容）
        System.out.println("OK, LET'S BEGIN.");  # 打印开始游戏的提示
    }
}

```