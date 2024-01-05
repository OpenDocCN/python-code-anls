# `33_Dice\java\src\Dice.java`

```
import java.util.Arrays;  // 导入 Arrays 类，用于操作数组
import java.util.Scanner;  // 导入 Scanner 类，用于接收键盘输入

/**
 * Game of Dice
 * <p>
 * Based on the Basic game of Dice here
 * https://github.com/coding-horror/basic-computer-games/blob/main/33%20Dice/dice.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Dice {

    // Used for keyboard input
    private final Scanner kbScanner;  // 创建 Scanner 对象，用于接收键盘输入

    private enum GAME_STATE {  // 创建枚举类型 GAME_STATE
        START_GAME,  // 游戏开始状态
        INPUT_AND_CALCULATE,  // 输入和计算状态
        RESULTS,  // 定义游戏结果的枚举值
        GAME_OVER  // 定义游戏结束的枚举值
    }

    // 当前游戏状态
    private GAME_STATE gameState;  // 声明一个私有的游戏状态变量

    private int[] spots;  // 声明一个整型数组变量

    public Dice() {  // 构造函数
        kbScanner = new Scanner(System.in);  // 创建一个用于接收用户输入的 Scanner 对象

        gameState = GAME_STATE.START_GAME;  // 初始化游戏状态为开始游戏
    }

    /**
     * Main game loop
     */
    public void play() {  // 游戏主循环函数
// 使用 do-while 循环来执行游戏状态的切换
do {
    // 根据游戏状态执行相应的操作
    switch (gameState) {

        // 游戏开始状态
        case START_GAME:
            // 执行游戏介绍
            intro();
            // 初始化长度为12的整型数组
            spots = new int[12];
            // 切换游戏状态到输入和计算
            gameState = GAME_STATE.INPUT_AND_CALCULATE;
            break;

        // 输入和计算状态
        case INPUT_AND_CALCULATE:
            // 获取用户输入的掷骰子次数
            int howManyRolls = displayTextAndGetNumber("HOW MANY ROLLS? ");
            // 循环执行掷骰子操作
            for (int i = 0; i < howManyRolls; i++) {
                // 生成两个骰子的点数并保存在数组中
                int diceRoll = (int) (Math.random() * 6 + 1) + (int) (Math.random() * 6 + 1);
                // 将骰子点数保存在以0为基础的数组中
                spots[diceRoll - 1]++;
            }
            // 切换游戏状态到结果展示
            gameState = GAME_STATE.RESULTS;
            break;
                case RESULTS: // 当游戏状态为RESULTS时
                    System.out.println("TOTAL SPOTS" + simulateTabs(8) + "NUMBER OF TIMES"); // 打印总点数和次数的表头
                    for (int i = 1; i < 12; i++) { // 循环12次，i从1开始，小于12结束
                        // 使用基于零的数组索引显示输出
                        System.out.println(simulateTabs(5) + (i + 1) + simulateTabs(20) + spots[i]); // 打印点数和对应的次数
                    }
                    System.out.println(); // 打印空行
                    if (yesEntered(displayTextAndGetInput("TRY AGAIN? "))) { // 如果输入是yes
                        gameState = GAME_STATE.START_GAME; // 将游戏状态设置为START_GAME
                    } else {
                        gameState = GAME_STATE.GAME_OVER; // 否则将游戏状态设置为GAME_OVER
                    }
                    break; // 结束switch语句
            }
        } while (gameState != GAME_STATE.GAME_OVER); // 当游戏状态不是GAME_OVER时循环
    }

    private void intro() { // intro方法
        System.out.println(simulateTabs(34) + "DICE"); // 打印"DICE"，并使用simulateTabs方法模拟缩进
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"); // 打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并使用simulateTabs方法模拟缩进
        System.out.println();  // 打印空行
        System.out.println("THIS PROGRAM SIMULATES THE ROLLING OF A");  // 打印提示信息
        System.out.println("PAIR OF DICE.");  // 打印提示信息
        System.out.println("YOU ENTER THE NUMBER OF TIMES YOU WANT THE COMPUTER TO");  // 打印提示信息
        System.out.println("'ROLL' THE DICE.  WATCH OUT, VERY LARGE NUMBERS TAKE");  // 打印提示信息
        System.out.println("A LONG TIME.  IN PARTICULAR, NUMBERS OVER 5000.");  // 打印提示信息
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));  // 将输入的内容转换为整数并返回
    }

    /*
# 在屏幕上打印一条消息，然后接受键盘输入。
# @param text 要显示在屏幕上的消息。
# @return 玩家输入的内容。
*/
private String displayTextAndGetInput(String text) {
    System.out.print(text);
    return kbScanner.next();
}

/**
# 检查玩家是否输入了Y或YES作为答案。
# @param text 从键盘输入的字符串
# @return 如果输入了Y或YES，则返回true，否则返回false
*/
private boolean yesEntered(String text) {
    return stringIsAnyValue(text, "Y", "YES");
}
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，'r'表示只读模式
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
    # 定义一个名为simulateTabs的私有方法，接受一个整数参数spaces
    def simulateTabs(spaces):
        # 创建一个长度为spaces的空格字符数组
        spacesTemp = [' ' for _ in range(spaces)]
        # 将字符数组转换为字符串并返回
        return ''.join(spacesTemp)
```