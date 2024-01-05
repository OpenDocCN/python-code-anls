# `24_Chemist\java\src\Chemist.java`

```
import java.util.Arrays;  # 导入 java.util.Arrays 包
import java.util.Scanner;  # 导入 java.util.Scanner 包

/**
 * Game of Chemist
 * <p>
 * Based on the Basic game of Chemist here
 * https://github.com/coding-horror/basic-computer-games/blob/main/24%20Chemist/chemist.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Chemist {

    public static final int MAX_LIVES = 9;  # 定义常量 MAX_LIVES，值为 9

    // Used for keyboard input
    private final Scanner kbScanner;  # 声明一个私有的 Scanner 对象 kbScanner，用于键盘输入

    private enum GAME_STATE {  # 声明一个枚举类型 GAME_STATE
        START_GAME,          // 定义游戏状态为开始游戏
        INPUT,               // 定义游戏状态为输入
        BLOWN_UP,            // 定义游戏状态为爆炸
        SURVIVED,            // 定义游戏状态为幸存
        GAME_OVER            // 定义游戏状态为游戏结束
    }

    // 当前游戏状态
    private GAME_STATE gameState;  // 声明游戏状态变量

    private int timesBlownUp;      // 声明变量记录爆炸次数

    public Chemist() {
        kbScanner = new Scanner(System.in);  // 创建一个用于接收键盘输入的Scanner对象

        gameState = GAME_STATE.START_GAME;   // 初始化游戏状态为开始游戏
    }

    /**
     * Main game loop
    */
    public void play() {

        do {
            switch (gameState) {

                case START_GAME:
                    // 游戏开始时播放介绍
                    intro();
                    // 初始化爆炸次数
                    timesBlownUp = 0;
                    // 切换游戏状态为输入状态
                    gameState = GAME_STATE.INPUT;
                    break;

                case INPUT:
                    // 生成随机的酸的数量
                    int amountOfAcid = (int) (Math.random() * 50);
                    // 计算正确的水的数量
                    int correctAmountOfWater = (7 * amountOfAcid) / 3;
                    // 获取玩家输入的水的数量
                    int water = displayTextAndGetNumber(amountOfAcid + " LITERS OF KRYPTOCYANIC ACID.  HOW MUCH WATER? ");

                    // 计算玩家是否混合了足够的水
                    int result = Math.abs(correctAmountOfWater - water);
                    // 如果水的比例错误
                    if (result > (correctAmountOfWater / 20)) {
                        gameState = GAME_STATE.BLOWN_UP;  // 游戏状态变为爆炸
                    } else {
                        // 比例正确
                        gameState = GAME_STATE.SURVIVED;  // 游戏状态变为幸存
                    }
                    break;

                case BLOWN_UP:
                    System.out.println(" SIZZLE!  YOU HAVE JUST BEEN DESALINATED INTO A BLOB");
                    System.out.println(" OF QUIVERING PROTOPLASM!");

                    timesBlownUp++;  // 爆炸次数加一

                    if (timesBlownUp < MAX_LIVES) {
                        System.out.println(" HOWEVER, YOU MAY TRY AGAIN WITH ANOTHER LIFE.");
                        gameState = GAME_STATE.INPUT;  // 如果还有生命，游戏状态变为输入
                    } else {
                        System.out.println(" YOUR " + MAX_LIVES + " LIVES ARE USED, BUT YOU WILL BE LONG REMEMBERED FOR");
                        // 打印玩家使用了多少条命，并且提醒玩家会因为对漫画书化学领域的贡献而被长久记住
                        System.out.println(" YOUR CONTRIBUTIONS TO THE FIELD OF COMIC BOOK CHEMISTRY.");
                        // 打印玩家对漫画书化学领域的贡献
                        gameState = GAME_STATE.GAME_OVER;
                        // 将游戏状态设置为 GAME_OVER
                    }

                    break;

                case SURVIVED:
                    System.out.println(" GOOD JOB! YOU MAY BREATHE NOW, BUT DON'T INHALE THE FUMES!");
                    // 打印玩家做得很好，并提醒玩家可以呼吸了，但不要吸入有毒气体
                    System.out.println();
                    // 打印空行
                    gameState = GAME_STATE.INPUT;
                    // 将游戏状态设置为 INPUT
                    break;

            }
        } while (gameState != GAME_STATE.GAME_OVER);
        // 当游戏状态不是 GAME_OVER 时继续循环
    }

    private void intro() {
        System.out.println(simulateTabs(33) + "CHEMIST");
        // 打印 CHEMIST，使用 simulateTabs 方法模拟出缩进效果
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        // 打印 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY，使用 simulateTabs 方法模拟出缩进效果
        // 打印空行
        System.out.println();
        // 打印虚构化学品KRYPTOCYANIC ACID的稀释说明
        System.out.println("THE FICTITIOUS CHEMICAL KRYPTOCYANIC ACID CAN ONLY BE");
        System.out.println("DILUTED BY THE RATIO OF 7 PARTS WATER TO 3 PARTS ACID.");
        System.out.println("IF ANY OTHER RATIO IS ATTEMPTED, THE ACID BECOMES UNSTABLE");
        System.out.println("AND SOON EXPLODES.  GIVEN THE AMOUNT OF ACID, YOU MUST");
        System.out.println("DECIDE WHO MUCH WATER TO ADD FOR DILUTION.  IF YOU MISS");
        System.out.println("YOU FACE THE CONSEQUENCES.");
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }
    # 打印屏幕上的消息，然后从键盘接受输入。
    # @param text 要显示在屏幕上的消息。
    # @return 玩家输入的内容。
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    # 模拟旧的基本tab(xx)命令，该命令通过xx个空格缩进文本。
    # @param spaces 需要的空格数
    # @return 具有空格数的字符串
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }
}
```

这部分代码是一个方法的结束标志，表示方法的结束。在这个示例中，它标志着 read_zip 方法的结束。
```