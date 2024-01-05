# `d:/src/tocomm/basic-computer-games\76_Russian_Roulette\java\src\RussianRoulette.java`

```
# 导入 java.util.Arrays 包
import java.util.Arrays;
# 导入 java.util.Scanner 包

/**
 * 俄罗斯轮盘游戏
 * <p>
 * 基于这里的基本俄罗斯轮盘游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/76%20Russian%20Roulette/russianroulette.bas
 * <p>
 * 注意：本意是在Java中创建一个1970年代Basic游戏的版本，没有引入新功能-没有添加额外的文本、错误检查等。
 */

public class RussianRoulette {

    public static final int BULLETS_IN_CHAMBER = 10;
    public static final double CHANCE_OF_GETTING_SHOT = .833333d;

    // 用于键盘输入
    private final Scanner kbScanner;
    private enum GAME_STATE {  # 定义了一个枚举类型 GAME_STATE，包含了四个状态：INIT、GAME_START、FIRE_BULLET、NEXT_VICTIM
        INIT,
        GAME_START,
        FIRE_BULLET,
        NEXT_VICTIM
    }

    // Current game state  # 定义了一个私有变量 gameState，用来表示当前游戏状态
    private GAME_STATE gameState;

    int bulletsShot;  # 定义了一个整型变量 bulletsShot，用来表示射出的子弹数

    public RussianRoulette() {  # 定义了一个构造函数 RussianRoulette
        kbScanner = new Scanner(System.in);  # 创建了一个 Scanner 对象 kbScanner，用来接收用户输入
        gameState = GAME_STATE.INIT;  # 将游戏状态初始化为 INIT
    }

    /**
     * Main game loop  # 主游戏循环
    */
    public void play() {
        // 执行游戏循环
        do {
            // 根据游戏状态进行不同的操作
            switch (gameState) {

                // 游戏初始化状态
                case INIT:
                    // 执行游戏介绍
                    intro();
                    // 将游戏状态设置为游戏开始
                    gameState = GAME_STATE.GAME_START;
                    break;

                // 游戏开始状态
                case GAME_START:
                    // 初始化射击次数
                    bulletsShot = 0;
                    // 打印提示信息
                    System.out.println();
                    System.out.println("HERE IS A REVOLVER.");
                    System.out.println("TYPE '1' TO SPIN CHAMBER AND PULL TRIGGER.");
                    System.out.println("TYPE '2' TO GIVE UP.");
                    // 将游戏状态设置为开火状态
                    gameState = GAME_STATE.FIRE_BULLET;
                    break;
                case FIRE_BULLET: // 当前状态为开枪

                    int choice = displayTextAndGetNumber("GO "); // 显示文本并获取数字

                    // 除了选择放弃以外的任何选择都会开枪
                    if (choice != 2) { // 如果选择不是放弃
                        bulletsShot++; // 子弹射出数加一
                        if (Math.random() > CHANCE_OF_GETTING_SHOT) { // 如果随机数大于被射中的几率
                            System.out.println("     BANG!!!!!   YOU'RE DEAD!"); // 打印玩家被射中的消息
                            System.out.println("CONDOLENCES WILL BE SENT TO YOUR RELATIVES."); // 打印慰问家属的消息
                            gameState = GAME_STATE.NEXT_VICTIM; // 游戏状态变为下一个受害者
                        } else if (bulletsShot > BULLETS_IN_CHAMBER) { // 如果射出的子弹数大于弹夹中的子弹数
                            System.out.println("YOU WIN!!!!!"); // 打印玩家获胜的消息
                            System.out.println("LET SOMEONE ELSE BLOW HIS BRAINS OUT."); // 打印让其他人自杀的消息
                            gameState = GAME_STATE.GAME_START; // 游戏状态变为游戏开始
                        } else {
                            // 哎呀，玩家在这一轮幸存下来了
                            System.out.println("- CLICK -"); // 打印扳机声
                        }
                    } else {
                        // 玩家放弃
                        System.out.println("     CHICKEN!!!!!");
                        gameState = GAME_STATE.NEXT_VICTIM;

                    }
                    break;

                case NEXT_VICTIM:
                    System.out.println("...下一个受害者...");
                    gameState = GAME_STATE.GAME_START;
            }
            // 无限循环 - 基于原始的基本版本
        } while (true);
    }

    private void intro() {
        System.out.println(addSpaces(28) + "俄罗斯轮盘");
        System.out.println(addSpaces(15) + "创意计算  新泽西州莫里斯敦");
        System.out.println();
# 打印屏幕上的消息，然后从键盘接受输入。将输入转换为整数
# @param text 在屏幕上显示的消息。
# @return 玩家输入的内容。
private int displayTextAndGetNumber(String text) {
    return Integer.parseInt(displayTextAndGetInput(text));
}

# 打印屏幕上的消息，然后从键盘接受输入。
# @param text 在屏幕上显示的消息。
# @return 玩家输入的内容。
    private String displayTextAndGetInput(String text) {
        // 打印文本
        System.out.print(text);
        // 从键盘输入获取用户输入
        return kbScanner.nextLine();
    }

    /**
     * 返回包含 x 个空格的字符串
     *
     * @param spaces 所需的空格数
     * @return 包含指定数量空格的字符串
     */
    private String addSpaces(int spaces) {
        // 创建一个包含指定数量空格的字符数组
        char[] spacesTemp = new char[spaces];
        // 用空格填充字符数组
        Arrays.fill(spacesTemp, ' ');
        // 将字符数组转换为字符串并返回
        return new String(spacesTemp);
    }

    public static void main(String[] args) {
        // 创建一个 RussianRoulette 对象
        RussianRoulette russianRoulette = new RussianRoulette();
        russianRoulette.play();  # 调用名为russianRoulette的对象的play方法
    }
}
```