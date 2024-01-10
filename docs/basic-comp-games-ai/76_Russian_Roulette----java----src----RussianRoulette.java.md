# `basic-computer-games\76_Russian_Roulette\java\src\RussianRoulette.java`

```
// 导入必要的类库
import java.util.Arrays;
import java.util.Scanner;

/**
 * 俄罗斯轮盘游戏
 * <p>
 * 基于这里的基本俄罗斯轮盘游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/76%20Russian%20Roulette/russianroulette.bas
 * <p>
 * 注意：这个想法是在Java中创建一个1970年代基本游戏的版本，没有引入新功能-没有添加额外的文本，错误检查等。
 */
public class RussianRoulette {

    // 轮盘中的子弹数
    public static final int BULLETS_IN_CHAMBER = 10;
    // 被击中的概率
    public static final double CHANCE_OF_GETTING_SHOT = .833333d;

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private enum GAME_STATE {
        INIT,
        GAME_START,
        FIRE_BULLET,
        NEXT_VICTIM
    }

    // 当前游戏状态
    private GAME_STATE gameState;

    // 射出的子弹数
    int bulletsShot;

    // 构造函数
    public RussianRoulette() {
        kbScanner = new Scanner(System.in);
        gameState = GAME_STATE.INIT;
    }

    /**
     * 主游戏循环
     */
    private void intro() {
        // 打印游戏标题
        System.out.println(addSpaces(28) + "RUSSIAN ROULETTE");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THIS IS A GAME OF >>>>>>>>>>RUSSIAN ROULETTE.");
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.nextLine();
    }
}
    /**
     * 返回一个包含 x 个空格的字符串
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
        // 创建 RussianRoulette 对象
        RussianRoulette russianRoulette = new RussianRoulette();
        // 调用 play 方法开始游戏
        russianRoulette.play();
    }
# 闭合前面的函数定义
```