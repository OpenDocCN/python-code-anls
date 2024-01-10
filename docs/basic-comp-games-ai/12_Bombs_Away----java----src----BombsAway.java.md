# `basic-computer-games\12_Bombs_Away\java\src\BombsAway.java`

```
import java.util.Scanner;

/**
 * Game of Bombs Away
 *
 * Based on the Basic game of Bombs Away here
 * https://github.com/coding-horror/basic-computer-games/blob/main/12_Bombs_Away/bombsaway.bas
 *
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without adding new features.
 * Obvious bugs where found have been fixed, but the playability and overlook and feel
 * of the game have been faithfully reproduced.
 *
 * Modern Java coding conventions have been employed and JDK 11 used for maximum compatibility.
 *
 * Java port by https://github.com/journich
 *
 */
public class BombsAway {

    // 定义常量，最大飞行员任务数
    public static final int MAX_PILOT_MISSIONS = 160;
    // 定义常量，最大伤亡人数
    public static final int MAX_CASUALTIES = 100;
    // 定义常量，未命中目标的常数1
    public static final int MISSED_TARGET_CONST_1 = 2;
    // 定义常量，未命中目标的常数2
    public static final int MISSED_TARGET_CONST_2 = 30;
    // 定义常量，被击落的基础概率
    public static final int CHANCE_OF_BEING_SHOT_DOWN_BASE = 100;
    // 定义常量，65%的概率
    public static final double SIXTY_FIVE_PERCENT = .65;

    // 定义游戏状态枚举
    private enum GAME_STATE {
        START,
        CHOOSE_SIDE,
        CHOOSE_PLANE,
        CHOOSE_TARGET,
        CHOOSE_MISSIONS,
        CHOOSE_ENEMY_DEFENCES,
        FLY_MISSION,
        DIRECT_HIT,
        MISSED_TARGET,
        PROCESS_FLAK,
        SHOT_DOWN,
        MADE_IT_THROUGH_FLAK,
        PLAY_AGAIN,
        GAME_OVER
    }

    // 定义阵营枚举
    public enum SIDE {
        ITALY(1),
        ALLIES(2),
        JAPAN(3),
        GERMANY(4);

        private final int value;

        SIDE(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    // 定义目标枚举
    public enum TARGET {
        ALBANIA(1),
        GREECE(2),
        NORTH_AFRICA(3),
        RUSSIA(4),
        ENGLAND(5),
        FRANCE(6);

        private final int value;

        TARGET(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }
}
    // 定义敌方防御方式的枚举类型
    public enum ENEMY_DEFENCES {
        GUNS(1),  // 敌方防御方式为枪支
        MISSILES(2),  // 敌方防御方式为导弹
        BOTH(3);  // 敌方防御方式为枪支和导弹

        private final int value;

        // 枚举类型构造函数，初始化枚举值
        ENEMY_DEFENCES(int value) {
            this.value = value;
        }

        // 获取枚举值对应的整数值
        public int getValue() {
            return value;
        }
    }

    // 定义飞机类型的枚举
    public enum AIRCRAFT {
        LIBERATOR(1),  // 飞机类型为解放者
        B29(2),  // 飞机类型为B29
        B17(3),  // 飞机类型为B17
        LANCASTER(4);  // 飞机类型为兰开斯特

        private final int value;

        // 枚举类型构造函数，初始化枚举值
        AIRCRAFT(int value) {
            this.value = value;
        }

        // 获取枚举值对应的整数值
        public int getValue() {
            return value;
        }
    }

    // 用于键盘输入的扫描器
    private final Scanner kbScanner;

    // 当前游戏状态
    private GAME_STATE gameState;

    private SIDE side;  // 玩家所在阵营

    private int missions;  // 任务数量

    private int chanceToBeHit;  // 被击中的几率
    private int percentageHitRateOfGunners;  // 炮手的命中率百分比
    private boolean liar;  // 是否说谎

    // 游戏初始化函数
    public BombsAway() {
        // 设置游戏状态为开始
        gameState = GAME_STATE.START;

        // 初始化键盘扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * 主游戏循环
     *
     */
    }

    /**
     * 显示简要介绍
     */
    public void intro() {
        System.out.println("YOU ARE A PILOT IN A WORLD WAR II BOMBER.");
    }

    /**
     * 确定玩家所在阵营
     * @param message 在键盘输入前显示的消息
     * @return 玩家选择的阵营枚举
     */
    private SIDE getSide(String message) {
        int valueEntered = getNumberFromKeyboard(message);
        for(SIDE side : SIDE.values()) {
            if(side.getValue() == valueEntered) {
                return side;
            }
        }

        // 输入超出范围
        return null;
    }

    /**
     * 确定玩家的目标
     * @param message 在键盘输入前显示的消息
     * @return 玩家选择的目标枚举
     */
    // 根据用户输入的消息获取目标值
    private TARGET getTarget(String message) {
        // 从键盘获取输入的数字
        int valueEntered = getNumberFromKeyboard(message);

        // 遍历目标枚举，查找与输入值匹配的目标
        for(TARGET target : TARGET.values()) {
            if(target.getValue() == valueEntered) {
                return target;
            }
        }

        // 输入超出范围
        return null;
    }

    /**
     * 确定玩家将要驾驶的飞机。
     * @param message 在键盘输入之前显示的消息
     * @return 玩家选择的飞机枚举
     */
    private AIRCRAFT getPlane(String message) {
        // 从键盘获取输入的数字
        int valueEntered = getNumberFromKeyboard(message);

        // 遍历飞机枚举，查找与输入值匹配的飞机
        for(AIRCRAFT plane : AIRCRAFT.values()) {
            if(plane.getValue() == valueEntered) {
                return plane;
            }
        }

        // 输入超出范围
        return null;

    }

    /**
     * 选择敌人的防御类型。
     *
     * @param message 在键盘输入之前显示的消息
     * @return 玩家选择的敌人防御类型枚举
     */
    private ENEMY_DEFENCES getEnemyDefences(String message) {
        // 从键盘获取输入的数字
        int valueEntered = getNumberFromKeyboard(message);
        // 遍历敌人防御类型枚举，查找与输入值匹配的类型
        for (ENEMY_DEFENCES enemyDefences : ENEMY_DEFENCES.values()) {
            if(enemyDefences.getValue() == valueEntered) {
                return enemyDefences;
            }
        }

        // 输入超出范围
        return null;
    }

    // 根据选择的目标输出特定消息
    // 根据目标类型显示相应的消息
    private void displayTargetMessage(TARGET target) {

        // 根据不同的目标类型进行不同的消息显示
        switch (target) {

            case ALBANIA:
                System.out.println("SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE.");
                break;
            case GREECE:
                System.out.println("BE CAREFUL!!!");
                break;
            case NORTH_AFRICA:
                System.out.println("YOU'RE GOING FOR THE OIL, EH?");
                break;
            case RUSSIA:
                System.out.println("YOU'RE NEARING STALINGRAD.");
                break;
            case ENGLAND:
                System.out.println("NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR.");
                break;
            case FRANCE:
                System.out.println("NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS.");
                break;
        }
    }

    /**
     * 从键盘接受一个字符串，并转换为整数
     *
     * @param message 在键盘输入前在屏幕上显示的文本
     *
     * @return 玩家输入的数字
     */
    private int getNumberFromKeyboard(String message) {

        // 显示文本并获取键盘输入
        String answer = displayTextAndGetInput(message);
        // 将输入的字符串转换为整数并返回
        return Integer.parseInt(answer);
    }

    /**
     * 检查玩家是否输入了 Y 或 YES
     *
     * @param text  来自键盘的玩家字符串
     * @return 如果输入了 Y 或 YES 则返回 true，否则返回 false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * 检查字符串是否等于一系列变量值中的一个
     * 用于检查是否输入了 Y 或 YES 等情况
     * 比较是不区分大小写的
     *
     * @param text 源字符串
     * @param values 要与源字符串进行比较的一系列值
     * @return 如果在传递的一系列字符串中找到了匹配则返回 true
     */
    # 检查字符串是否与给定的值中的任何一个相等
    private boolean stringIsAnyValue(String text, String... values) {

        # 遍历可变数量的值，并测试每个值
        for(String val:values) {
            if(text.equalsIgnoreCase(val)) {
                return true;
            }
        }

        # 没有匹配的值
        return false;
    }

    /*
     * 在屏幕上打印消息，然后接受键盘输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 生成随机数
     * 用作计算机玩家的单个数字
     *
     * @return 随机数
     */
    private double randomNumber(int range) {
        return (Math.random()
                * (range));
    }
# 闭合前面的函数定义
```