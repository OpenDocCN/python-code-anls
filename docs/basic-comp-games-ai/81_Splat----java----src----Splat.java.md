# `basic-computer-games\81_Splat\java\src\Splat.java`

```
# 导入 java.util 包
import java.util.*;

/**
 * SPLAT 模拟了一个跳伞跳跃，你需要在最后可能的时刻打开降落伞，而不会摔成一团！
 * 你可以选择自己的终端速度，或者让计算机为你选择。你也可以选择重力加速度，或者再次让计算机选择，这样你可能会落在八个行星（直到海王星）、月球或太阳上。
 * <p>
 * 计算机然后告诉你你跳跃的高度，并询问自由落体的时间。然后将你的自由落体时间分成八个间隔，并在下降过程中给你进度报告。计算机还会跟踪数组 A 中的所有先前跳跃，并告诉你与先前成功跳跃的比较情况。如果你想要回忆以前运行的信息，那么你应该在每次运行之前将数组 A 存储在磁盘或取文件中并读取它。
 * <p>
 * John Yegge 在 Oak Ridge Associated Universities 时创建了这个程序。
 * <p>
 * 由 jason plumb (@breedx2) 从 BASIC 移植
 * </p>
 */
public class Splat {
    private static final Random random = new Random();  # 创建一个静态的随机数生成器对象
    private final Scanner scanner = new Scanner(System.in);  # 创建一个用于从控制台读取输入的 Scanner 对象
    private final List<Float> pastSuccessfulJumpDistances = new ArrayList<>();  # 创建一个用于存储先前成功跳跃距离的列表

    public static void main(String[] args) {  # 主函数
        new Splat().run();  # 创建 Splat 对象并调用其 run 方法
    }
}
    // 运行游戏
    public void run() {
        // 显示游戏介绍
        showIntroduction();

        // 循环进行游戏
        while (true) {
            // 构建初始条件
            InitialJumpConditions initial = buildInitialConditions();

            // 打印初始条件信息
            System.out.println();
            System.out.printf("    ALTITUDE         = %d FT\n", initial.getAltitude());
            System.out.printf("    TERM. VELOCITY   = %.2f FT/SEC +/-5%%\n", initial.getOriginalTerminalVelocity());
            System.out.printf("    ACCELERATION     = %.2f FT/SEC/SEC +/-5%%\n", initial.getOriginalAcceleration());

            // 提示设置自由落体时间
            System.out.println("SET THE TIMER FOR YOUR FREEFALL.");
            float freefallTime = promptFloat("HOW MANY SECONDS ");
            System.out.println("HERE WE GO.\n");
            System.out.println("TIME (SEC)  DIST TO FALL (FT)");
            System.out.println("==========  =================");

            // 执行跳跃并获取结果
            JumpResult jump = executeJump(initial, freefallTime);
            // 显示跳跃结果
            showJumpResults(initial, jump);

            // 询问是否再玩一次
            if (!playAgain()) {
                System.out.println("SSSSSSSSSS.");
                return;
            }
        }
    }

    // 显示游戏介绍
    private void showIntroduction() {
        System.out.printf("%33s%s\n", " ", "SPLAT");
        System.out.printf("%15s%s\n", " ", "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.print("\n\n\n");
        System.out.println("WELCOME TO 'SPLAT' -- THE GAME THAT SIMULATES A PARACHUTE");
        System.out.println("JUMP.  TRY TO OPEN YOUR CHUTE AT THE LAST POSSIBLE");
        System.out.println("MOMENT WITHOUT GOING SPLAT.");
    }

    // 构建初始条件
    private InitialJumpConditions buildInitialConditions() {
        System.out.print("\n\n");
        // 提示输入终端速度
        float terminalVelocity = promptTerminalVelocity();
        // 提示输入重力加速度
        float acceleration = promptGravitationalAcceleration();
        // 创建初始跳跃条件对象
        return InitialJumpConditions.create(terminalVelocity, acceleration);
    }
    // 提示用户输入终端速度，并返回转换后的速度值
    private float promptTerminalVelocity() {
        // 如果用户选择自定义终端速度
        if (askYesNo("SELECT YOUR OWN TERMINAL VELOCITY")) {
            // 提示用户输入终端速度，并返回转换后的速度值
            float terminalVelocity = promptFloat("WHAT TERMINAL VELOCITY (MI/HR) ");
            return mphToFeetPerSec(terminalVelocity);
        }
        // 如果用户不选择自定义终端速度，则随机生成终端速度
        float terminalVelocity = (int) (1000 * random.nextFloat());
        System.out.printf("OK.  TERMINAL VELOCITY = %.2f MI/HR\n", terminalVelocity);
        return mphToFeetPerSec(terminalVelocity);
    }

    // 提示用户输入浮点数，并返回输入的值
    private float promptFloat(String prompt){
        while(true){
            System.out.print(prompt);
            try {
                return scanner.nextFloat();
            } catch (Exception e) {
                scanner.next(); // 清除当前输入
            }
        }
    }

    // 提示用户输入重力加速度，并返回输入的值
    private float promptGravitationalAcceleration() {
        // 如果用户选择自定义重力加速度
        if (askYesNo("WANT TO SELECT ACCELERATION DUE TO GRAVITY")) {
            // 提示用户输入重力加速度，并返回输入的值
            return promptFloat("WHAT ACCELERATION (FT/SEC/SEC) ");
        }
        // 如果用户不选择自定义重力加速度，则随机选择一个重力加速度
        return chooseRandomAcceleration();
    }

    // 执行跳跃操作，计算跳跃结果
    private JumpResult executeJump(InitialJumpConditions initial, float chuteOpenTime) {
        // 创建跳跃结果对象
        JumpResult jump = new JumpResult(initial.getAltitude());
        // 循环计算跳跃过程中的各个时间点的距离
        for (float time = 0.0f; time < chuteOpenTime; time += chuteOpenTime / 8) {
            // 如果尚未达到终端速度，并且时间大于终端加速度达到的时间
            if (!jump.hasReachedTerminalVelocity() && time > initial.getTimeOfTerminalAccelerationReached()) {
                jump.setReachedTerminalVelocity();
                System.out.printf("TERMINAL VELOCITY REACHED AT T PLUS %f SECONDS.\n", initial.getTimeOfTerminalAccelerationReached());
            }
            // 计算新的距离
            float newDistance = computeDistance(initial, time, jump.hasReachedTerminalVelocity());
            jump.setDistance(newDistance);

            // 如果已经发生坠落
            if (jump.isSplat()) {
                return jump;
            }
            System.out.printf("%10.2f  %f\n", time, jump.getDistance());
        }
        return jump;
    }
    // 计算距离
    private float computeDistance(InitialJumpConditions initial, float i, boolean hasReachedTerminalVelocity) {
        // 获取终端速度
        final float V = initial.getTerminalVelocity();
        // 获取加速度
        final float A = initial.getAcceleration();
        // 如果已达到终端速度
        if (hasReachedTerminalVelocity) {
            // 返回距离
            return initial.getAltitude() - ((V * V / (2 * A)) + (V * (i - (V / A))));
        }
        // 返回距离
        return initial.getAltitude() - ((A / 2) * i * i);
    }

    // 显示坠落信息
    private void showSplatMessage(InitialJumpConditions initial, JumpResult jump) {
        // 计算坠落时间
        double timeOfSplat = computeTimeOfSplat(initial, jump);
        // 打印坠落时间
        System.out.printf("%10.2f  SPLAT\n", timeOfSplat);
    }

    /**
     * 计算坠落时间
     * 返回此跳跃比较好的跳跃次数
     */
    private double computeTimeOfSplat(InitialJumpConditions initial, JumpResult jump) {
        // 获取终端速度
        final float V = initial.getTerminalVelocity();
        // 获取加速度
        final float A = initial.getAcceleration();
        // 如果已达到终端速度
        if (jump.hasReachedTerminalVelocity()) {
            // 返回坠落时间
            return (V / A) + ((initial.getAltitude() - (V * V / (2 * A))) / V);
        }
        // 返回坠落时间
        return Math.sqrt(2 * initial.getAltitude() / A);
    }

    // 计算历史跳跃次数
    private int countWorseHistoricalJumps(JumpResult jump) {
        // 返回跳跃次数
        return (int) pastSuccessfulJumpDistances.stream()
                .filter(distance -> jump.getDistance() < distance)
                .count();
    }

    // 显示聪明的坠落信息
    private void showCleverSplatMessage() {
        // 创建信息列表
        List<String> messages = Arrays.asList(
                "REQUIESCAT IN PACE.",
                "MAY THE ANGEL OF HEAVEN LEAD YOU INTO PARADISE.",
                "REST IN PEACE.",
                "SON-OF-A-GUN.",
                "#$%&&%!$",
                "A KICK IN THE PANTS IS A BOOST IF YOU'RE HEADED RIGHT.",
                "HMMM. SHOULD HAVE PICKED A SHORTER TIME.",
                "MUTTER. MUTTER. MUTTER.",
                "PUSHING UP DAISIES.",
                "EASY COME, EASY GO."
        );
        // 打印随机信息
        System.out.println(messages.get(random.nextInt(10)));
    }
    // 询问用户是否想再玩一次游戏，根据用户输入返回布尔值
    private boolean playAgain() {
        // 如果用户输入是肯定的，则返回 true
        if (askYesNo("DO YOU WANT TO PLAY AGAIN ")) {
            return true;
        }
        // 如果用户输入是否定的，则再次询问用户，并返回结果
        return askYesNo("PLEASE");
    }

    // 将英里/小时转换为每秒英尺，根据给定的速度返回转换后的值
    private float mphToFeetPerSec(float speed) {
        return speed * (5280.0f / 3600.0f);
    }

    // 询问用户问题，并根据用户输入返回布尔值
    private boolean askYesNo(String prompt) {
        // 打印提示信息，等待用户输入
        System.out.printf("%s (YES OR NO) ", prompt);
        // 循环直到用户输入有效的答案
        while (true) {
            String answer = scanner.next();
            // 根据用户输入返回相应的布尔值
            switch (answer) {
                case "YES":
                    return true;
                case "NO":
                    return false;
                default:
                    System.out.print("YES OR NO ");
            }
        }
    }

    // 选择一个随机的行星，并返回其加速度
    private float chooseRandomAcceleration() {
        // 从 Planet 枚举中随机选择一个行星
        Planet planet = Planet.pickRandom();
        // 打印行星信息和加速度，返回加速度值
        System.out.printf("%s %s. ACCELERATION=%.2f FT/SEC/SEC.\n", planet.getMessage(), planet.name(), planet.getAcceleration());
        return planet.getAcceleration();
    }

    // 行星枚举类型
    enum Planet {
        // 枚举值包括行星的信息和加速度
        MERCURY("FINE. YOU'RE ON", 12.2f),
        VENUS("ALL RIGHT. YOU'RE ON", 28.3f),
        EARTH("THEN YOU'RE ON", 32.16f),
        // ... 其他行星的信息和加速度
        SUN("FINE. YOU'RE ON THE", 896.0f);

        // 静态随机对象
        private static final Random random = new Random();
        // 行星的信息和加速度
        private final String message;
        private final float acceleration;

        // 构造函数，初始化行星的信息和加速度
        Planet(String message, float acceleration) {
            this.message = message;
            this.acceleration = acceleration;
        }

        // 随机选择一个行星
        static Planet pickRandom() {
            return values()[random.nextInt(Planet.values().length)];
        }

        // 获取行星的信息
        String getMessage() {
            return message;
        }

        // 获取行星的加速度
        float getAcceleration() {
            return acceleration;
        }
    }

    // 可变的
    # 定义一个内部静态类 JumpResult
    static class JumpResult {
        # 定义私有属性 reachedTerminalVelocity，表示是否达到了终端速度
        private boolean reachedTerminalVelocity = false;
        # 定义属性 distance，表示距离地面的距离
        private float distance; // from the ground

        # 构造函数，初始化距离属性
        public JumpResult(float distance) {
            this.distance = distance;
        }

        # 判断是否已经坠毁
        boolean isSplat() {
            return distance <= 0;
        }

        # 判断是否达到了终端速度
        boolean hasReachedTerminalVelocity() {
            return reachedTerminalVelocity;
        }

        # 获取距离属性的值
        float getDistance() {
            return distance;
        }

        # 设置距离属性的值
        void setDistance(float distance) {
            this.distance = distance;
        }

        # 设置已达到终端速度
        void setReachedTerminalVelocity() {
            reachedTerminalVelocity = true;
        }
    }

    # 不可变的
    # 表示该类是不可变的
    // 定义初始跳跃条件的静态内部类
    static class InitialJumpConditions {
        // 初始终端速度
        private final float originalTerminalVelocity;
        // 初始加速度
        private final float originalAcceleration;
        // 终端速度
        private final float terminalVelocity;
        // 加速度
        private final float acceleration;
        // 初始高度
        private final int altitude;

        // 私有构造函数，用于初始化初始跳跃条件
        private InitialJumpConditions(float originalTerminalVelocity, float originalAcceleration,
                                      float terminalVelocity, float acceleration, int altitude) {
            this.originalTerminalVelocity = originalTerminalVelocity;
            this.originalAcceleration = originalAcceleration;
            this.terminalVelocity = terminalVelocity;
            this.acceleration = acceleration;
            this.altitude = altitude;
        }

        // 创建具有调整速度/加速度和随机初始高度的初始跳跃条件
        private static InitialJumpConditions create(float terminalVelocity, float gravitationalAcceleration) {
            // 随机生成初始高度
            final int altitude = (int) (9001.0f * random.nextFloat() + 1000);
            // 返回新的初始跳跃条件对象
            return new InitialJumpConditions(terminalVelocity, gravitationalAcceleration,
                    plusMinus5Percent(terminalVelocity), plusMinus5Percent(gravitationalAcceleration), altitude);
        }

        // 根据给定值返回加速度的加减5%的值
        private static float plusMinus5Percent(float value) {
            return value + ((value * random.nextFloat()) / 20.0f) - ((value * random.nextFloat()) / 20.0f);
        }

        // 获取初始终端速度
        float getOriginalTerminalVelocity() {
            return originalTerminalVelocity;
        }

        // 获取初始加速度
        float getOriginalAcceleration() {
            return originalAcceleration;
        }

        // 获取终端速度
        float getTerminalVelocity() {
            return terminalVelocity;
        }

        // 获取加速度
        float getAcceleration() {
            return acceleration;
        }

        // 获取初始高度
        int getAltitude() {
            return altitude;
        }

        // 获取达到终端加速度所需的时间
        float getTimeOfTerminalAccelerationReached() {
            return terminalVelocity / acceleration;
        }
    }
# 闭合前面的函数定义
```