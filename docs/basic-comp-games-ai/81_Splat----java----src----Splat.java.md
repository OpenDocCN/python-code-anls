# `81_Splat\java\src\Splat.java`

```
import java.util.*;  // 导入 Java 的工具包

/**
 * SPLAT 模拟了一个跳伞跳跃，你要在最后一刻打开降落伞，而不会摔得稀烂！
 * 你可以选择自己的终端速度，或者让计算机为你选择。你也可以选择重力加速度，或者再次让计算机选择，这样你可能会落在八个行星（直到海王星）、月球或太阳上。
 * 计算机然后告诉你你跳跃的高度，并询问自由落体的时间。然后它将你的自由落体时间分成八个间隔，并在下降过程中向你报告进展情况。
 * 计算机还会跟踪数组 A 中的所有先前跳跃，并告诉你与先前成功跳跃的比较情况。如果你想要回忆以前运行的信息，那么你应该在每次运行之前将数组 A 存储在磁盘或文件中，并在每次运行之前读取它。
 * John Yegge 在 Oak Ridge Associated Universities 时创建了这个程序。
 * 由 jason plumb (@breedx2) 从 BASIC 移植而来
 */
public class Splat {
    private static final Random random = new Random(); // 创建一个静态的随机数生成器对象
    private final Scanner scanner = new Scanner(System.in); // 创建一个用于从控制台输入的扫描器对象
    private final List<Float> pastSuccessfulJumpDistances = new ArrayList<>(); // 创建一个存储成功跳跃距离的列表

    public static void main(String[] args) {
        new Splat().run(); // 创建一个 Splat 对象并调用其 run 方法
    }

    public void run() {
        showIntroduction(); // 调用显示游戏介绍的方法

        while (true) { // 进入无限循环

            InitialJumpConditions initial = buildInitialConditions(); // 调用方法构建初始跳跃条件对象

            System.out.println(); // 打印空行
            System.out.printf("    ALTITUDE         = %d FT\n", initial.getAltitude()); // 打印初始高度
            System.out.printf("    TERM. VELOCITY   = %.2f FT/SEC +/-5%%\n", initial.getOriginalTerminalVelocity()); // 打印初始终端速度
            System.out.printf("    ACCELERATION     = %.2f FT/SEC/SEC +/-5%%\n", initial.getOriginalAcceleration()); // 打印初始加速度
            System.out.println("SET THE TIMER FOR YOUR FREEFALL."); // 输出提示信息，要求设置自由落体的时间
            float freefallTime = promptFloat("HOW MANY SECONDS "); // 调用 promptFloat 方法，获取用户输入的自由落体时间
            System.out.println("HERE WE GO.\n"); // 输出提示信息，表示即将开始自由落体
            System.out.println("TIME (SEC)  DIST TO FALL (FT)"); // 输出表头，显示时间和下落距离
            System.out.println("==========  ================="); // 输出分隔线

            JumpResult jump = executeJump(initial, freefallTime); // 调用 executeJump 方法，执行自由落体计算
            showJumpResults(initial, jump); // 调用 showJumpResults 方法，展示自由落体结果

            if (!playAgain()) { // 判断是否要再次进行自由落体
                System.out.println("SSSSSSSSSS."); // 输出提示信息，表示结束
                return; // 结束方法
            }
        }
    }

    private void showIntroduction() { // 定义方法，用于展示介绍信息
        System.out.printf("%33s%s\n", " ", "SPLAT"); // 输出介绍信息
        System.out.printf("%15s%s\n", " ", "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"); // 输出介绍信息
        System.out.print("\n\n\n"); // 输出空行
        System.out.println("WELCOME TO 'SPLAT' -- THE GAME THAT SIMULATES A PARACHUTE");
        # 打印欢迎消息
        System.out.println("JUMP.  TRY TO OPEN YOUR CHUTE AT THE LAST POSSIBLE");
        # 打印提示信息
        System.out.println("MOMENT WITHOUT GOING SPLAT.");
        # 打印提示信息
    }

    private InitialJumpConditions buildInitialConditions() {
        System.out.print("\n\n");
        # 打印换行
        float terminalVelocity = promptTerminalVelocity();
        # 调用promptTerminalVelocity()方法获取终端速度
        float acceleration = promptGravitationalAcceleration();
        # 调用promptGravitationalAcceleration()方法获取重力加速度
        return InitialJumpConditions.create(terminalVelocity, acceleration);
        # 返回InitialJumpConditions对象
    }

    private float promptTerminalVelocity() {
        if (askYesNo("SELECT YOUR OWN TERMINAL VELOCITY")) {
            # 如果用户选择自定义终端速度
            float terminalVelocity = promptFloat("WHAT TERMINAL VELOCITY (MI/HR) ");
            # 调用promptFloat()方法获取用户输入的终端速度
            return mphToFeetPerSec(terminalVelocity);
            # 返回转换后的速度
        }
        float terminalVelocity = (int) (1000 * random.nextFloat());
        # 生成随机的终端速度
        System.out.printf("OK.  TERMINAL VELOCITY = %.2f MI/HR\n", terminalVelocity);
        # 打印终端速度
        return mphToFeetPerSec(terminalVelocity);
        # 返回转换后的速度
    }

    # 定义一个私有方法，用于提示用户输入一个浮点数
    private float promptFloat(String prompt){
        # 循环直到用户输入一个有效的浮点数
        while(true){
            # 打印提示信息
            System.out.print(prompt);
            try {
                # 尝试读取用户输入的浮点数并返回
                return scanner.nextFloat();
            } catch (Exception e) {
                scanner.next(); # 清除当前输入
            }
        }
    }

    # 定义一个私有方法，用于提示用户输入重力加速度
    private float promptGravitationalAcceleration() {
        # 如果用户选择了选择重力加速度
        if (askYesNo("WANT TO SELECT ACCELERATION DUE TO GRAVITY")) {
            # 提示用户输入重力加速度
            return promptFloat("WHAT ACCELERATION (FT/SEC/SEC) ");
        }
        # 否则随机选择一个重力加速度
        return chooseRandomAcceleration();
    }
    # 执行跳跃操作，根据初始条件和降落伞开启时间计算跳跃结果
    def executeJump(initial, chuteOpenTime):
        # 创建一个跳跃结果对象，初始高度为初始条件中的高度
        jump = JumpResult(initial.getAltitude())
        # 循环计算跳跃过程中的各个时间点的情况
        for time in range(0.0, chuteOpenTime, chuteOpenTime / 8):
            # 如果还未达到终端速度，并且时间超过了达到终端加速度的时间
            if not jump.hasReachedTerminalVelocity() and time > initial.getTimeOfTerminalAccelerationReached():
                # 设置已达到终端速度的标志
                jump.setReachedTerminalVelocity()
                # 打印信息，表示已经达到终端速度
                print("TERMINAL VELOCITY REACHED AT T PLUS %f SECONDS.\n" % initial.getTimeOfTerminalAccelerationReached())
            # 计算当前时间点的距离
            newDistance = computeDistance(initial, time, jump.hasReachedTerminalVelocity())
            # 设置距离
            jump.setDistance(newDistance)

            # 如果已经坠毁，返回跳跃结果
            if jump.isSplat():
                return jump
            # 打印当前时间点和距离
            print("%10.2f  %f\n" % (time, jump.getDistance()))
        # 返回跳跃结果
        return jump

    # 计算距离
    def computeDistance(initial, i, hasReachedTerminalVelocity):
        # 获取终端速度
        V = initial.getTerminalVelocity()
        final float A = initial.getAcceleration();  # 获取初始加速度
        if (hasReachedTerminalVelocity) {  # 如果已经达到了终端速度
            return initial.getAltitude() - ((V * V / (2 * A)) + (V * (i - (V / A))));  # 返回计算后的高度
        }
        return initial.getAltitude() - ((A / 2) * i * i);  # 返回计算后的高度
    }

    private void showJumpResults(InitialJumpConditions initial, JumpResult jump) {  # 显示跳跃结果
        if (jump.isSplat()) {  # 如果跳跃失败
            showSplatMessage(initial, jump);  # 显示跳跃失败信息
            showCleverSplatMessage();  # 显示聪明的跳跃失败信息
            return;  # 返回
        }
        System.out.println("CHUTE OPEN");  # 打印信息
        int worseJumpCount = countWorseHistoricalJumps(jump);  # 计算历史跳跃次数
        int successfulJumpCt = pastSuccessfulJumpDistances.size();  # 获取成功跳跃的次数
        pastSuccessfulJumpDistances.add(jump.getDistance());  # 添加成功跳跃的距离

        if (pastSuccessfulJumpDistances.size() <= 2) {  # 如果成功跳跃的次数小于等于2
            List<String> ordinals = Arrays.asList("1ST", "2ND", "3RD");  # 创建顺序列表
            System.out.printf("AMAZING!!! NOT BAD FOR YOUR %s SUCCESSFUL JUMP!!!\n", ordinals.get(successfulJumpCt));
            return;  # 结束函数的执行

        }

        int betterThanCount = successfulJumpCt - worseJumpCount;  # 计算成功跳跃次数减去失败跳跃次数的差值
        if (betterThanCount <= 0.1 * successfulJumpCt) {  # 如果差值小于等于成功跳跃次数的10%
            System.out.printf("WOW!  THAT'S SOME JUMPING.  OF THE %d SUCCESSFUL JUMPS\n", successfulJumpCt);  # 打印成功跳跃次数
            System.out.printf("BEFORE YOURS, ONLY %d OPENED THEIR CHUTES LOWER THAN\n", betterThanCount);  # 打印比你跳得低的人数
            System.out.println("YOU DID.");  # 打印提示信息
        } else if (betterThanCount <= 0.25 * successfulJumpCt) {  # 如果差值小于等于成功跳跃次数的25%
            System.out.printf("PRETTY GOOD!  %d SUCCESSFUL JUMPS PRECEDED YOURS AND ONLY\n", successfulJumpCt);  # 打印成功跳跃次数
            System.out.printf("%d OF THEM GOT LOWER THAN YOU DID BEFORE THEIR CHUTES\n", betterThanCount);  # 打印比你跳得低的人数
            System.out.println("OPENED.");  # 打印提示信息
        } else if (betterThanCount <= 0.5 * successfulJumpCt) {  # 如果差值小于等于成功跳跃次数的50%
            System.out.printf("NOT BAD.  THERE HAVE BEEN %d SUCCESSFUL JUMPS BEFORE YOURS.\n", successfulJumpCt);  # 打印成功跳跃次数
            System.out.printf("YOU WERE BEATEN OUT BY %d OF THEM.\n", betterThanCount);  # 打印比你跳得低的人数
        } else if (betterThanCount <= 0.75 * successfulJumpCt) {  # 如果差值小于等于成功跳跃次数的75%
            System.out.printf("CONSERVATIVE, AREN'T YOU?  YOU RANKED ONLY %d IN THE\n", betterThanCount);  # 打印比你跳得低的人数
            System.out.printf("%d SUCCESSFUL JUMPS BEFORE YOURS.\n", successfulJumpCt);  # 打印成功跳跃次数
        } else if (betterThanCount <= -0.9 * successfulJumpCt) {  # 如果差值小于等于成功跳跃次数的-90%
            System.out.println("HUMPH!  DON'T YOU HAVE ANY SPORTING BLOOD?  THERE WERE");
            // 打印文本信息
            System.out.printf("%d SUCCESSFUL JUMPS BEFORE YOURS AND YOU CAME IN %d JUMPS\n", successfulJumpCt, worseJumpCount);
            // 格式化打印文本信息，包括成功跳跃次数和比你差的跳跃次数
            System.out.println("BETTER THAN THE WORST.  SHAPE UP!!!\n");
            // 打印文本信息
        } else {
            // 如果条件不满足
            System.out.printf("HEY!  YOU PULLED THE RIP CORD MUCH TOO SOON.  %d SUCCESSFUL\n", successfulJumpCt);
            // 格式化打印文本信息，包括成功跳跃次数
            System.out.printf("JUMPS BEFORE YOURS AND YOU CAME IN NUMBER %d!  GET WITH IT!\n", betterThanCount);
            // 格式化打印文本信息，包括比你好的跳跃次数
        }
    }

    private void showSplatMessage(InitialJumpConditions initial, JumpResult jump) {
        // 定义一个方法，用于展示失败信息
        double timeOfSplat = computeTimeOfSplat(initial, jump);
        // 计算跳伞失败的时间
        System.out.printf("%10.2f  SPLAT\n", timeOfSplat);
        // 格式化打印跳伞失败的时间
    }

    /**
     * Returns the number of jumps for which this jump was better
     */
    private double computeTimeOfSplat(InitialJumpConditions initial, JumpResult jump) {
        // 定义一个方法，用于计算跳伞失败的时间
        final float V = initial.getTerminalVelocity();
        // 获取初始跳伞条件的终端速度
        final float A = initial.getAcceleration();
        // 获取初始跳伞条件的加速度
        if (jump.hasReachedTerminalVelocity()) {  # 如果跳跃达到了终端速度
            return (V / A) + ((initial.getAltitude() - (V * V / (2 * A))) / V);  # 返回计算后的值
        }
        return Math.sqrt(2 * initial.getAltitude() / A);  # 否则返回计算后的值

    }

    private int countWorseHistoricalJumps(JumpResult jump) {  # 计算历史跳跃中比当前跳跃更糟糕的次数
        return (int) pastSuccessfulJumpDistances.stream()  # 将过去成功的跳跃距离转换为流
                .filter(distance -> jump.getDistance() < distance)  # 过滤出距离比当前跳跃距离小的
                .count();  # 计算数量
    }

    private void showCleverSplatMessage() {  # 显示聪明的失败信息
        List<String> messages = Arrays.asList(  # 创建包含多个信息的列表
                "REQUIESCAT IN PACE.",  # 安息
                "MAY THE ANGEL OF HEAVEN LEAD YOU INTO PARADISE.",  # 愿天堂的天使引领你进入天堂
                "REST IN PEACE.",  # 安息
                "SON-OF-A-GUN.",  # 该死
                "#$%&&%!$",  # 一些无意义的字符
                "A KICK IN THE PANTS IS A BOOST IF YOU'RE HEADED RIGHT."  # 如果你朝着正确的方向，踢屁股是一种提升
    # 如果用户输入的速度大于等于 0，则返回 true，否则返回 false
    private boolean isSpeedValid(float speed) {
        return speed >= 0;
    }

    # 如果用户输入的时间大于等于 0，则返回 true，否则返回 false
    private boolean isTimeValid(int time) {
        return time >= 0;
    }

    # 计算距离
    private float calculateDistance(float speed, int time) {
        # 如果速度和时间都有效，则返回速度乘以时间的结果，否则返回 -1
        if (isSpeedValid(speed) && isTimeValid(time)) {
            return speed * time;
        } else {
            return -1;
        }
    }

    # 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
    def read_zip(fname):
        # 根据 ZIP 文件名读取其二进制，封装成字节流
        bio = BytesIO(open(fname, 'rb').read())
        # 使用字节流里面内容创建 ZIP 对象
        zip = zipfile.ZipFile(bio, 'r')
        # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
        fdict = {n:zip.read(n) for n in zip.namelist()}
        # 关闭 ZIP 对象
        zip.close()
        # 返回结果字典
        return fdict
        System.out.printf("%s (YES OR NO) ", prompt);  // 打印提示信息，要求用户输入“YES”或“NO”
        while (true) {  // 进入无限循环
            String answer = scanner.next();  // 从用户输入中获取字符串
            switch (answer) {  // 根据用户输入的字符串进行判断
                case "YES":  // 如果用户输入为“YES”
                    return true;  // 返回 true
                case "NO":  // 如果用户输入为“NO”
                    return false;  // 返回 false
                default:  // 如果用户输入既不是“YES”也不是“NO”
                    System.out.print("YES OR NO ");  // 打印提示信息，要求用户重新输入“YES”或“NO”
            }
        }
    }

    private float chooseRandomAcceleration() {
        Planet planet = Planet.pickRandom();  // 从 Planet 类中随机选择一个行星
        System.out.printf("%s %s. ACCELERATION=%.2f FT/SEC/SEC.\n", planet.getMessage(), planet.name(), planet.getAcceleration());  // 打印行星的信息和加速度
        return planet.getAcceleration();  // 返回行星的加速度
    }
    enum Planet {  # 定义一个枚举类型 Planet
        MERCURY("FINE. YOU'RE ON", 12.2f),  # 枚举值 MERCURY，包含消息和加速度
        VENUS("ALL RIGHT. YOU'RE ON", 28.3f),  # 枚举值 VENUS，包含消息和加速度
        EARTH("THEN YOU'RE ON", 32.16f),  # 枚举值 EARTH，包含消息和加速度
        MOON("FINE. YOU'RE ON THE", 5.15f),  # 枚举值 MOON，包含消息和加速度
        MARS("ALL RIGHT. YOU'RE ON", 12.5f),  # 枚举值 MARS，包含消息和加速度
        JUPITER("THEN YOU'RE ON", 85.2f),  # 枚举值 JUPITER，包含消息和加速度
        SATURN("FINE. YOU'RE ON", 37.6f),  # 枚举值 SATURN，包含消息和加速度
        URANUS("ALL RIGHT. YOU'RE ON", 33.8f),  # 枚举值 URANUS，包含消息和加速度
        NEPTUNE("THEN YOU'RE ON", 39.6f),  # 枚举值 NEPTUNE，包含消息和加速度
        SUN("FINE. YOU'RE ON THE", 896.0f);  # 枚举值 SUN，包含消息和加速度

        private static final Random random = new Random();  # 创建一个静态的随机数生成器
        private final String message;  # 声明一个私有的字符串变量 message
        private final float acceleration;  # 声明一个私有的浮点数变量 acceleration

        Planet(String message, float acceleration) {  # 枚举类型 Planet 的构造函数，接受消息和加速度参数
            this.message = message;  # 初始化消息变量
            this.acceleration = acceleration;  # 初始化加速度变量
        }
        static Planet pickRandom() {  // 静态方法，用于随机选择一个行星
            return values()[random.nextInt(Planet.values().length)];  // 返回一个随机选择的行星
        }

        String getMessage() {  // 返回消息的方法
            return message;  // 返回消息
        }

        float getAcceleration() {  // 返回加速度的方法
            return acceleration;  // 返回加速度
        }
    }

    // Mutable
    static class JumpResult {  // 可变类 JumpResult
        private boolean reachedTerminalVelocity = false;  // 到达终端速度的布尔值，默认为false
        private float distance; // from the ground  // 距离地面的浮点数

        public JumpResult(float distance) {  // JumpResult 类的构造函数，接受一个距离参数
        // 设置距离
        void setDistance(float distance) {
            this.distance = distance;
        }

        // 判断是否已经落地
        boolean isSplat() {
            return distance <= 0;
        }

        // 判断是否已经达到最终速度
        boolean hasReachedTerminalVelocity() {
            return reachedTerminalVelocity;
        }

        // 获取距离
        float getDistance() {
            return distance;
        }

        // 设置距离
        void setDistance(float distance) {
            this.distance = distance;
        }

        // 设置已达到最终速度
        void setReachedTerminalVelocity() {
            reachedTerminalVelocity = true;  // 设置 reachedTerminalVelocity 变量为 true，表示达到了终端速度
        }
    }

    // Immutable  // 不可变类
    static class InitialJumpConditions {  // 初始跳跃条件类
        private final float originalTerminalVelocity;  // 初始终端速度
        private final float originalAcceleration;  // 初始加速度
        private final float terminalVelocity;  // 终端速度
        private final float acceleration;  // 加速度
        private final int altitude;  // 海拔高度

        private InitialJumpConditions(float originalTerminalVelocity, float originalAcceleration,
                                      float terminalVelocity, float acceleration, int altitude) {  // 初始跳跃条件构造函数
            this.originalTerminalVelocity = originalTerminalVelocity;  // 设置初始终端速度
            this.originalAcceleration = originalAcceleration;  // 设置初始加速度
            this.terminalVelocity = terminalVelocity;  // 设置终端速度
            this.acceleration = acceleration;  // 设置加速度
            this.altitude = altitude;  // 设置海拔高度
        }
        // 创建具有调整速度/加速度和随机初始高度的初始跳跃条件
        private static InitialJumpConditions create(float terminalVelocity, float gravitationalAcceleration) {
            // 生成随机初始高度
            final int altitude = (int) (9001.0f * random.nextFloat() + 1000);
            // 返回初始跳跃条件对象
            return new InitialJumpConditions(terminalVelocity, gravitationalAcceleration,
                    plusMinus5Percent(terminalVelocity), plusMinus5Percent(gravitationalAcceleration), altitude);
        }

        // 返回值增加或减少5%的浮点数
        private static float plusMinus5Percent(float value) {
            return value + ((value * random.nextFloat()) / 20.0f) - ((value * random.nextFloat()) / 20.0f);
        }

        // 获取原始的终端速度
        float getOriginalTerminalVelocity() {
            return originalTerminalVelocity;
        }

        // 获取原始的加速度
        float getOriginalAcceleration() {
            return originalAcceleration;
        }
# 返回终端速度
float getTerminalVelocity() {
    return terminalVelocity;
}

# 返回加速度
float getAcceleration() {
    return acceleration;
}

# 返回海拔高度
int getAltitude() {
    return altitude;
}

# 返回达到终端加速度所需的时间
float getTimeOfTerminalAccelerationReached() {
    return terminalVelocity / acceleration;
}
```