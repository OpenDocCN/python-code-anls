# BasicComputerGames源码解析 74

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Java](https://openjdk.java.net/)


# `81_Splat/java/src/Splat.java`

这段代码是一个名为“SPLAT”的程序，它通过模拟降落伞的降落来展示如何模拟现实生活中的问题。这个程序的主要目的是让人们了解如何根据实际需要来选择不同的飞行参数，包括终端速度、重力加速度等。

具体来说，这段代码实现了以下功能：

1. 读取用户输入的值，包括终端速度、自由落体时间、跳跃高度等。
2. 根据用户输入的值，计算出自由落体时间，并将其分成8个区间。
3. 给用户提供前几次跳跃的高度和速度，以及当前跳跃的速度。
4. 如果需要查看前几次跳跃的信息，可以从文件中读取并显示出来。

这段代码的主要目的是提供一个简单的工具，让用户可以了解如何根据需要选择不同的参数，并了解如何计算出实际结果。这个工具可以用于各种实际应用，如航空航天、机器人等。


```
import java.util.*;

/**
 * SPLAT simulates a parachute jump in which you try to open your parachute at the last possible moment without going
 * splat! You may select your own terminal velocity or let the computer do it for you. You many also select the
 * acceleration due to gravity or, again, let the computer do it in which case you might wind up on any of eight
 * planets (out to Neptune), the moon, or the sun.
 * <p>
 * The computer then tells you the height you’re jumping from and asks for the seconds of free fall. It then divides
 * your free fall time into eight intervals and gives you progress reports on your way down. The computer also keeps
 * track of all prior jumps in the array A and lets you know how you compared with previous successful jumps. If you
 * want to recall information from previous runs, then you should store array A in a disk or take file and read it
 * before each run.
 * <p>
 * John Yegge created this program while at the Oak Ridge Associated Universities.
 * <p>
 * Ported from BASIC by jason plumb (@breedx2)
 * </p>
 */
```

This is a Java class called "InitialJumpConditions" that represents the initial conditions for a jump from ground level. The class contains private variables to store the original terminal velocity, original acceleration, terminal velocity, acceleration, and altitude. It also contains private methods to create initial jump conditions with adjusted velocity and acceleration, and a random initial altitude.

The create method takes two parameters: terminal velocity and gravitational acceleration, and returns an instance of the InitialJumpConditions class with those values modified.

The main method in the "InitialJumpConditions" class sets the random seed and generates a random integer within the range of 1 to 100, and uses that random integer to determine the initial altitude for the jump. It then uses the variables in the "InitialJumpConditions" class to create an instance of the class and returns it.

The "plusMinus5Percent" method adds 5% random variation to the given value.


```
public class Splat {
    private static final Random random = new Random();
    private final Scanner scanner = new Scanner(System.in);
    private final List<Float> pastSuccessfulJumpDistances = new ArrayList<>();

    public static void main(String[] args) {
        new Splat().run();
    }

    public void run() {
        showIntroduction();

        while (true) {

            InitialJumpConditions initial = buildInitialConditions();

            System.out.println();
            System.out.printf("    ALTITUDE         = %d FT\n", initial.getAltitude());
            System.out.printf("    TERM. VELOCITY   = %.2f FT/SEC +/-5%%\n", initial.getOriginalTerminalVelocity());
            System.out.printf("    ACCELERATION     = %.2f FT/SEC/SEC +/-5%%\n", initial.getOriginalAcceleration());

            System.out.println("SET THE TIMER FOR YOUR FREEFALL.");
            float freefallTime = promptFloat("HOW MANY SECONDS ");
            System.out.println("HERE WE GO.\n");
            System.out.println("TIME (SEC)  DIST TO FALL (FT)");
            System.out.println("==========  =================");

            JumpResult jump = executeJump(initial, freefallTime);
            showJumpResults(initial, jump);

            if (!playAgain()) {
                System.out.println("SSSSSSSSSS.");
                return;
            }
        }
    }

    private void showIntroduction() {
        System.out.printf("%33s%s\n", " ", "SPLAT");
        System.out.printf("%15s%s\n", " ", "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.print("\n\n\n");
        System.out.println("WELCOME TO 'SPLAT' -- THE GAME THAT SIMULATES A PARACHUTE");
        System.out.println("JUMP.  TRY TO OPEN YOUR CHUTE AT THE LAST POSSIBLE");
        System.out.println("MOMENT WITHOUT GOING SPLAT.");
    }

    private InitialJumpConditions buildInitialConditions() {
        System.out.print("\n\n");
        float terminalVelocity = promptTerminalVelocity();
        float acceleration = promptGravitationalAcceleration();
        return InitialJumpConditions.create(terminalVelocity, acceleration);
    }

    private float promptTerminalVelocity() {
        if (askYesNo("SELECT YOUR OWN TERMINAL VELOCITY")) {
            float terminalVelocity = promptFloat("WHAT TERMINAL VELOCITY (MI/HR) ");
            return mphToFeetPerSec(terminalVelocity);
        }
        float terminalVelocity = (int) (1000 * random.nextFloat());
        System.out.printf("OK.  TERMINAL VELOCITY = %.2f MI/HR\n", terminalVelocity);
        return mphToFeetPerSec(terminalVelocity);
    }

    private float promptFloat(String prompt){
        while(true){
            System.out.print(prompt);
            try {
                return scanner.nextFloat();
            } catch (Exception e) {
                scanner.next(); // clear current input
            }
        }
    }

    private float promptGravitationalAcceleration() {
        if (askYesNo("WANT TO SELECT ACCELERATION DUE TO GRAVITY")) {
            return promptFloat("WHAT ACCELERATION (FT/SEC/SEC) ");
        }
        return chooseRandomAcceleration();
    }

    private JumpResult executeJump(InitialJumpConditions initial, float chuteOpenTime) {
        JumpResult jump = new JumpResult(initial.getAltitude());
        for (float time = 0.0f; time < chuteOpenTime; time += chuteOpenTime / 8) {
            if (!jump.hasReachedTerminalVelocity() && time > initial.getTimeOfTerminalAccelerationReached()) {
                jump.setReachedTerminalVelocity();
                System.out.printf("TERMINAL VELOCITY REACHED AT T PLUS %f SECONDS.\n", initial.getTimeOfTerminalAccelerationReached());
            }
            float newDistance = computeDistance(initial, time, jump.hasReachedTerminalVelocity());
            jump.setDistance(newDistance);

            if (jump.isSplat()) {
                return jump;
            }
            System.out.printf("%10.2f  %f\n", time, jump.getDistance());
        }
        return jump;
    }

    private float computeDistance(InitialJumpConditions initial, float i, boolean hasReachedTerminalVelocity) {
        final float V = initial.getTerminalVelocity();
        final float A = initial.getAcceleration();
        if (hasReachedTerminalVelocity) {
            return initial.getAltitude() - ((V * V / (2 * A)) + (V * (i - (V / A))));
        }
        return initial.getAltitude() - ((A / 2) * i * i);
    }

    private void showJumpResults(InitialJumpConditions initial, JumpResult jump) {
        if (jump.isSplat()) {
            showSplatMessage(initial, jump);
            showCleverSplatMessage();
            return;
        }
        System.out.println("CHUTE OPEN");
        int worseJumpCount = countWorseHistoricalJumps(jump);
        int successfulJumpCt = pastSuccessfulJumpDistances.size();
        pastSuccessfulJumpDistances.add(jump.getDistance());

        if (pastSuccessfulJumpDistances.size() <= 2) {
            List<String> ordinals = Arrays.asList("1ST", "2ND", "3RD");
            System.out.printf("AMAZING!!! NOT BAD FOR YOUR %s SUCCESSFUL JUMP!!!\n", ordinals.get(successfulJumpCt));
            return;
        }

        int betterThanCount = successfulJumpCt - worseJumpCount;
        if (betterThanCount <= 0.1 * successfulJumpCt) {
            System.out.printf("WOW!  THAT'S SOME JUMPING.  OF THE %d SUCCESSFUL JUMPS\n", successfulJumpCt);
            System.out.printf("BEFORE YOURS, ONLY %d OPENED THEIR CHUTES LOWER THAN\n", betterThanCount);
            System.out.println("YOU DID.");
        } else if (betterThanCount <= 0.25 * successfulJumpCt) {
            System.out.printf("PRETTY GOOD!  %d SUCCESSFUL JUMPS PRECEDED YOURS AND ONLY\n", successfulJumpCt);
            System.out.printf("%d OF THEM GOT LOWER THAN YOU DID BEFORE THEIR CHUTES\n", betterThanCount);
            System.out.println("OPENED.");
        } else if (betterThanCount <= 0.5 * successfulJumpCt) {
            System.out.printf("NOT BAD.  THERE HAVE BEEN %d SUCCESSFUL JUMPS BEFORE YOURS.\n", successfulJumpCt);
            System.out.printf("YOU WERE BEATEN OUT BY %d OF THEM.\n", betterThanCount);
        } else if (betterThanCount <= 0.75 * successfulJumpCt) {
            System.out.printf("CONSERVATIVE, AREN'T YOU?  YOU RANKED ONLY %d IN THE\n", betterThanCount);
            System.out.printf("%d SUCCESSFUL JUMPS BEFORE YOURS.\n", successfulJumpCt);
        } else if (betterThanCount <= -0.9 * successfulJumpCt) {
            System.out.println("HUMPH!  DON'T YOU HAVE ANY SPORTING BLOOD?  THERE WERE");
            System.out.printf("%d SUCCESSFUL JUMPS BEFORE YOURS AND YOU CAME IN %d JUMPS\n", successfulJumpCt, worseJumpCount);
            System.out.println("BETTER THAN THE WORST.  SHAPE UP!!!\n");
        } else {
            System.out.printf("HEY!  YOU PULLED THE RIP CORD MUCH TOO SOON.  %d SUCCESSFUL\n", successfulJumpCt);
            System.out.printf("JUMPS BEFORE YOURS AND YOU CAME IN NUMBER %d!  GET WITH IT!\n", betterThanCount);
        }
    }

    private void showSplatMessage(InitialJumpConditions initial, JumpResult jump) {
        double timeOfSplat = computeTimeOfSplat(initial, jump);
        System.out.printf("%10.2f  SPLAT\n", timeOfSplat);
    }

    /**
     * Returns the number of jumps for which this jump was better
     */
    private double computeTimeOfSplat(InitialJumpConditions initial, JumpResult jump) {
        final float V = initial.getTerminalVelocity();
        final float A = initial.getAcceleration();
        if (jump.hasReachedTerminalVelocity()) {
            return (V / A) + ((initial.getAltitude() - (V * V / (2 * A))) / V);
        }
        return Math.sqrt(2 * initial.getAltitude() / A);
    }

    private int countWorseHistoricalJumps(JumpResult jump) {
        return (int) pastSuccessfulJumpDistances.stream()
                .filter(distance -> jump.getDistance() < distance)
                .count();
    }

    private void showCleverSplatMessage() {
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
        System.out.println(messages.get(random.nextInt(10)));
    }

    private boolean playAgain() {
        if (askYesNo("DO YOU WANT TO PLAY AGAIN ")) {
            return true;
        }
        return askYesNo("PLEASE");
    }

    private float mphToFeetPerSec(float speed) {
        return speed * (5280.0f / 3600.0f);
    }

    private boolean askYesNo(String prompt) {
        System.out.printf("%s (YES OR NO) ", prompt);
        while (true) {
            String answer = scanner.next();
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

    private float chooseRandomAcceleration() {
        Planet planet = Planet.pickRandom();
        System.out.printf("%s %s. ACCELERATION=%.2f FT/SEC/SEC.\n", planet.getMessage(), planet.name(), planet.getAcceleration());
        return planet.getAcceleration();
    }

    enum Planet {
        MERCURY("FINE. YOU'RE ON", 12.2f),
        VENUS("ALL RIGHT. YOU'RE ON", 28.3f),
        EARTH("THEN YOU'RE ON", 32.16f),
        MOON("FINE. YOU'RE ON THE", 5.15f),
        MARS("ALL RIGHT. YOU'RE ON", 12.5f),
        JUPITER("THEN YOU'RE ON", 85.2f),
        SATURN("FINE. YOU'RE ON", 37.6f),
        URANUS("ALL RIGHT. YOU'RE ON", 33.8f),
        NEPTUNE("THEN YOU'RE ON", 39.6f),
        SUN("FINE. YOU'RE ON THE", 896.0f);

        private static final Random random = new Random();
        private final String message;
        private final float acceleration;

        Planet(String message, float acceleration) {
            this.message = message;
            this.acceleration = acceleration;
        }

        static Planet pickRandom() {
            return values()[random.nextInt(Planet.values().length)];
        }

        String getMessage() {
            return message;
        }

        float getAcceleration() {
            return acceleration;
        }
    }

    // Mutable
    static class JumpResult {
        private boolean reachedTerminalVelocity = false;
        private float distance; // from the ground

        public JumpResult(float distance) {
            this.distance = distance;
        }

        boolean isSplat() {
            return distance <= 0;
        }

        boolean hasReachedTerminalVelocity() {
            return reachedTerminalVelocity;
        }

        float getDistance() {
            return distance;
        }

        void setDistance(float distance) {
            this.distance = distance;
        }

        void setReachedTerminalVelocity() {
            reachedTerminalVelocity = true;
        }
    }

    // Immutable
    static class InitialJumpConditions {
        private final float originalTerminalVelocity;
        private final float originalAcceleration;
        private final float terminalVelocity;
        private final float acceleration;
        private final int altitude;

        private InitialJumpConditions(float originalTerminalVelocity, float originalAcceleration,
                                      float terminalVelocity, float acceleration, int altitude) {
            this.originalTerminalVelocity = originalTerminalVelocity;
            this.originalAcceleration = originalAcceleration;
            this.terminalVelocity = terminalVelocity;
            this.acceleration = acceleration;
            this.altitude = altitude;
        }

        // Create initial jump conditions with adjusted velocity/acceleration and a random initial altitude
        private static InitialJumpConditions create(float terminalVelocity, float gravitationalAcceleration) {
            final int altitude = (int) (9001.0f * random.nextFloat() + 1000);
            return new InitialJumpConditions(terminalVelocity, gravitationalAcceleration,
                    plusMinus5Percent(terminalVelocity), plusMinus5Percent(gravitationalAcceleration), altitude);
        }

        private static float plusMinus5Percent(float value) {
            return value + ((value * random.nextFloat()) / 20.0f) - ((value * random.nextFloat()) / 20.0f);
        }

        float getOriginalTerminalVelocity() {
            return originalTerminalVelocity;
        }

        float getOriginalAcceleration() {
            return originalAcceleration;
        }

        float getTerminalVelocity() {
            return terminalVelocity;
        }

        float getAcceleration() {
            return acceleration;
        }

        int getAltitude() {
            return altitude;
        }

        float getTimeOfTerminalAccelerationReached() {
            return terminalVelocity / acceleration;
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `81_Splat/javascript/splat.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

1. `print` 函数的作用是在文档中创建了一个输出元素（一个 `<textarea>` 元素），并将输入的文本内容将其添加到该元素中。这里的文本内容是通过 `input` 函数获取的，该函数会等待用户输入一段文本，并将其存储在变量 `input_str` 中。

2. `input` 函数的作用是获取用户输入的文本内容。它创建了一个 `<input>` 元素，设置了其 `type` 属性为 `text`，并设置了其 `length` 属性为 `50`（即允许用户输入的最大字符数）。这个 `<input>` 元素被添加到了文档中的一个元素（在文档中创建了一个 `<textarea>` 元素），然后该元素的 `focus` 事件被绑定。当用户按下键盘上的 `13` 键时，`input` 函数会获取到用户输入的文本内容，并将其存储在变量 `input_str` 中。

注意：`input` 函数创建的 `<input>` 元素只支持文本输入，不支持其他类型的输入（如 `file`、`submit` 等）。


```
// SPLAT
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```

This program appears to be a simple game where the user is prompted to choose a response when prompted to enter a password. The password is required to be at least 8 characters long. The program uses a random number generator to generate a password of a random length and difficulty level. The program also includes a condition that checks if the entered password matches the required length.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var aa = [];

// Main program
async function main()
{
    print(tab(33) + "SPLAT\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    for (i = 0; i <= 42; i++)
        aa[i] = 0;
    print("WELCOME TO 'SPLAT' -- THE GAME THAT SIMULATES A PARACHUTE\n");
    print("JUMP.  TRY TO OPEN YOUR CHUTE AT THE LAST POSSIBLE\n");
    print("MOMENT WITHOUT GOING SPLAT.\n");
    while (1) {
        print("\n");
        print("\n");
        d1 = 0;
        v = 0;
        a = 0;
        n = 0;
        m = 0;
        d1 = Math.floor(9001 * Math.random() + 1000);
        print("SELECT YOUR OWN TERMINAL VELOCITY (YES OR NO)");
        while (1) {
            a1s = await input();
            if (a1s == "YES" || a1s == "NO")
                break;
            print("YES OR NO");
        }
        if (a1s == "YES") {
            print("WHAT TERMINAL VELOCITY (MI/HR)");
            v1 = parseFloat(await input());
            v1 = v1 * (5280 / 3600);
        } else {
            v1 = Math.floor(1000 * Math.random());
            print("OK.  TERMINAL VELOCITY = " + v1 + " MI/HR\n");
        }
        v = v1 + ((v1 * Math.random()) / 20) - ((v1 * Math.random()) / 20);
        print("WANT TO SELECT ACCELERATION DUE TO GRAVITY (YES OR NO)");
        while (1) {
            b1s = await input();
            if (b1s == "YES" || b1s == "NO")
                break;
            print("YES OR NO");
        }
        if (b1s == "YES") {
            print("WHAT ACCELERATION (FT/SEC/SEC)");
            a2 = parseFloat(await input());
        } else {
            switch (Math.floor(1 + (10 * Math.random()))) {
                case 1:
                    print("FINE. YOU'RE ON MERCURY. ACCELERATION=12.2 FT/SEC/SEC.\n");
                    a2 = 12.2;
                    break;
                case 2:
                    print("ALL RIGHT. YOU'RE ON VENUS. ACCELERATION=28.3 FT/SEC/SEC.\n");
                    a2 = 28.3;
                    break;
                case 3:
                    print("THEN YOU'RE ON EARTH. ACCELERATION=32.16 FT/SEC/SEC.\n");
                    a2 = 32.16;
                    break;
                case 4:
                    print("FINE. YOU'RE ON THE MOON. ACCELERATION=5.15 FT/SEC/SEC.\n");
                    a2 = 5.15;
                    break;
                case 5:
                    print("ALL RIGHT. YOU'RE ON MARS. ACCELERATION=12.5 FT/SEC/SEC.\n");
                    a2 = 12.5;
                    break;
                case 6:
                    print("THEN YOU'RE ON JUPITER. ACCELERATION=85.2 FT/SEC/SEC.\n");
                    a2 = 85.2;
                    break;
                case 7:
                    print("FINE. YOU'RE ON SATURN. ACCELERATION=37.6 FT/SEC/SEC.\n");
                    a2 = 37.6;
                    break;
                case 8:
                    print("ALL RIGHT. YOU'RE ON URANUS. ACCELERATION=33.8 FT/SEC/SEC.\n");
                    a2 = 33.8;
                    break;
                case 9:
                    print("THEN YOU'RE ON NEPTUNE. ACCELERATION=39.6 FT/SEC/SEC.\n");
                    a2 = 39.6;
                    break;
                case 10:
                    print("FINE. YOU'RE ON THE SUN. ACCELERATION=896 FT/SEC/SEC.\n");
                    a2 = 896;
                    break;
            }
        }
        a = a2 + ((a2 * Math.random()) / 20) - ((a2 * Math.random()) / 20);
        print("\n");
        print("    ALTITUDE         = " + d1 + " FT\n");
        print("    TERM. VELOCITY   = " + v1 + " FT/SEC +/-5%\n");
        print("    ACCELERATION     = " + a2 + " FT/SEC/SEC +/-5%\n");
        print("SET THE TIMER FOR YOUR FREEFALL.\n");
        print("HOW MANY SECONDS");
        t = parseFloat(await input());
        print("HERE WE GO.\n");
        print("\n");
        print("TIME (SEC)\tDIST TO FALL (FT)\n");
        print("==========\t=================\n");
        terminal = false;
        crash = false;
        for (i = 0; i <= t; i += t / 8) {
            if (i > v / a) {
                terminal = true;
                break;
            }
            d = d1 - ((a / 2) * Math.pow(i, 2));
            if (d <= 0) {
                print(Math.sqrt(2 * d1 / a) + "\tSPLAT\n");
                crash = true;
                break;
            }
            print(i + "\t" + d + "\n");
        }
        if (terminal) {
            print("TERMINAL VELOCITY REACHED AT T PLUS " + v/a + " SECONDS.\n");
            for (; i <= t; i += t / 8) {
                d = d1 - ((Math.pow(v, 2) / (2 * a)) + (v * (i - (v / a))));
                if (d <= 0) {
                    print(((v / a) + ((d1 - (Math.pow(v, 2) / (2 * a))) / v)) + "\tSPLAT\n");
                    crash = true;
                    break;
                }
                print(i + "\t" + d + "\n");
            }
        }
        if (!crash) {
            print("CHUTE OPEN\n");
            k = 0;
            k1 = 0;
            for (j = 0; j <= 42; j++) {
                if (aa[j] == 0)
                    break;
                k++;
                if (d < aa[j])
                    k1++;
            }
            // In original jumps to line 540 (undefined) when table is full
            aa[j] = d;
            if (j <= 2) {
                print("AMAZING!!! NOT BAD FOR YOUR ");
                if (j == 0)
                    print("1ST ");
                else if (j == 1)
                    print("2ND ");
                else
                    print("3RD ");
                print("SUCCESSFUL JUMP!!!\n");
            } else {
                if (k - k1 <= 0.1 * k) {
                    print("WOW!  THAT'S SOME JUMPING.  OF THE " + k + " SUCCESSFUL JUMPS\n");
                    print("BEFORE YOURS, ONLY " + (k - k1) + " OPENED THEIR CHUTES LOWER THAN\n");
                    print("YOU DID.\n");
                } else if (k - k1 <= 0.25 * k) {
                    print("PRETTY GOOD! " + k + " SUCCESSFUL JUMPS PRECEDED YOURS AND ONLY\n");
                    print((k - k1) + " OF THEM GOT LOWER THAN YOU DID BEFORE THEIR CHUTES\n");
                    print("OPENED.\n");
                } else if (k - k1 <= 0.5 * k) {
                    print("NOT BAD.  THERE HAVE BEEN " + k + " SUCCESSFUL JUMPS BEFORE YOURS.\n");
                    print("YOU WERE BEATEN OUT BY " + (k - k1) + " OF THEM.\n");
                } else if (k - k1 <= 0.75 * k) {
                    print("CONSERVATIVE, AREN'T YOU?  YOU RANKED ONLY " + (k - k1) + " IN THE\n");
                    print(k + " SUCCESSFUL JUMPS BEFORE YOURS.\n");
                } else if (k - k1 <= 0.9 * k) {
                    print("HUMPH!  DON'T YOU HAVE ANY SPORTING BLOOD?  THERE WERE\n");
                    print(k + " SUCCESSFUL JUMPS BEFORE YOURS AND YOU CAME IN " + k1 + "JUMPS\n");
                    print("BETTER THAN THE WORST.  SHAPE UP!!!\n");
                } else {
                    print("HEY!  YOU PULLED THE RIP CORD MUCH TOO SOON.  " + k + " SUCCESSFUL\n");
                    print("JUMPS BEFORE YOURS AND YOU CAME IN NUMBER " + (k - k1) + "!  GET WITH IT!\n");
                }
            }
        } else {
            switch (Math.floor(1 + 10 * Math.random())) {
                case 1:
                    print("REQUIESCAT IN PACE.\n");
                    break;
                case 2:
                    print("MAY THE ANGEL OF HEAVEN LEAD YOU INTO PARADISE.\n");
                    break;
                case 3:
                    print("REST IN PEACE.\n");
                    break;
                case 4:
                    print("SON-OF-A-GUN.\n");
                    break;
                case 5:
                    print("#%&&%!$\n");
                    break;
                case 6:
                    print("A KICK IN THE PANTS IS A BOOST IF YOU'RE HEADED RIGHT.\n");
                    break;
                case 7:
                    print("HMMM. SHOULD HAVE PICKED A SHORTER TIME.\n");
                    break;
                case 8:
                    print("MUTTER. MUTTER. MUTTER.\n");
                    break;
                case 9:
                    print("PUSHING UP DAISIES.\n");
                    break;
                case 10:
                    print("EASY COME, EASY GO.\n");
                    break;
            }
            print("I'LL GIVE YOU ANOTHER CHANCE.\n");
        }
        while (1) {
            print("DO YOU WANT TO PLAY AGAIN");
            str = await input();
            if (str == "YES" || str == "NO")
                break;
            print("YES OR NO\n");
        }
        if (str == "YES")
            continue;
        print("PLEASE");
        while (1) {
            str = await input();
            if (str == "YES" || str == "NO")
                break;
            print("YES OR NO");
        }
        if (str == "YES")
            continue;
        break;
    }
    print("SSSSSSSSSS.\n");
    print("\n");
}

```

这是 C 语言中的一个简单程序，名为 `main()`。程序的主要作用是输出 "Hello World!"。

在 C 语言中，`main()` 是程序的入口点，当程序运行时，它首先会执行这个函数。所以，在这个程序中，`main()` 函数会首先输出 "Hello World!"，然后结束程序的运行。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `81_Splat/python/splat.py`

上述代码的作用是模拟一个降落伞打开的情况，玩家需要在最后一秒打开降落伞，否则就会坠落。玩家可以选择自己的终端速度，或者让计算机帮忙选择。此外，玩家还可以选择重力加速度，如果是的话，玩家将会降落在其中的一个行星上（除日地外）。

计算机程序会告诉玩家他们所跳跃的高度，以及自由落体的时间。然后它会将自由落体时间划分为八个时间段，并给出玩家下落过程中的报告。程序还会记录所有之前的跳跃，告诉玩家他们与之前成功跳跃的玩家的比较结果。如果玩家想要回顾之前的跳跃信息，也可以通过调用相应的方法来得到。


```
"""
SPLAT

Splat similates a parachute jump in which you try to open your parachute
at the last possible moment without going splat! You may select your own
terminal velocity or let the computer do it for you. You may also select
the acceleration due to gravity or, again, let the computer do it
in which case you might wind up on any one of the eight planets (out to
Neptune), the moon, or the sun.

The computer then tells you the height you're jumping from and asks for
the seconds of free fall. It then divides your free fall time into eight
intervals and gives you progress reports on the way down. The computer
also keeps track of all prior jumps and lets you know how you compared
with previous successful jumps. If you want to recall information from
```

这段代码是一个Python程序，它的主要目的是比较前一个运行成功跳跃的次数，然后将数组《successful_jumps》存储到磁盘上，并在每个运行开始时从磁盘上读取它。这个数组将用于记录成功跳跃的次数，以便在程序运行结束时统计成功跳跃的次数。

具体来说，这段代码首先导入了Python内置的`math`、`random`和`typing`模块。然后定义了一个`PAGE_WIDTH`常量，表示纸张的宽度为72字节。

接下来，定义了一个名为`successful_jumps`的列表类型。这个列表用于存储成功跳跃的次数。然后，编写了一段比较`successful_jumps`数组元素与前一个运行成功跳跃次数的代码。如果当前运行成功跳跃的次数与前一个运行成功跳跃的次数相同，则将《successful_jumps》数组从磁盘上读取并将其存储到`successful_jumps`数组中。

最后，这段代码还添加了一段注释，说明这个程序是在Oak Ridge Associated Universities开发的，以及说明这个程序是在2021年由Jonas Nockert开发的。


```
previous runs, then you should store the array `successful_jumps` on
disk and read it before each run.

John Yegge created this program while at the Oak Ridge Associated
Universities.

Ported in 2021 by Jonas Nockert / @lemonad

"""
from math import sqrt
from random import choice, random, uniform
from typing import List, Tuple

PAGE_WIDTH = 72


```

这两段代码分别是Python中的函数，其作用分别是询问用户输入数字和询问用户是否为“是”或“否”。

对于第一个函数 `numeric_input`，它接受一个question和一个默认值（default=0）。该函数通过while循环，不断询问用户输入数字。如果用户输入的answer为数字，函数将返回该数字，否则将返回默认值0。该函数不会抛出任何异常，而是直接返回用户输入的数字或默认值。

对于第二个函数 `yes_no_input`，它接受一个question和一个默认值（default="YES"）。该函数通过while循环，不断询问用户输入是否为"是"或"否"。如果用户输入的answer为"是"或"yes"，函数将返回True，否则返回False。该函数不会抛出任何异常，而是直接返回用户输入的answer值。


```
def numeric_input(question, default=0) -> float:
    """Ask user for a numeric value."""
    while True:
        answer_str = input(f"{question} [{default}]: ").strip() or default
        try:
            return float(answer_str)
        except ValueError:
            pass


def yes_no_input(question: str, default="YES") -> bool:
    """Ask user a yes/no question and returns True if yes, otherwise False."""
    answer = input(f"{question} (YES OR NO) [{default}]: ").strip() or default
    while answer.lower() not in ["n", "no", "y", "yes"]:
        answer = input(f"YES OR NO [{default}]: ").strip() or default
    return answer.lower() in ["y", "yes"]


```

这两函数的作用如下：

1. `get_terminal_velocity()`的作用是通过用户选择自己的终端速度或由计算机选择，返回终端速度（以米每秒为单位的浮点数）。函数首先会提示用户选择自己的终端速度，如果没有用户输入，则默认值为100，然后将选择好的值转换为英尺每秒，最后将其乘以5280除以3600得到米每秒的值。
2. `get_acceleration()`的作用是获取引力加速度，即自由落体运动的加速度。函数同样先提示用户是否要选择引力加速度，如果没有用户输入，则默认值为32.16。如果用户选择了引力加速度，则函数会根据用户选择的来源，选择不同的值，然后返回该值。


```
def get_terminal_velocity() -> float:
    """Terminal velocity by user or picked by computer."""
    if yes_no_input("SELECT YOUR OWN TERMINAL VELOCITY", default="NO"):
        v1 = numeric_input("WHAT TERMINAL VELOCITY (MI/HR)", default=100)
    else:
        # Computer picks 0-1000 terminal velocity.
        v1 = int(1000 * random())
        print(f"OK.  TERMINAL VELOCITY = {v1} MI/HR")

    # Convert miles/h to feet/s.
    return v1 * (5280 / 3600)


def get_acceleration() -> float:
    """Acceleration due to gravity by user or picked by computer."""
    if yes_no_input("WANT TO SELECT ACCELERATION DUE TO GRAVITY", default="NO"):
        a2 = numeric_input("WHAT ACCELERATION (FT/SEC/SEC)", default=32.16)
    else:
        body, a2 = pick_random_celestial_body()
        print(f"FINE. YOU'RE ON {body}. ACCELERATION={a2} FT/SEC/SEC.")
    return a2


```

这段代码定义了一个名为 `get_freefall_time` 的函数，用于计算给定初始高度、末速度和加速度的零或负自由落体时间。

自由落体运动中，物体在开始时具有一定的初速度和加速度，因此可以使用公式 `h = (1/2)gt^2 + vt + at^2` 计算物体在达到地面时的时间 `t`，其中 `h` 表示初始高度， `g` 表示重力加速度， `v` 表示初始速度， `a` 表示加速度。

然而，在自由落体运动中，如果初始速度为零，则需要通过计算得到时间 `t`。这就是这段代码的作用。函数接受一个参数 `t_freefall`，表示零或负自由落体时间，单位为秒。函数内部首先判断 `t_freefall` 的值是否为零，如果不是，则执行计算自由落体时间的数值输入命令，假设最大值为 10 秒。

如果 `t_freefall` 的值为零，说明已经给出了正确答案，函数返回零即可。否则，函数将返回计算得到的最小值，以确保不会出现负数。


```
def get_freefall_time() -> float:
    """User-guessed freefall time.

    The idea of the game is to pick a freefall time, given initial
    altitude, terminal velocity and acceleration, so the parachute
    as close to the ground as possible without going splat.
    """
    t_freefall: float = 0
    # A zero or negative freefall time is not handled by the motion
    # equations during the jump.
    while t_freefall <= 0:
        t_freefall = numeric_input("HOW MANY SECONDS", default=10)
    return t_freefall


```

This appears to be a Python implementation of a falling object simulation. The `simulate_falling_object` function takes in an initial altitude and terminal velocity as input and outputs the time taken for the object to reach the ground. If the object reaches the ground, the function prints a message and returns the ground time. If the object does not reach the ground, the function prints a message and returns the ground time. The function uses the principle of accounting for both the time up to reaching terminal velocity and the time beyond reaching terminal velocity in order to calculate the total time taken for the object to fall to the ground.


```
def jump() -> float:
    """Simulate a jump and returns the altitude where the chute opened.

    The idea is to open the chute as late as possible -- but not too late.
    """
    v: float = 0  # Terminal velocity.
    a: float = 0  # Acceleration.
    initial_altitude = int(9001 * random() + 1000)

    v1 = get_terminal_velocity()
    # Actual terminal velocity is +/-5% of v1.
    v = v1 * uniform(0.95, 1.05)

    a2 = get_acceleration()
    # Actual acceleration is +/-5% of a2.
    a = a2 * uniform(0.95, 1.05)

    print(
        "\n"
        f"    ALTITUDE         = {initial_altitude} FT\n"
        f"    TERM. VELOCITY   = {v1:.2f} FT/SEC +/-5%\n"
        f"    ACCELERATION     = {a2:.2f} FT/SEC/SEC +/-5%\n"
        "SET THE TIMER FOR YOUR FREEFALL."
    )
    t_freefall = get_freefall_time()
    print(
        "HERE WE GO.\n\n"
        "TIME (SEC)\tDIST TO FALL (FT)\n"
        "==========\t================="
    )

    terminal_velocity_reached = False
    is_splat = False
    for i in range(9):
        # Divide time for freefall into 8 intervals.
        t = i * (t_freefall / 8)
        # From the first equation of motion, v = v_0 + a * delta_t, with
        # initial velocity v_0 = 0, we can get the time when terminal velocity
        # is reached: delta_t = v / a.
        if t > v / a:
            if not terminal_velocity_reached:
                print(f"TERMINAL VELOCITY REACHED AT T PLUS {v / a:.2f} SECONDS.")
                terminal_velocity_reached = True
            # After having reached terminal velocity, the displacement is
            # composed of two parts:
            # 1. Displacement up to reaching terminal velocity:
            #    From the third equation of motion, v^2 = v_0^2 + 2 * a * d,
            #    with v_0 = 0, we can get the displacement using
            #    d1 = v^2 / (2 * a).
            # 2. Displacement beyond having reached terminal velocity:
            #    here, the displacement is just a function of the terminal
            #    velocity and the time passed after having reached terminal
            #    velocity: d2 = v * (t - t_reached_term_vel)
            d1 = (v**2) / (2 * a)
            d2 = v * (t - (v / a))
            altitude = initial_altitude - (d1 + d2)
            if altitude <= 0:
                # Time taken for an object to fall to the ground given
                # an initial altitude is composed of two parts after having
                # reached terminal velocity:
                # 1. time up to reaching terminal velocity: t1 = v / a
                # 2. time beyond having reached terminal velocity:
                #    here, the altitude that remains after having reached
                #    terminal velocity can just be divided by the constant
                #    terminal velocity to get the time it takes to reach the
                #    ground: t2 = altitude_remaining / v
                t1 = v / a
                t2 = (initial_altitude - d1) / v
                print_splat(t1 + t2)
                is_splat = True
                break
        else:
            # 1. Displacement before reaching terminal velocity:
            #    From the second equation of motion,
            #    d = v_0 * t + 0.5 * a * t^2, with v_0 = 0, we can get
            #    the displacement using d1 = a / 2 * t^2
            d1 = (a / 2) * (t**2)
            altitude = initial_altitude - d1
            if altitude <= 0:
                # Time taken for an object to fall to the ground given that
                # it never reaches terminal velocity can be calculated by
                # using the second equation of motion:
                # d = v_0 * t + 0.5 * a * t^2, with v_0 = 0, which
                # when solved for t becomes
                # t1 = sqrt(2 * d / a).
                t1 = sqrt(2 * initial_altitude / a)
                print_splat(t1)
                is_splat = True
                break
        print(f"{t:.2f}\t\t{altitude:.1f}")

    if not is_splat:
        print("CHUTE OPEN")
    return altitude


```

这段代码定义了一个名为 `pick_random_celestial_body` 的函数，它随机选择一个行星(可能是行星、月球或太阳)，并返回其名称和相应的引力常数(单位为千克·米平方/秒的平方)。

函数使用了Python标准库中的 `choice` 函数，它会在内置选项中随机选择一个元素。这里，我们可以看到它选择了以下行星：

- 水星(MERCURY)：引力值为 12.2 千克·米平方/秒的平方
- 金星(VENUS)：引力值为 28.3 千克·米平方/秒的平方
- 地球(EARTH)：引力值为 32.16 千克·米平方/秒的平方
- 月球(THE MOON)：引力值为 5.15 千克·米平方/秒的平方
- 马尔库斯(MARS)：引力值为 12.5 千克·米平方/秒的平方
- 木星(JUPITER)：引力值为 85.2 千克·米平方/秒的平方
- 土星(SATURN)：引力值为 37.6 千克·米平方/秒的平方
- 天王星(URANUS)：引力值为 33.8 千克·米平方/秒的平方
- 海王星(NEPTUNE)：引力值为 39.6 千克·米平方/秒的平方
- 太阳(THE SUN)：引力值为 896.0 千克·米平方/秒的平方

每次调用函数时，它都会随机选择其中一个行星，并返回其名称和相应的引力常数。


```
def pick_random_celestial_body() -> Tuple[str, float]:
    """Pick a random planet, the moon, or the sun with associated gravity."""
    return choice(
        [
            ("MERCURY", 12.2),
            ("VENUS", 28.3),
            ("EARTH", 32.16),
            ("THE MOON", 5.15),
            ("MARS", 12.5),
            ("JUPITER", 85.2),
            ("SATURN", 37.6),
            ("URANUS", 33.8),
            ("NEPTUNE", 39.6),
            ("THE SUN", 896.0),
        ]
    )


```



这个代码定义了一个名为 `jump_stats` 的函数，它接受两个参数 `previous_jumps` 是一个列表，表示之前的跳跃记录，`chute_altitude` 是一个浮点数，表示降落伞打开的高度。函数返回之前跳跃的数量和当前跳跃是否更好的次数。

函数实现了一个比较两个降落伞打开高度的方法。首先，遍历 `previous_jumps` 中的所有跳跃记录，计算每个跳跃高度和当前降落伞打开的高度之间的差距，然后使用一个循环将这些差距累积起来。接下来，遍历 `previous_jumps` 中的所有跳跃记录，对于每个跳跃，如果当前降落伞打开的高度低于该跳跃的高度，就累积起来。最后，函数返回之前跳跃的数量和当前跳跃是否更好的次数。

函数中还有一个名为 `print_splat` 的函数，它接受一个参数 `time_on_impact`，表示跳跃后到达的时间。函数实现了一个打印 parachute 打开太晚的提示信息。它从 `previous_jumps` 中的所有时间记录中选择一个适当的格言，打印出来。


```
def jump_stats(previous_jumps, chute_altitude) -> Tuple[int, int]:
    """Compare altitude when chute opened with previous successful jumps.

    Return the number of previous jumps and the number of times
    the current jump is better.
    """
    n_previous_jumps = len(previous_jumps)
    n_better = sum(1 for pj in previous_jumps if chute_altitude < pj)
    return n_previous_jumps, n_better


def print_splat(time_on_impact) -> None:
    """Parachute opened too late!"""
    print(f"{time_on_impact:.2f}\t\tSPLAT")
    print(
        choice(
            [
                "REQUIESCAT IN PACE.",
                "MAY THE ANGEL OF HEAVEN LEAD YOU INTO PARADISE.",
                "REST IN PEACE.",
                "SON-OF-A-GUN.",
                "#$%&&%!$",
                "A KICK IN THE PANTS IS A BOOST IF YOU'RE HEADED RIGHT.",
                "HMMM. SHOULD HAVE PICKED A SHORTER TIME.",
                "MUTTER. MUTTER. MUTTER.",
                "PUSHING UP DAISIES.",
                "EASY COME, EASY GO.",
            ]
        )
    )


```

It looks like this is a Python function that takes a value `k` and a variable `k1`, and prints a message depending on how many successful jumps there are before `k` happened. The message ranges from something like "PRETTY GOOD!" to "HEY! You Pulled The RIP CORD Much Too Soon."

If there were `k` successful jumps before `k` happened, the function would print something like this:
```
HEY! You Pulled The RIP CORD Much Too Soon. k Successful JUMPS BEFORE YOURS AND YOU CAME IN NUMBER k - k1!
```
If there were fewer than `k` successful jumps before `k` happened, the function would print something like this:
```
HEY! You Pulled The RIP CORD Much Too Soon. k Successful JUMPS BEFORE YOURS AND k - k1 IN THE Model
```
If there were any successful jumps at all before `k` happened, the function would print something like this:
```
HEY! You Pulled The RIP CORD Much Too Soon. PRETTY GOOD! k Successful JUMPS BEFORE YOURS
```
I'm sorry, but I don't have any information about the variable `k1`. It could be any number.


```
def print_results(n_previous_jumps, n_better) -> None:
    """Compare current jump to previous successful jumps."""
    k = n_previous_jumps
    k1 = n_better
    n_jumps = k + 1
    if n_jumps <= 3:
        order = ["1ST", "2ND", "3RD"]
        nth = order[n_jumps - 1]
        print(f"AMAZING!!! NOT BAD FOR YOUR {nth} SUCCESSFUL JUMP!!!")
    elif k - k1 <= 0.1 * k:
        print(
            f"WOW!  THAT'S SOME JUMPING.  OF THE {k} SUCCESSFUL JUMPS\n"
            f"BEFORE YOURS, ONLY {k - k1} OPENED THEIR CHUTES LOWER THAN\n"
            "YOU DID."
        )
    elif k - k1 <= 0.25 * k:
        print(
            f"PRETTY GOOD!  {k} SUCCESSFUL JUMPS PRECEDED YOURS AND ONLY\n"
            f"{k - k1} OF THEM GOT LOWER THAN YOU DID BEFORE THEIR CHUTES\n"
            "OPENED."
        )
    elif k - k1 <= 0.5 * k:
        print(
            f"NOT BAD.  THERE HAVE BEEN {k} SUCCESSFUL JUMPS BEFORE YOURS.\n"
            f"YOU WERE BEATEN OUT BY {k - k1} OF THEM."
        )
    elif k - k1 <= 0.75 * k:
        print(
            f"CONSERVATIVE, AREN'T YOU?  YOU RANKED ONLY {k - k1} IN THE\n"
            f"{k} SUCCESSFUL JUMPS BEFORE YOURS."
        )
    elif k - k1 <= 0.9 * k:
        print(
            "HUMPH!  DON'T YOU HAVE ANY SPORTING BLOOD?  THERE WERE\n"
            f"{k} SUCCESSFUL JUMPS BEFORE YOURS AND YOU CAME IN {k1} JUMPS\n"
            "BETTER THAN THE WORST.  SHAPE UP!!!"
        )
    else:
        print(
            f"HEY!  YOU PULLED THE RIP CORD MUCH TOO SOON.  {k} SUCCESSFUL\n"
            f"JUMPS BEFORE YOURS AND YOU CAME IN NUMBER {k - k1}!"
            "  GET WITH IT!"
        )


```

这两函数的主要目的是在适当的时机输出特定的字符串，而不是为了输出游戏。

第一个函数 `print_centered` 接收一个字符串参数 `msg`，并计算出该字符串需要向左或向右扩展多少空格才能使其在 `PAGE_WIDTH` 除以 2 的整数倍的位置上居中显示。然后，该函数打印出计算出的空格数量，然后将其与原始字符串连接起来，以在字符串中间居中显示该字符串。

第二个函数 `print_header` 接收一个字符串参数，该函数将调用 `print_centered` 函数打印出游戏标题。具体来说，该函数将调用 `print_centered` 函数打印出游戏标题中的 "SPLAT"，然后打印出 "CREATIVE COMPUTING" 和 "MORRISTOWN, NEW JERSEY"，接着打印一行新的空白行，该行包含游戏标题中所有空格的位置。然后，该函数将打印一行 "WELCOME TO 'SPLAT' -- THE GAME THAT SIMULATES A PARACHUTE\n" 消息。


```
def print_centered(msg: str) -> None:
    """Print centered text."""
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


def print_header() -> None:
    print_centered("SPLAT")
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print(
        "\n\n\n"
        "WELCOME TO 'SPLAT' -- THE GAME THAT SIMULATES A PARACHUTE\n"
        "JUMP.  TRY TO OPEN YOUR CHUTE AT THE LAST POSSIBLE\n"
        "MOMENT WITHOUT GOING SPLAT.\n\n"
    )


```

这段代码是一个Python程序，主要作用是实现一个跳伞游戏。在这个游戏中，玩家需要不断地跳跃，以避开各种障碍物和地面的攻击。游戏的胜利条件是成功跳出一定高度的障碍物。

具体来说，这段代码实现了一个以下几个功能：

1. 打印游戏开始时在屏幕上显示的标题。

2. 从一组跳跃次数成功的前几次跳跃中提取统计信息，如跳跃次数和跳跃成功次数，并打印出来。

3. 循环跳跃高度，如果跳跃高度大于0，就执行以下操作：

  a. 统计之前跳跃的跳跃次数和跳跃成功次数。

  b. 将跳跃高度大于0的跳跃次数和跳跃成功次数添加到“成功跳跃次数”列表中。

  c. 打印结果，包括之前跳跃的跳跃次数和跳跃成功次数。

4. 如果跳跃高度为0，显示“I'LL GIVE YOU ANOTHER CHANCE.”并再次询问玩家是否要再次跳跃。

5. 如果玩家不回答或两次输入“NO”，程序会一直显示“SSSSSSSSSS.”，表示游戏失败。


```
def main() -> None:
    print_header()

    successful_jumps: List[float] = []
    while True:
        chute_altitude = jump()
        if chute_altitude > 0:
            # We want the statistics on previous jumps (i.e. not including the
            # current jump.)
            n_previous_jumps, n_better = jump_stats(successful_jumps, chute_altitude)
            successful_jumps.append(chute_altitude)
            print_results(n_previous_jumps, n_better)
        else:
            # Splat!
            print("I'LL GIVE YOU ANOTHER CHANCE.")
        z = yes_no_input("DO YOU WANT TO PLAY AGAIN")
        if not z:
            z = yes_no_input("PLEASE")
            if not z:
                print("SSSSSSSSSS.")
                break


```

这段代码是一个条件判断语句，它会判断当前脚本是否作为主程序运行。如果是主程序运行，那么程序会执行if语句块内的内容。

具体来说，这段代码的意义是：如果当前脚本作为主程序运行，那么执行if语句块内的内容。否则，不执行if语句块内的内容。

换句话说，如果用户运行了这段代码所在的脚本，并且这个脚本不是作为主程序运行，那么脚本不会执行if语句块内的内容。如果用户运行了脚本并将其作为主程序运行，那么脚本会执行if语句块内的内容，从而执行main()函数。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Stars

In this game, the computer selects a random number from 1 to 100 (or any value you set). You try to guess the number and the computer gives you clues to tell you how close you’re getting. One star (\*) means you’re far away from the number; seven stars (\*\*\*\*\*\*\*) means you’re really close. You get 7 guesses.

On the surface this game is similar to GUESS; however, the guessing strategy is quite different. See if you can come up with one or more approaches to finding the mystery number.

Bob Albrecht of People’s Computer Company created this game.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=153)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=166)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `82_Stars/csharp/Game.cs`

This looks like a game of猜数字. The player is asked to guess a target number, and at each guess, the game will tell the player if their guess is too high or too low. The game will continue until the player either correctly guesses the target number or reaches the maximum number of guesses allowed. The game will also display the number of guesses it took the player to guess the target number.

The `AcceptGuesses` method is called when the player makes a guess. It reads the guess from the user, compares it to the target number, and displays appropriate stars based on the number of guesses it took to guess the number. If the player's guess is correct, the game will display a win message and end the game. If the player's guess is too high or too low, the game will display a loss message and end the game.

The `Play` method is called when the player makes a guess. It displays the instructions for the game, sets the maximum number of guesses allowed, and calls the `AcceptGuesses` method to accept the guess.


```
using System;
using Games.Common.IO;
using Games.Common.Randomness;
using Stars.Resources;

namespace Stars;

internal class Game
{
    private readonly TextIO _io;
    private readonly IRandom _random;
    private readonly int _maxNumber;
    private readonly int _maxGuessCount;

    public Game(TextIO io, IRandom random, int maxNumber, int maxGuessCount)
    {
        _io = io;
        _random = random;
        _maxNumber = maxNumber;
        _maxGuessCount = maxGuessCount;
    }

    internal void Play(Func<bool> playAgain)
    {
        DisplayIntroduction();

        do
        {
            Play();
        } while (playAgain.Invoke());
    }

    private void DisplayIntroduction()
    {
        _io.Write(Resource.Streams.Title);

        if (_io.ReadString("Do you want instructions").Equals("N", StringComparison.InvariantCultureIgnoreCase))
        {
            return;
        }

        _io.WriteLine(Resource.Formats.Instructions, _maxNumber, _maxGuessCount);
    }

    private void Play()
    {
        _io.WriteLine();
        _io.WriteLine();

        var target = _random.Next(_maxNumber) + 1;

        _io.WriteLine("Ok, I am thinking of a number.  Start guessing.");

        AcceptGuesses(target);
    }

    private void AcceptGuesses(int target)
    {
        for (int guessCount = 1; guessCount <= _maxGuessCount; guessCount++)
        {
            _io.WriteLine();
            var guess = _io.ReadNumber("Your guess");

            if (guess == target)
            {
                DisplayWin(guessCount);
                return;
            }

            DisplayStars(target, guess);
        }

        DisplayLoss(target);
    }

    private void DisplayStars(int target, float guess)
    {
        var stars = Math.Abs(guess - target) switch
        {
            >= 64 => "*",
            >= 32 => "**",
            >= 16 => "***",
            >= 8  => "****",
            >= 4  => "*****",
            >= 2  => "******",
            _     => "*******"
        };

        _io.WriteLine(stars);
    }

    private void DisplayWin(int guessCount)
    {
        _io.WriteLine();
        _io.WriteLine(new string('*', 79));
        _io.WriteLine();
        _io.WriteLine($"You got it in {guessCount} guesses!!!  Let's play again...");
    }

    private void DisplayLoss(int target)
    {
        _io.WriteLine();
        _io.WriteLine($"Sorry, that's {_maxGuessCount} guesses. The number was {target}.");
    }
}

```

# `82_Stars/csharp/Program.cs`

这段代码是一个基于控制台游戏的实现，主要作用是让玩家在最大猜测次数内猜出一个随机数。

首先，它引入了三个命名空间：Games.Common.IO、Games.Common.Randomness 和 Stars。

接着，它创建了一个名为 game 的新的上下文对象，通过 new ConsoleIO() 和 new RandomNumberGenerator() 方法设置输入输出流和随机数生成器，以及最大的猜测数量和最大猜测次数。

然后，它调用 game.Play() 方法来开始游戏。在此方法中，调用 game.WaitForAnGuess() 方法来等待玩家猜测，直到玩家猜中了为止。

总的来说，这段代码的主要目的是提供一个简单的基于控制台的游戏，让玩家猜测一个随机的数字，并在此过程中限制猜测次数和猜测的时间。


```
﻿using Games.Common.IO;
using Games.Common.Randomness;
using Stars;

var game = new Game(new ConsoleIO(), new RandomNumberGenerator(), maxNumber: 100, maxGuessCount: 7);

game.Play(() => true);

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `82_Stars/csharp/Resources/Resource.cs`



该代码是一个自定义的 .NET 类，名为 `Resource`。它包含两个内部类 `Streams` 和 `Formats`，以及一个私有方法 `GetString`。

`.NET` 中的 `System.IO` 和 `System.Reflection` 命名空间包含许多文件和类，这些都可以用于输入和输出数据。另外，`.NET` 的 `System.Runtime.CompilerServices` 命名空间中的 `Assembly` 类型可以用来获取可执行程序的资源和二进制文件。

`Stars.Resources.Streams` 类包含一个名为 `Title` 的类常量，它从名为 `GetStream` 的方法中获取。

`Stars.Resources.Formats` 类包含一个名为 `Instructions` 的类常量，它从名为 `GetString` 的方法中获取。

`GetString` 方法是一个私有方法，它使用 `Assembly` 和 `System.IO` 命名空间中的类和接口。它有一个参数 `name`，用于指定要获取的资源文件的名字。方法使用 `GetStream` 方法从指定的资源文件中读取内容，并将其返回给调用者。

`GetStream` 方法使用 `Assembly` 命名空间中的 `GetExecutingAssembly` 方法获取当前可执行程序的资源，并使用 `System.IO` 命名空间中的 `GetManifestResourceStream` 方法获取指定名称的资源文件 stream。

因此，该代码的作用是定义了一个可以获取本地资源文件中内容的类，这个类包含了从 `System.IO` 和 `System.Reflection` 命名空间中获取的文件读取和输出相关的类和方法，以及一个用于获取指定名称的资源文件名的私有方法。


```
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Stars.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Title => GetStream();
    }

    internal static class Formats
    {
        public static string Instructions => GetString();
    }

    private static string GetString([CallerMemberName] string name = null)
    {
        using var stream = GetStream(name);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private static Stream GetStream([CallerMemberName] string name = null)
        => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Stars.Resources.{name}.txt");
}
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `82_Stars/java/src/Stars.java`

This is a Java program that simulates a creative computing scenario. Here's a brief overview of what it does:

1. Creates a creative computing environment with a login screen and a player inputting their guess.
2. Displays a message asking for a guess, then accepts a keyword input from the player.
3. Checks whether the entered keyword is "Y" or "YES", and uses anyOneOf method to check.
4. Checks whether the entered text matches one of the following strings: "YOUR GUESS?"
```SQL
5. Y or
```
1. If the entered text matches, the program returns the entered integer.
2. If the entered text does not match any of the specified strings, the program returns a random number.
3. Prints a message asking the player to enter a guess.
4. Accepts a keyword input from the player.
5. Generates a random number.

Note: The random number generator is not thread-safe, and should be accessed with a synchronization mechanism to avoid race conditions.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Stars
 *
 * Based on the Basic game of Stars here
 * https://github.com/coding-horror/basic-computer-games/blob/main/82%20Stars/stars.bas
 *
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 *        new features - no additional text, error checking, etc has been added.
 */
public class Stars {

    public static final int HIGH_NUMBER_RANGE = 100;
    public static final int MAX_GUESSES = 7;

    private enum GAME_STATE {
        STARTING,
        INSTRUCTIONS,
        START_GAME,
        GUESSING,
        WON,
        LOST,
        GAME_OVER
    }

    // Used for keyboard input
    private final Scanner kbScanner;

    // Current game state
    private GAME_STATE gameState;

    // Players guess count;
    private int playerTotalGuesses;

    // Players current guess
    private int playerCurrentGuess;

    // Computers random number
    private int computersNumber;

    public Stars() {

        gameState = GAME_STATE.STARTING;

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     *
     */
    public void play() {

        do {
            switch (gameState) {

                // Show an introduction the first time the game is played.
                case STARTING:
                    intro();
                    gameState = GAME_STATE.INSTRUCTIONS;
                    break;

                // Ask if instructions are needed and display if yes
                case INSTRUCTIONS:
                    if(yesEntered(displayTextAndGetInput("DO YOU WANT INSTRUCTIONS? "))) {
                        instructions();
                    }
                    gameState = GAME_STATE.START_GAME;
                    break;

                // Generate computers number for player to guess, etc.
                case START_GAME:
                    init();
                    System.out.println("OK, I AM THINKING OF A NUMBER, START GUESSING.");
                    gameState = GAME_STATE.GUESSING;
                    break;

                // Player guesses the number until they get it or run out of guesses
                case GUESSING:
                    playerCurrentGuess = playerGuess();

                    // Check if the player guessed the number
                    if(playerCurrentGuess == computersNumber) {
                        gameState = GAME_STATE.WON;
                    } else {
                        // incorrect guess
                        showStars();
                        playerTotalGuesses++;
                        // Ran out of guesses?
                        if (playerTotalGuesses > MAX_GUESSES) {
                            gameState = GAME_STATE.LOST;
                        }
                    }
                    break;

                // Won game.
                case WON:

                    System.out.println(stars(79));
                    System.out.println("YOU GOT IT IN " + playerTotalGuesses
                            + " GUESSES!!!  LET'S PLAY AGAIN...");
                    gameState = GAME_STATE.START_GAME;
                    break;

                // Lost game by running out of guesses
                case LOST:
                    System.out.println("SORRY, THAT'S " + MAX_GUESSES
                            + " GUESSES. THE NUMBER WAS " + computersNumber);
                    gameState = GAME_STATE.START_GAME;
                    break;
            }
            // Endless loop since the original code did not allow the player to exit
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * Shows how close a players guess is to the computers number by
     * showing a series of stars - the more stars the closer to the
     * number.
     *
     */
    private void showStars() {
        int d = Math.abs(playerCurrentGuess - computersNumber);
        int starsToShow;
        if(d >=64) {
            starsToShow = 1;
        } else if(d >=32) {
            starsToShow = 2;
        } else if (d >= 16) {
            starsToShow = 3;
        } else if (d >=8) {
            starsToShow = 4;
        } else if( d>= 4) {
            starsToShow = 5;
        } else if(d>= 2) {
            starsToShow = 6;
        } else {
            starsToShow = 7;
        }
        System.out.println(stars(starsToShow));
    }

    /**
     * Show a number of stars (asterisks)
     * @param number the number of stars needed
     * @return the string encoded with the number of required stars
     */
    private String stars(int number) {
        char[] stars = new char[number];
        Arrays.fill(stars, '*');
        return new String(stars);
    }

    /**
     * Initialise variables before each new game
     *
     */
    private void init() {
        playerTotalGuesses = 1;
        computersNumber = randomNumber();
    }

    public void instructions() {
        System.out.println("I AM THINKING OF A WHOLE NUMBER FROM 1 TO " + HIGH_NUMBER_RANGE);
        System.out.println("TRY TO GUESS MY NUMBER.  AFTER YOU GUESS, I");
        System.out.println("WILL TYPE ONE OR MORE STARS (*).  THE MORE");
        System.out.println("STARS I TYPE, THE CLOSER YOU ARE TO MY NUMBER.");
        System.out.println("ONE STAR (*) MEANS FAR AWAY, SEVEN STARS (*******)");
        System.out.println("MEANS REALLY CLOSE!  YOU GET " + MAX_GUESSES + " GUESSES.");
    }

    public void intro() {
        System.out.println("STARS");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
    }

    /**
     * Get players guess from kb
     *
     * @return players guess as an int
     */
    private int playerGuess() {
        return Integer.parseInt((displayTextAndGetInput("YOUR GUESS? ")));
    }

    /**
     * Checks whether player entered Y or YES to a question.
     *
     * @param text  player string from kb
     * @return true of Y or YES was entered, otherwise false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * Check whether a string equals one of a variable number of values
     * Useful to check for Y or YES for example
     * Comparison is case insensitive.
     *
     * @param text source string
     * @param values a range of values to compare against the source string
     * @return true if a comparison was found in one of the variable number of strings passed
     */
    private boolean stringIsAnyValue(String text, String... values) {

        // Cycle through the variable number of values and test each
        for(String val:values) {
            if(text.equalsIgnoreCase(val)) {
                return true;
            }
        }

        // no matches
        return false;
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * Generate random number
     *
     * @return random number
     */
    private int randomNumber() {
        return (int) (Math.random()
                * (HIGH_NUMBER_RANGE) + 1);
    }
}

```