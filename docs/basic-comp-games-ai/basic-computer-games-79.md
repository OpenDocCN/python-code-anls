# BasicComputerGames源码解析 79

# `84_Super_Star_Trek/java/GameCallback.java`



这段代码定义了一个名为 `GameCallback` 的接口，旨在分离游戏控制中的翻滚控制，从星际地图和企业中分离，将其与游戏类分离。

具体来说，这个接口有以下几个方法：

- `enterNewQuadrant()`：进入一个新的 quadrant。
- `incrementStardate(double increment)`：增加星际地图中的 stardate 值。
- `endGameSuccess()`：游戏成功结束时调用。
- `endGameFail(boolean enterpriseDestroyed)`：游戏失败或企业没有摧毁时调用。

每当进入新的 quadrant、增加星际地图的 stardate 值或游戏成功结束时，`enterNewQuadrant()` 和 `incrementStardate()` 方法都会被调用。而当游戏失败或企业没有摧毁时，`endGameFail()` 方法会被调用。


```
/**
 * Interface for decoupling inversion of control from GalaxyMap and Enterprise towards the game class.
 */
public interface GameCallback {
    void enterNewQuadrant();
    void incrementStardate(double increment);
    void endGameSuccess();
    void endGameFail(boolean enterpriseDestroyed);
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/) by [Taciano Dreckmann Perez](https://github.com/taciano-perez).

Overview of Java classes:
- SuperStarTrekInstructions: displays game instructions
- SuperStarTrekGame: main game class
- GalaxyMap: map of the galaxy divided in quadrants and sectors, containing stars, bases, klingons, and the Enterprise
- Enterprise: the starship Enterprise
- GameCallback: interface allowing other classes to interact with the game class without circular dependencies 
- Util: utility methods

[This video](https://www.youtube.com/watch?v=cU3NKOnRNCI) describes the approach and the different steps followed to translate the game.

# `84_Super_Star_Trek/java/SuperStarTrekGame.java`

这段代码使用了Java的Stream API，创建了一个名为IntStream的流的实例。这个流可以接受任意数量的整数作为输入，并返回一个IntStream实例。

在这段注释中，作者解释了这段代码的目的是让读者了解这段代码的作用，以及它如何为人们带来什么样的启发和教育。作者还提到了原始程序的作者以及版权信息，还提到了这段代码的修改历程以及最终的日期。


```
import java.util.stream.IntStream;

/**
 * SUPER STARTREK - MAY 16,1978
 * ****        **** STAR TREK ****        ****
 * **** SIMULATION OF A MISSION OF THE STARSHIP ENTERPRISE,
 * **** AS SEEN ON THE STAR TREK TV SHOW.
 * **** ORIGINAL PROGRAM BY MIKE MAYFIELD, MODIFIED VERSION
 * **** PUBLISHED IN DEC'S "101 BASIC GAMES", BY DAVE AHL.
 * **** MODIFICATIONS TO THE LATTER (PLUS DEBUGGING) BY BOB
 * *** LEEDOM - APRIL & DECEMBER 1974,
 * *** WITH A LITTLE HELP FROM HIS FRIENDS . . .
 *
 * Ported to Java in Jan-Mar 2022 by
 * Taciano Dreckmann Perez (taciano.perez@gmail.com)
 */
```



This is a class that simulates a stardate progression game where the player controls a Klingon Battle Cruiser. The player can interact with the game using different commands, such as:

* incrementStardate(double increment): Increments the stardate by a specified amount.
* endGameFail(final boolean enterpriseDestroyed): Endcases the game with a failed mission or when the enterprise is destroyed.
* endGameSuccess(): Endcases the game with a successful mission.
* repeatGame(): Begins a new game and repeats the previous one until the player decides to exit.

The player can also use additional commands, such as:

* getEnterprise(): Retrieves the name of the enterprise.
* getRemainingKlingons(): Retrieves the number of Klingons remaining in the galaxy.
* setCommander(KlingonBattleCruiser command): Replaces the default commander for the Klingon Battle Cruiser with the given KlingonBattleCruiser.

Note that this is a basic implementation and does not include all features and functionalities that could be included in a real game.


```
public class SuperStarTrekGame implements GameCallback {

    // commands
    static final int COMMAND_NAV = 1;
    static final int COMMAND_SRS = 2;
    static final int COMMAND_LRS = 3;
    static final int COMMAND_PHA = 4;
    static final int COMMAND_TOR = 5;
    static final int COMMAND_SHE = 6;
    static final int COMMAND_DAM = 7;
    static final int COMMAND_COM = 8;
    static final int COMMAND_XXX = 9;

    // computer commands
    static final int COMPUTER_COMMAND_CUMULATIVE_GALACTIC_RECORD = 1;
    static final int COMPUTER_COMMAND_STATUS_REPORT = 2;
    static final int COMPUTER_COMMAND_PHOTON_TORPEDO_DATA = 3;
    static final int COMPUTER_COMMAND_STARBASE_NAV_DATA = 4;
    static final int COMPUTER_COMMAND_DIR_DIST_CALC = 5;
    static final int COMPUTER_COMMAND_GALAXY_MAP = 6;

    // other constants
    static final String COMMANDS = "NAVSRSLRSPHATORSHEDAMCOMXXX";

    // game state
    final GalaxyMap galaxyMap = new GalaxyMap();
    double stardate = Util.toInt(Util.random() * 20 + 20);
    int missionDuration = Math.max((25 + Util.toInt(Util.random() * 10)), galaxyMap.getKlingonsInGalaxy()+1);    // T9 (mission duration in stardates)
    boolean restart = false;

    // initial values
    final double initialStardate = stardate;

    public static void main(String[] args) {
        final SuperStarTrekGame game = new SuperStarTrekGame();
        printBanner();
        while (true) {
            game.orders();
            game.enterNewQuadrant();
            game.restart = false;
            game.commandLoop();
        }
    }

    static void printBanner() {
        IntStream.range(1, 10).forEach(i -> {
            Util.println("");
        });
        Util.println(
                """
                                                            ,------*------,
                                            ,-------------   '---  ------'
                                             '-------- --'      / /
                                                 ,---' '-------/ /--,
                                                  '----------------'

                                            THE USS ENTERPRISE --- NCC-1701"

                        """
        );
    }

    void orders() {
        Util.println("YOUR ORDERS ARE AS FOLLOWS:\n" +
                "     DESTROY THE " + galaxyMap.getKlingonsInGalaxy() + " KLINGON WARSHIP" + ((galaxyMap.getKlingonsInGalaxy() == 1) ? "" : "S") + " WHICH HAVE INVADED\n" +
                "   THE GALAXY BEFORE THEY CAN ATTACK FEDERATION HEADQUARTERS\n" +
                "   ON STARDATE " + initialStardate + missionDuration + "  THIS GIVES YOU " + missionDuration + " DAYS.  THERE " + ((galaxyMap.getBasesInGalaxy() == 1) ? "IS" : "ARE") + "\n" +
                "  " + galaxyMap.getBasesInGalaxy() + " STARBASE" + ((galaxyMap.getBasesInGalaxy() == 1) ? "" : "S") + " IN THE GALAXY FOR RESUPPLYING YOUR SHIP");
    }

    public void enterNewQuadrant() {
        galaxyMap.newQuadrant(stardate, initialStardate);
        shortRangeSensorScan();
    }

    void commandLoop() {
        while (!this.restart) {
            checkShipEnergy();
            String cmdStr = "";
            while ("".equals(cmdStr)) cmdStr = Util.inputStr("COMMAND");
            boolean foundCommand = false;
            for (int i = 1; i <= 9; i++) {
                if (Util.leftStr(cmdStr, 3).equals(Util.midStr(COMMANDS, 3 * i - 2, 3))) {
                    switch (i) {
                        case COMMAND_NAV:
                            navigation();
                            foundCommand = true;
                            break;
                        case COMMAND_SRS:
                            shortRangeSensorScan();
                            foundCommand = true;
                            break;
                        case COMMAND_LRS:
                            longRangeSensorScan();
                            foundCommand = true;
                            break;
                        case COMMAND_PHA:
                            firePhasers();
                            foundCommand = true;
                            break;
                        case COMMAND_TOR:
                            firePhotonTorpedo();
                            foundCommand = true;
                            break;
                        case COMMAND_SHE:
                            shieldControl();
                            foundCommand = true;
                            break;
                        case COMMAND_DAM:
                            galaxyMap.getEnterprise().damageControl(this);
                            foundCommand = true;
                            break;
                        case COMMAND_COM:
                            libraryComputer();
                            foundCommand = true;
                            break;
                        case COMMAND_XXX:
                            endGameFail(false);
                            foundCommand = true;
                            break;
                        default:
                            printCommandOptions();
                            foundCommand = true;
                    }
                }
            }
            if (!foundCommand) printCommandOptions();
        }
    }

    void checkShipEnergy() {
        final Enterprise enterprise = galaxyMap.getEnterprise();
        if (enterprise.getTotalEnergy() < 10 && (enterprise.getEnergy() <= 10 || enterprise.getDeviceStatus()[Enterprise.DEVICE_SHIELD_CONTROL] != 0)) {
            Util.println("\n** FATAL ERROR **   YOU'VE JUST STRANDED YOUR SHIP IN ");
            Util.println("SPACE");
            Util.println("YOU HAVE INSUFFICIENT MANEUVERING ENERGY,");
            Util.println(" AND SHIELD CONTROL");
            Util.println("IS PRESENTLY INCAPABLE OF CROSS");
            Util.println("-CIRCUITING TO ENGINE ROOM!!");
            endGameFail(false);
        }
    }

    void printCommandOptions() {
        Util.println("ENTER ONE OF THE FOLLOWING:");
        Util.println("  NAV  (TO SET COURSE)");
        Util.println("  SRS  (FOR SHORT RANGE SENSOR SCAN)");
        Util.println("  LRS  (FOR LONG RANGE SENSOR SCAN)");
        Util.println("  PHA  (TO FIRE PHASERS)");
        Util.println("  TOR  (TO FIRE PHOTON TORPEDOES)");
        Util.println("  SHE  (TO RAISE OR LOWER SHIELDS)");
        Util.println("  DAM  (FOR DAMAGE CONTROL REPORTS)");
        Util.println("  COM  (TO CALL ON LIBRARY-COMPUTER)");
        Util.println("  XXX  (TO RESIGN YOUR COMMAND)\n");
    }

    void navigation() {
        float course = Util.toInt(Util.inputFloat("COURSE (0-9)"));
        if (course == 9) course = 1;
        if (course < 1 || course >= 9) {
            Util.println("   LT. SULU REPORTS, 'INCORRECT COURSE DATA, SIR!'");
            return;
        }
        final Enterprise enterprise = galaxyMap.getEnterprise();
        final double[] deviceStatus = enterprise.getDeviceStatus();
        Util.println("WARP FACTOR (0-" + ((deviceStatus[Enterprise.DEVICE_WARP_ENGINES] < 0) ? "0.2" : "8") + ")");
        float warp = Util.inputFloat("");
        if (deviceStatus[Enterprise.DEVICE_WARP_ENGINES] < 0 && warp > .2) {
            Util.println("WARP ENGINES ARE DAMAGED.  MAXIMUM SPEED = WARP 0.2");
            return;
        }
        if (warp == 0) return;
        if (warp > 0 && warp <= 8) {
            int n = Util.toInt(warp * 8);
            if (enterprise.getEnergy() - n >= 0) {
                galaxyMap.klingonsMoveAndFire(this);
                repairDamagedDevices(course, warp, n);
                galaxyMap.moveEnterprise(course, warp, n, stardate, initialStardate, missionDuration, this);
            } else {
                Util.println("ENGINEERING REPORTS   'INSUFFICIENT ENERGY AVAILABLE");
                Util.println("                       FOR MANEUVERING AT WARP " + warp + "!'");
                if (enterprise.getShields() < n - enterprise.getEnergy() || deviceStatus[Enterprise.DEVICE_SHIELD_CONTROL] < 0) return;
                Util.println("DEFLECTOR CONTROL ROOM ACKNOWLEDGES " + enterprise.getShields() + " UNITS OF ENERGY");
                Util.println("                         PRESENTLY DEPLOYED TO SHIELDS.");
            }
        } else {
            Util.println("   CHIEF ENGINEER SCOTT REPORTS 'THE ENGINES WON'T TAKE");
            Util.println(" WARP " + warp + "!'");
        }
    }

    void repairDamagedDevices(final float course, final float warp, final int N) {
        final Enterprise enterprise = galaxyMap.getEnterprise();
        // repair damaged devices and print damage report
        enterprise.repairDamagedDevices(warp);
        if (Util.random() > .2) return;  // 80% chance no damage nor repair
        int randomDevice = Util.fnr();    // random device
        final double[] deviceStatus = enterprise.getDeviceStatus();
        if (Util.random() >= .6) {   // 40% chance of repair of random device
            enterprise.setDeviceStatus(randomDevice, deviceStatus[randomDevice] + Util.random() * 3 + 1);
            Util.println("DAMAGE CONTROL REPORT:  " + Enterprise.printDeviceName(randomDevice) + " STATE OF REPAIR IMPROVED\n");
        } else {            // 60% chance of damage of random device
            enterprise.setDeviceStatus(randomDevice, deviceStatus[randomDevice] - (Util.random() * 5 + 1));
            Util.println("DAMAGE CONTROL REPORT:  " + Enterprise.printDeviceName(randomDevice) + " DAMAGED");
        }
    }

    void longRangeSensorScan() {
        // LONG RANGE SENSOR SCAN CODE
        galaxyMap.longRangeSensorScan();
    }

    void firePhasers() {
        galaxyMap.firePhasers(this);
    }

    void firePhotonTorpedo() {
        galaxyMap.firePhotonTorpedo(stardate, initialStardate, missionDuration, this);
    }

    void shieldControl() {
        galaxyMap.getEnterprise().shieldControl();
    }

    void shortRangeSensorScan() {
        // SHORT RANGE SENSOR SCAN & STARTUP SUBROUTINE
        galaxyMap.shortRangeSensorScan(stardate);
    }

    void libraryComputer() {
        // REM LIBRARY COMPUTER CODE
        if (galaxyMap.getEnterprise().getDeviceStatus()[Enterprise.DEVICE_LIBRARY_COMPUTER] < 0) {
            Util.println("COMPUTER DISABLED");
            return;
        }
        while (true) {
            final float commandInput = Util.inputFloat("COMPUTER ACTIVE AND AWAITING COMMAND");
            if (commandInput < 0) return;
            Util.println("");
            int command = Util.toInt(commandInput) + 1;
            if (command >= COMPUTER_COMMAND_CUMULATIVE_GALACTIC_RECORD && command <= COMPUTER_COMMAND_GALAXY_MAP) {
                switch (command) {
                    case COMPUTER_COMMAND_CUMULATIVE_GALACTIC_RECORD:
                        galaxyMap.cumulativeGalacticRecord(true);
                        return;
                    case COMPUTER_COMMAND_STATUS_REPORT:
                        statusReport();
                        return;
                    case COMPUTER_COMMAND_PHOTON_TORPEDO_DATA:
                        galaxyMap.photonTorpedoData();
                        return;
                    case COMPUTER_COMMAND_STARBASE_NAV_DATA:
                        galaxyMap.starbaseNavData();
                        return;
                    case COMPUTER_COMMAND_DIR_DIST_CALC:
                        galaxyMap.directionDistanceCalculator();
                        return;
                    case COMPUTER_COMMAND_GALAXY_MAP:
                        galaxyMap.cumulativeGalacticRecord(false);
                        return;
                }
            } else {
                // invalid command
                Util.println("FUNCTIONS AVAILABLE FROM LIBRARY-COMPUTER:");
                Util.println("   0 = CUMULATIVE GALACTIC RECORD");
                Util.println("   1 = STATUS REPORT");
                Util.println("   2 = PHOTON TORPEDO DATA");
                Util.println("   3 = STARBASE NAV DATA");
                Util.println("   4 = DIRECTION/DISTANCE CALCULATOR");
                Util.println("   5 = GALAXY 'REGION NAME' MAP");
                Util.println("");
            }
        }
    }

    void statusReport() {
        Util.println("   STATUS REPORT:");
        Util.println("KLINGON" + ((galaxyMap.getKlingonsInGalaxy() > 1)? "S" : "")  + " LEFT: " + galaxyMap.getKlingonsInGalaxy());
        Util.println("MISSION MUST BE COMPLETED IN " + .1 * Util.toInt((initialStardate + missionDuration - stardate) * 10) + " STARDATES");
        if (galaxyMap.getBasesInGalaxy() >= 1) {
            Util.println("THE FEDERATION IS MAINTAINING " + galaxyMap.getBasesInGalaxy() + " STARBASE" + ((galaxyMap.getBasesInGalaxy() > 1)? "S" : "") + " IN THE GALAXY");
        } else {
            Util.println("YOUR STUPIDITY HAS LEFT YOU ON YOUR OWN IN");
            Util.println("  THE GALAXY -- YOU HAVE NO STARBASES LEFT!");
        }
        galaxyMap.getEnterprise().damageControl(this);
    }

    public void incrementStardate(double increment) {
        this.stardate += increment;
    }

    public void endGameFail(final boolean enterpriseDestroyed) {    // 6220
        if (enterpriseDestroyed) {
            Util.println("\nTHE ENTERPRISE HAS BEEN DESTROYED.  THEN FEDERATION ");
            Util.println("WILL BE CONQUERED");
        }
        Util.println("\nIT IS STARDATE " + stardate);
        Util.println("THERE WERE " + galaxyMap.getKlingonsInGalaxy() + " KLINGON BATTLE CRUISERS LEFT AT");
        Util.println("THE END OF YOUR MISSION.");
        repeatGame();
    }

    public void endGameSuccess() {
        Util.println("CONGRATULATION, CAPTAIN!  THE LAST KLINGON BATTLE CRUISER");
        Util.println("MENACING THE FEDERATION HAS BEEN DESTROYED.\n");
        Util.println("YOUR EFFICIENCY RATING IS " + (Math.sqrt(1000 * (galaxyMap.getRemainingKlingons() / (stardate - initialStardate)))));
        repeatGame();
    }

    void repeatGame() {
        Util.println("\n");
        if (galaxyMap.getBasesInGalaxy() != 0) {
            Util.println("THE FEDERATION IS IN NEED OF A NEW STARSHIP COMMANDER");
            Util.println("FOR A SIMILAR MISSION -- IF THERE IS A VOLUNTEER,");
            final String reply = Util.inputStr("LET HIM STEP FORWARD AND ENTER 'AYE'");
            if ("AYE".equals(reply)) {
                this.restart = true;
            } else {
                System.exit(0);
            }
        }
    }

}

```

# `84_Super_Star_Trek/java/SuperStarTrekInstructions.java`

It looks like this is a Java program that allows the user to choose from a list of options for a space game. The program uses a前期规划缓冲区 (BufferedReader and BufferedWriter) to read input from the user.

The program has several helper函数：print()用于打印消息；tab()用于打印制表符；收集函数(Collectors.joining())用于将文本字符串连接起来。

主要函数包括：

- print()：打印选择的游戏选项。
- tab()：打印制表符。
- inputStr()：从用户接收输入并返回字符串。

它还包含一个辅助类SensorScan，它实现了在空间游戏中的一个位置传感器扫描。


```
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * SUPER STARTREK INSTRUCTIONS
 * MAR 5, 1978
 * Just the instructions for SUPERSTARTREK
 *
 * Ported to Java in Jan-Feb 2022 by
 * Taciano Dreckmann Perez (taciano.perez@gmail.com)
 */
public class SuperStarTrekInstructions {

    public static void main(String[] args) {
        printBanner();
        final String reply = inputStr("DO YOU NEED INSTRUCTIONS (Y/N)? ");
        if ("Y".equals(reply)) {
            printInstructions();
        }
    }

    static void printBanner() {
        print(tab(10)+"*************************************");
        print(tab(10)+"*                                   *");
        print(tab(10)+"*                                   *");
        print(tab(10)+"*      * * SUPER STAR TREK * *      *");
        print(tab(10)+"*                                   *");
        print(tab(10)+"*                                   *");
        print(tab(10)+"*************************************");
    }

    static void printInstructions() {
        print("      INSTRUCTIONS FOR 'SUPER STAR TREK'");
        print("");
        print("1. WHEN YOU SEE \\COMMAND ?\\ PRINTED, ENTER ONE OF THE LEGAL");
        print("     COMMANDS (NAV,SRS,LRS,PHA,TOR,SHE,DAM,COM, OR XXX).");
        print("2. IF YOU SHOULD TYPE IN AN ILLEGAL COMMAND, YOU'LL GET A SHORT");
        print("     LIST OF THE LEGAL COMMANDS PRINTED OUT.");
        print("3. SOME COMMANDS REQUIRE YOU TO ENTER DATA (FOR EXAMPLE, THE");
        print("     'NAV' COMMAND COMES BACK WITH 'COURSE (1-9) ?'.)  IF YOU");
        print("     TYPE IN ILLEGAL DATA (LIKE NEGATIVE NUMBERS), THAN COMMAND");
        print("     WILL BE ABORTED");
        print("");
        print("     THE GALAXY IS DIVIDED INTO AN 8 X 8 QUADRANT GRID,");
        print("AND EACH QUADRANT IS FURTHER DIVIDED INTO AN 8 X 8 SECTOR GRID.");
        print("");
        print("     YOU WILL BE ASSIGNED A STARTING POINT SOMEWHERE IN THE");
        print("GALAXY TO BEGIN A TOUR OF DUTY AS COMANDER OF THE STARSHIP");
        print("\\ENTERPRISE\\; YOUR MISSION: TO SEEK AND DESTROY THE FLEET OF");
        print("KLINGON WARWHIPS WHICH ARE MENACING THE UNITED FEDERATION OF");
        print("PLANETS.");
        print("");
        print("     YOU HAVE THE FOLLOWING COMMANDS AVAILABLE TO YOU AS CAPTAIN");
        print("OF THE STARSHIP ENTERPRISE:");
        print("");
        print("\\NAV\\ COMMAND = WARP ENGINE CONTROL --");
        print("     COURSE IS IN A CIRCULAR NUMERICAL      4  3  2");
        print("     VECTOR ARRANGEMENT AS SHOWN             . . .");
        print("     INTEGER AND REAL VALUES MAY BE           ...");
        print("     USED.  (THUS COURSE 1.5 IS HALF-     5 ---*--- 1");
        print("     WAY BETWEEN 1 AND 2                      ...");
        print("                                             . . .");
        print("     VALUES MAY APPROACH 9.0, WHICH         6  7  8");
        print("     ITSELF IS EQUIVALENT TO 1.0");
        print("                                            COURSE");
        print("     ONE WARP FACTOR IS THE SIZE OF ");
        print("     ONE QUADTANT.  THEREFORE, TO GET");
        print("     FROM QUADRANT 6,5 TO 5,5, YOU WOULD");
        print("     USE COURSE 3, WARP FACTOR 1.");
        print("");
        print("\\SRS\\ COMMAND = SHORT RANGE SENSOR SCAN");
        print("     SHOWS YOU A SCAN OF YOUR PRESENT QUADRANT.");
        print("");
        print("     SYMBOLOGY ON YOUR SENSOR SCREEN IS AS FOLLOWS:");
        print("        <*> = YOUR STARSHIP'S POSITION");
        print("        +K+ = KLINGON BATTLE CRUISER");
        print("        >!< = FEDERATION STARBASE (REFUEL/REPAIR/RE-ARM HERE!)");
        print("         *  = STAR");
        print("");
        print("     A CONDENSED 'STATUS REPORT' WILL ALSO BE PRESENTED.");
        print("");
        print("\\LRS\\ COMMAND = LONG RANGE SENSOR SCAN");
        print("     SHOWS CONDITIONS IN SPACE FOR ONE QUADRANT ON EACH SIDE");
        print("     OF THE ENTERPRISE (WHICH IS IN THE MIDDLE OF THE SCAN)");
        print("     THE SCAN IS CODED IN THE FORM \\###\\, WHERE TH UNITS DIGIT");
        print("     IS THE NUMBER OF STARS, THE TENS DIGIT IS THE NUMBER OF");
        print("     STARBASES, AND THE HUNDRESDS DIGIT IS THE NUMBER OF");
        print("     KLINGONS.");
        print("");
        print("     EXAMPLE - 207 = 2 KLINGONS, NO STARBASES, & 7 STARS.");
        print("");
        print("\\PHA\\ COMMAND = PHASER CONTROL.");
        print("     ALLOWS YOU TO DESTROY THE KLINGON BATTLE CRUISERS BY ");
        print("     ZAPPING THEM WITH SUITABLY LARGE UNITS OF ENERGY TO");
        print("     DEPLETE THEIR SHIELD POWER.  (REMEMBER, KLINGONS HAVE");
        print("     PHASERS TOO!)");
        print("");
        print("\\TOR\\ COMMAND = PHOTON TORPEDO CONTROL");
        print("     TORPEDO COURSE IS THE SAME AS USED IN WARP ENGINE CONTROL");
        print("     IF YOU HIT THE KLINGON VESSEL, HE IS DESTROYED AND");
        print("     CANNOT FIRE BACK AT YOU.  IF YOU MISS, YOU ARE SUBJECT TO");
        print("     HIS PHASER FIRE.  IN EITHER CASE, YOU ARE ALSO SUBJECT TO ");
        print("     THE PHASER FIRE OF ALL OTHER KLINGONS IN THE QUADRANT.");
        print("");
        print("     THE LIBRARY-COMPUTER (\\COM\\ COMMAND) HAS AN OPTION TO ");
        print("     COMPUTE TORPEDO TRAJECTORY FOR YOU (OPTION 2)");
        print("");
        print("\\SHE\\ COMMAND = SHIELD CONTROL");
        print("     DEFINES THE NUMBER OF ENERGY UNITS TO BE ASSIGNED TO THE");
        print("     SHIELDS.  ENERGY IS TAKEN FROM TOTAL SHIP'S ENERGY.  NOTE");
        print("     THAN THE STATUS DISPLAY TOTAL ENERGY INCLUDES SHIELD ENERGY");
        print("");
        print("\\DAM\\ COMMAND = DAMMAGE CONTROL REPORT");
        print("     GIVES THE STATE OF REPAIR OF ALL DEVICES.  WHERE A NEGATIVE");
        print("     'STATE OF REPAIR' SHOWS THAT THE DEVICE IS TEMPORARILY");
        print("     DAMAGED.");
        print("");
        print("\\COM\\ COMMAND = LIBRARY-COMPUTER");
        print("     THE LIBRARY-COMPUTER CONTAINS SIX OPTIONS:");
        print("     OPTION 0 = CUMULATIVE GALACTIC RECORD");
        print("        THIS OPTION SHOWES COMPUTER MEMORY OF THE RESULTS OF ALL");
        print("        PREVIOUS SHORT AND LONG RANGE SENSOR SCANS");
        print("     OPTION 1 = STATUS REPORT");
        print("        THIS OPTION SHOWS THE NUMBER OF KLINGONS, STARDATES,");
        print("        AND STARBASES REMAINING IN THE GAME.");
        print("     OPTION 2 = PHOTON TORPEDO DATA");
        print("        WHICH GIVES DIRECTIONS AND DISTANCE FROM THE ENTERPRISE");
        print("        TO ALL KLINGONS IN YOUR QUADRANT");
        print("     OPTION 3 = STARBASE NAV DATA");
        print("        THIS OPTION GIVES DIRECTION AND DISTANCE TO ANY ");
        print("        STARBASE WITHIN YOUR QUADRANT");
        print("     OPTION 4 = DIRECTION/DISTANCE CALCULATOR");
        print("        THIS OPTION ALLOWS YOU TO ENTER COORDINATES FOR");
        print("        DIRECTION/DISTANCE CALCULATIONS");
        print("     OPTION 5 = GALACTIC /REGION NAME/ MAP");
        print("        THIS OPTION PRINTS THE NAMES OF THE SIXTEEN MAJOR ");
        print("        GALACTIC REGIONS REFERRED TO IN THE GAME.");
    }

    static void print(final String s) {
        System.out.println(s);
    }

    static String tab(final int n) {
        return IntStream.range(1, n).mapToObj(num -> " ").collect(Collectors.joining());
    }

    static String inputStr(final String message) {
        System.out.print(message + "? ");
        final BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        try {
            return reader.readLine();
        } catch (IOException ioe) {
            ioe.printStackTrace();
            return "";
        }
    }

}

```

# `84_Super_Star_Trek/java/Util.java`

This code appears to be a Java class with several methods for processing text input. The main method for this class takes a string message and a optional input delimiter ("-" or " "), and returns the processed string.

The `inputStr()` method takes a string message and returns a new string with all non-text characters removed and all text converted to lowercase. The method splits the input string on whitespace using the `split()` method and returns the lowercase string.

The `inputFloat()` method takes a string message and returns a float value. The method reads the input string line by line, splits it on whitespace using the `split()` method, and then tries to parse it as a float by calling the `parseFloat()` method. If the parse is successful, the method returns the float value.

The other methods `leftStr()`, `midStr()`, and `rightStr()` are not used in this class, but they appear to be methods for left, mid, and right aligning the input string, respectively.

It is worth noting that this code uses a `try`/`catch` block in the `inputFloat()` method to handle exceptions that may occur when parsing the input string as a float.


```
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Convenience utility methods for the Super Star Trek game.
 */
public class Util {

    static final Random random = new Random();

    public static float random() {
        return random.nextFloat();
    }

    public static int fnr() {    // 475
        // Generate a random integer from 1 to 8 inclusive.
        return toInt(random() * 7 + 1);
    }

    public static int toInt(final double num) {
        int x = (int) Math.floor(num);
        if (x < 0) x *= -1;
        return x;
    }

    public static void println(final String s) {
        System.out.println(s);
    }

    public static void print(final String s) {
        System.out.print(s);
    }

    public static String tab(final int n) {
        return IntStream.range(1, n).mapToObj(num -> " ").collect(Collectors.joining());
    }

    public static int strlen(final String s) {
        return s.length();
    }

    public static String inputStr(final String message) {
        System.out.print(message + "? ");
        final BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        try {
            return reader.readLine();
        } catch (IOException ioe) {
            ioe.printStackTrace();
            return "";
        }
    }

    public static int[] inputCoords(final String message) {
        while (true) {
            final String input = inputStr(message);
            try {
                final String[] splitInput = input.split(",");
                if (splitInput.length == 2) {
                    int x = Integer.parseInt(splitInput[0]);
                    int y = Integer.parseInt(splitInput[0]);
                    return new int[]{x, y};
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static float inputFloat(final String message) {
        while (true) {
            System.out.print(message + "? ");
            final BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            try {
                final String input = reader.readLine();
                if (input.length() > 0) {
                    return Float.parseFloat(input);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static String leftStr(final String input, final int len) {
        if (input == null || input.length() < len) return input;
        return input.substring(0, len);
    }

    public static String midStr(final String input, final int start, final int len) {
        if (input == null || input.length() < ((start - 1) + len)) return input;
        return input.substring(start - 1, (start - 1) + len);
    }

    public static String rightStr(final String input, final int len) {
        if (input == null || input.length() < len) return "";
        return input.substring(input.length() - len);
    }

    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();
        BigDecimal bd = new BigDecimal(Double.toString(value));
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }


}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `84_Super_Star_Trek/python/superstartrek.py`

这段代码是一个游戏文本，描述了一个星际探险飞船的虚拟mission。它将这段文字输出到了屏幕上，并在屏幕底部显示了游戏的制作人和版权信息。


```
"""
****        **** STAR TREK ****        ****
**** SIMULATION OF A MISSION OF THE STARSHIP ENTERPRISE,
**** AS SEEN ON THE STAR TREK TV SHOW.
**** ORIGINAL PROGRAM BY MIKE MAYFIELD, MODIFIED VERSION
**** PUBLISHED IN DEC'S "101 BASIC GAMES", BY DAVE AHL.
**** MODIFICATIONS TO THE LATTER (PLUS DEBUGGING) BY BOB
**** LEEDOM - APRIL & DECEMBER 1974,
**** WITH A LITTLE HELP FROM HIS FRIENDS . . .

  Output is identical to BASIC version except for a few
  fixes (as noted, search `bug`) and minor cleanup.
"""


```

这段代码是一个Python程序，它的作用是获取用户输入的多浮点数。程序中定义了一个名为`get_user_float`的函数，该函数接收一个字符串类型的参数`prompt`，然后从用户那里获取输入，并返回输入的浮点数。

具体来说，代码首先导入了`random`、`sys`、`dataclasses`、`enum`和`math`库。接着定义了一个名为`EnableHigh由由为`Final`指定的`Enum`类，然后定义了一个`get_user_float`函数。该函数首先创建一个空函数实参列表`params`，然后创建一个空函数实参字典`arguments`。接下来，函数内部使用`while`循环，该循环会不断地从用户那里获取输入，并将其存储在变量`answer`中。如果输入得到验证为有效的浮点数，函数将其存储在变量`answer_float`中，并返回该浮点数。如果输入不是有效的浮点数，函数会捕获并跳过该循环。

最后，代码导入了`typing`库，该库定义了`Callable`、`Dict`、`List`、`Optional`和`Tuple`这些类型。


```
import random
import sys
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Callable, Dict, Final, List, Optional, Tuple


def get_user_float(prompt: str) -> float:
    """Get input from user and return it."""
    while True:
        answer = input(prompt)
        try:
            answer_float = float(answer)
            return answer_float
        except ValueError:
            pass


```

这段代码定义了一个名为Entity的类，它是一个枚举类型，包含四个枚举值：klingon、ship、empty和starbase。

klingon的值为"+K+"，表示这是一个K值为13的方块。
ship的值为"<*>"，表示这是一个空格，表示这个位置没有方块。
empty的值为"***"，表示这是一个四个星号组成的三角形空。
starbase的值为">!<"，表示这是一个包含多个星号的复杂形状。
star的值为"* "，表示一个星号。
void的值为"   "，表示一个空格。

定义了一个名为Point的类，它是一个数据类，包含x和y两个整数属性。

在Entity类中，定义了一个名为Point的类成员函数__str__，用于打印出点对象的字符串表示形式。

该代码创建了一个Entity类对象，并创建了一个Point类型的数据类对象。该数据类对象在Entity类中作为Entity.klingon属性的一部分存在。


```
class Entity(Enum):
    klingon = "+K+"
    ship = "<*>"
    empty = "***"
    starbase = ">!<"
    star = " * "
    void = "   "


@dataclass
class Point:
    x: int
    y: int

    def __str__(self) -> str:
        return f"{self.x + 1} , {self.y + 1}"


```

这段代码定义了一个名为 `Position` 的类，其中包括一个 ` quadrant` 字段和一个 ` sector` 字段。它们都是 `Point` 类的实例。

`@dataclass` 表示这段代码是dataclass,dataclass是一种特殊的类，用于定义具有默认值的类型。接下来的类定义了dataclass的一些特征，如可读性，可维护性和可扩展性等。

`Position` 类中包含两个 `Point` 类的实例，分别定义了 "上" 和 "左" 两个方向。`quadrant` 和 `sector` 字段都使用 `Point` 类来存储该对象的位置。

由于 ` quadrant` 和 `sector` 字段都使用 `Point` 类，所以它们的初始值是默认位置，即 `(0, 0)`。如果调用 `quadrant` 或 `sector` 字段的 `__getter__` 方法，它会返回一个表示当前 quadrant 或 sector 的 `Point` 对象。如果调用 `__setter__` 方法，它会将新的 quadrant 或 sector 设置为目标位置。


```
@dataclass
class Position:
    """
    Every quadrant has 8 sectors

    Hence the position could also be represented as:
    x = quadrant.x * 8 + sector.x
    y = quadrant.y * 8 + sector.y
    """

    quadrant: Point
    sector: Point


@dataclass
```

这段代码定义了一个名为 QuadrantData 的类，它有两个整型成员 klingons 和 bases，以及一个整型成员 stars。接着，它定义了一个名为 num 的方法，该方法返回 quadrantData 对象的数值。

接下来，定义了一个名为 KlingonShip 的类，它有两个整型成员 sector 和 shield，以及一个名为 star_shaft 的浮点型成员。

这段代码定义了一个名为 QuadrantData 的类，它有两个整型成员 klingons 和 bases，以及一个整型成员 stars，还有一个名为 num 的方法，用于计算并返回一个整型值，该值等于 100 * klingons + 10 * bases + stars。

接着，定义了一个名为 KlingonShip 的类，它有两个整型成员 sector 和 shield，以及一个名为 star_shaft 的浮点型成员。


```
class QuadrantData:
    klingons: int
    bases: int
    stars: int

    def num(self) -> int:
        return 100 * self.klingons + 10 * self.bases + self.stars


@dataclass
class KlingonShip:
    sector: Point
    shield: float


```



This is a Python implementation of a game room in a universe where players can control a ship with its engines and weapons. The player can use its energy and shields for navigation and other abilities. The game room has four possible outcomes:

1. The player can successfully navigate through the room and leave the game room.
2. The player runs out of energy and can't leave the game room.
3. The player runs out of shields and can't leave the game room.
4. The player attacks another player, causing a shield or energy consumption.

The player can use its energy and shields to attack other players or its own teammates. It can also use its torpedoes to launchMissile attacks at other players. The player's success or failure in each instance is determined by various factors, such as the player's starting health, the difficulty level of the room, and the player's strategic choices.


```
class Ship:
    energy_capacity: int = 3000
    torpedo_capacity: int = 10

    def __init__(self) -> None:
        self.position = Position(Point(fnr(), fnr()), Point(fnr(), fnr()))
        self.energy: int = Ship.energy_capacity
        self.devices: Tuple[str, ...] = (
            "WARP ENGINES",
            "SHORT RANGE SENSORS",
            "LONG RANGE SENSORS",
            "PHASER CONTROL",
            "PHOTON TUBES",
            "DAMAGE CONTROL",
            "SHIELD CONTROL",
            "LIBRARY-COMPUTER",
        )
        self.damage_stats: List[float] = [0] * len(self.devices)
        self.shields = 0
        self.torpedoes = Ship.torpedo_capacity
        self.docked: bool = False  # true when docked at starbase

    def refill(self) -> None:
        self.energy = Ship.energy_capacity
        self.torpedoes = Ship.torpedo_capacity

    def maneuver_energy(self, n: int) -> None:
        """Deduct the energy for navigation from energy/shields."""
        self.energy -= n + 10

        if self.energy <= 0:
            print("SHIELD CONTROL SUPPLIES ENERGY TO COMPLETE THE MANEUVER.")
            self.shields += self.energy
            self.energy = 0
            self.shields = max(0, self.shields)

    def shield_control(self) -> None:
        """Raise or lower the shields."""
        if self.damage_stats[6] < 0:
            print("SHIELD CONTROL INOPERABLE")
            return

        while True:
            energy_to_shield = input(
                f"ENERGY AVAILABLE = {self.energy + self.shields} NUMBER OF UNITS TO SHIELDS? "
            )
            if len(energy_to_shield) > 0:
                x = int(energy_to_shield)
                break

        if x < 0 or self.shields == x:
            print("<SHIELDS UNCHANGED>")
            return

        if x > self.energy + self.shields:
            print(
                "SHIELD CONTROL REPORTS  'THIS IS NOT THE FEDERATION "
                "TREASURY.'\n"
                "<SHIELDS UNCHANGED>"
            )
            return

        self.energy += self.shields - x
        self.shields = x
        print("DEFLECTOR CONTROL ROOM REPORT:")
        print(f"  'SHIELDS NOW AT {self.shields} UNITS PER YOUR COMMAND.'")


```

This is a class for a game board, where each row, column, and module of the board is represented by a list of `Entity` objects. The `Entity` class is a base class for an object that can have a value or a type.

The `GameBoard` class is a superclass of `Entity` that inherits from `Object` and has additional properties specific to a game board, such as `klingon_ships`, `starbase`, and `nb_stars`, which are not present in `Object`.

The `GameBoard` class has methods to set and get values for each cell, find empty cells, and populate the board with a ship's crew. It also has a method `find_empty_place` that returns the coordinates of the next empty cell, and a method `populate_quadrant` that fills the board with the ship's crew.

The `find_empty_place` method is used to find the position of a starbase in the current sector, and is an implementation of the `find_empty_place` method in the `Object` class.

The `populate_quadrant` method is used to populate the board with the ship's crew. It is an implementation of the `populate_board` method in the `Object` class.


```
class Quadrant:
    def __init__(
        self,
        point: Point,  # position of the quadrant
        population: QuadrantData,
        ship_position: Position,
    ) -> None:
        """Populate quadrant map"""
        assert 0 <= point.x <= 7 and 0 <= point.y <= 7
        self.name = Quadrant.quadrant_name(point.x, point.y, False)

        self.nb_klingons = population.klingons
        self.nb_bases = population.bases
        self.nb_stars = population.stars

        # extra delay in repairs at base
        self.delay_in_repairs_at_base: float = 0.5 * random.random()

        # Klingons in current quadrant
        self.klingon_ships: List[KlingonShip] = []

        # Initialize empty: save what is at which position
        self.data = [[Entity.void for _ in range(8)] for _ in range(8)]

        self.populate_quadrant(ship_position)

    @classmethod
    def quadrant_name(cls, row: int, col: int, region_only: bool = False) -> str:
        """Return quadrant name visible on scans, etc."""
        region1 = [
            "ANTARES",
            "RIGEL",
            "PROCYON",
            "VEGA",
            "CANOPUS",
            "ALTAIR",
            "SAGITTARIUS",
            "POLLUX",
        ]
        region2 = [
            "SIRIUS",
            "DENEB",
            "CAPELLA",
            "BETELGEUSE",
            "ALDEBARAN",
            "REGULUS",
            "ARCTURUS",
            "SPICA",
        ]
        modifier = ["I", "II", "III", "IV"]

        quadrant = region1[row] if col < 4 else region2[row]

        if not region_only:
            quadrant += " " + modifier[col % 4]

        return quadrant

    def set_value(self, x: float, y: float, entity: Entity) -> None:
        self.data[round(x)][round(y)] = entity

    def get_value(self, x: float, y: float) -> Entity:
        return self.data[round(x)][round(y)]

    def find_empty_place(self) -> Tuple[int, int]:
        """Find an empty location in the current quadrant."""
        while True:
            row, col = fnr(), fnr()
            if self.get_value(row, col) == Entity.void:
                return row, col

    def populate_quadrant(self, ship_position: Position) -> None:
        self.set_value(ship_position.sector.x, ship_position.sector.y, Entity.ship)
        for _ in range(self.nb_klingons):
            x, y = self.find_empty_place()
            self.set_value(x, y, Entity.klingon)
            self.klingon_ships.append(
                KlingonShip(
                    Point(x, y), klingon_shield_strength * (0.5 + random.random())
                )
            )
        if self.nb_bases > 0:
            # Position of starbase in current sector
            starbase_x, starbase_y = self.find_empty_place()
            self.starbase = Point(starbase_x, starbase_y)
            self.set_value(starbase_x, starbase_y, Entity.starbase)
        for _ in range(self.nb_stars):
            x, y = self.find_empty_place()
            self.set_value(x, y, Entity.star)

    def __str__(self) -> str:
        quadrant_string = ""
        for row in self.data:
            for entity in row:
                quadrant_string += entity.value
        return quadrant_string


```

It looks like this is a implementation of the galaxy-的意义， where the player controls a group of Klingons, and the objective is to defeat the other Klingons that have突然 appeared. It appears that the Klingons have different numbers of allies based on how many "clans" they have, and the player is trying to defeat the "evil" Klingons who have multiplied. It also appears that the player has a ship that moves around the galaxy and can conquer nearby planets, which increases the player's base count, and can also be used to defeat the evil Klingons. It's not clear from this code if there is any way for the player to retreat or if there is a time limit for the mission.


```
class World:
    def __init__(
        self,
        total_klingons: int = 0,  # Klingons at start of game
        bases_in_galaxy: int = 0,
    ) -> None:
        self.ship = Ship()
        self.initial_stardate = 100 * random.randint(20, 39)
        self.stardate: float = self.initial_stardate
        self.mission_duration = random.randint(25, 34)

        # Enemy
        self.remaining_klingons = total_klingons

        # Player starbases
        self.bases_in_galaxy = bases_in_galaxy

        self.galaxy_map: List[List[QuadrantData]] = [
            [QuadrantData(0, 0, 0) for _ in range(8)] for _ in range(8)
        ]
        self.charted_galaxy_map: List[List[QuadrantData]] = [
            [QuadrantData(0, 0, 0) for _ in range(8)] for _ in range(8)
        ]

        # initialize contents of galaxy
        for x in range(8):
            for y in range(8):
                r1 = random.random()

                if r1 > 0.98:
                    quadrant_klingons = 3
                elif r1 > 0.95:
                    quadrant_klingons = 2
                elif r1 > 0.80:
                    quadrant_klingons = 1
                else:
                    quadrant_klingons = 0
                self.remaining_klingons += quadrant_klingons

                quadrant_bases = 0
                if random.random() > 0.96:
                    quadrant_bases = 1
                    self.bases_in_galaxy += 1
                self.galaxy_map[x][y] = QuadrantData(
                    quadrant_klingons, quadrant_bases, 1 + fnr()
                )

        if self.remaining_klingons > self.mission_duration:
            self.mission_duration = self.remaining_klingons + 1

        if self.bases_in_galaxy == 0:  # original has buggy extra code here
            self.bases_in_galaxy = 1
            self.galaxy_map[self.ship.position.quadrant.x][
                self.ship.position.quadrant.y
            ].bases += 1

        curr = self.ship.position.quadrant
        self.quadrant = Quadrant(
            self.ship.position.quadrant,
            self.galaxy_map[curr.x][curr.y],
            self.ship.position,
        )

    def remaining_time(self) -> float:
        return self.initial_stardate + self.mission_duration - self.stardate

    def has_mission_ended(self) -> bool:
        return self.remaining_time() < 0


```

This appears to be a game of interstellarcraft where the player controls a Klingon Battle Cruiser, and the objective is to destroy the Federation (or


```
class Game:
    """Handle user actions"""

    def __init__(self) -> None:
        self.restart = False
        self.world = World()

    def startup(self) -> None:
        """Initialize the game variables and map, and print startup messages."""
        print(
            "\n\n\n\n\n\n\n\n\n\n\n"
            "                                    ,------*------,\n"
            "                    ,-------------   '---  ------'\n"
            "                     '-------- --'      / /\n"
            "                         ,---' '-------/ /--,\n"
            "                          '----------------'\n\n"
            "                    THE USS ENTERPRISE --- NCC-1701\n"
            "\n\n\n\n"
        )
        world = self.world
        print(
            "YOUR ORDERS ARE AS FOLLOWS:\n"
            f"     DESTROY THE {world.remaining_klingons} KLINGON WARSHIPS WHICH HAVE INVADED\n"
            "   THE GALAXY BEFORE THEY CAN ATTACK FEDERATION HEADQUARTERS\n"
            f"   ON STARDATE {world.initial_stardate+world.mission_duration}. "
            f" THIS GIVES YOU {world.mission_duration} DAYS. THERE "
            f"{'IS' if world.bases_in_galaxy == 1 else 'ARE'}\n"
            f"   {world.bases_in_galaxy} "
            f"STARBASE{'' if world.bases_in_galaxy == 1 else 'S'} IN THE GALAXY FOR "
            "RESUPPLYING YOUR SHIP.\n"
        )

    def new_quadrant(self) -> None:
        """Enter a new quadrant: populate map and print a short range scan."""
        world = self.world
        ship = world.ship
        q = ship.position.quadrant

        world.quadrant = Quadrant(
            q,
            world.galaxy_map[q.x][q.y],
            ship.position,
        )

        world.charted_galaxy_map[q.x][q.y] = world.galaxy_map[q.x][q.y]

        if world.stardate == world.initial_stardate:
            print("\nYOUR MISSION BEGINS WITH YOUR STARSHIP LOCATED")
            print(f"IN THE GALACTIC QUADRANT, '{world.quadrant.name}'.\n")
        else:
            print(f"\nNOW ENTERING {world.quadrant.name} QUADRANT . . .\n")

        if world.quadrant.nb_klingons != 0:
            print("COMBAT AREA      CONDITION RED")
            if ship.shields <= 200:
                print("   SHIELDS DANGEROUSLY LOW")
        self.short_range_scan()

    def fnd(self, i: int) -> float:
        """Find distance between Enterprise and i'th Klingon warship."""
        ship = self.world.ship.position.sector
        klingons = self.world.quadrant.klingon_ships[i].sector
        return sqrt((klingons.x - ship.x) ** 2 + (klingons.y - ship.y) ** 2)

    def klingons_fire(self) -> None:
        """Process nearby Klingons firing on Enterprise."""
        ship = self.world.ship

        if self.world.quadrant.nb_klingons <= 0:
            return
        if ship.docked:
            print("STARBASE SHIELDS PROTECT THE ENTERPRISE")
            return

        for i, klingon_ship in enumerate(self.world.quadrant.klingon_ships):
            if klingon_ship.shield <= 0:
                continue

            h = int((klingon_ship.shield / self.fnd(i)) * (random.random() + 2))
            ship.shields -= h
            klingon_ship.shield /= random.random() + 3
            print(f" {h} UNIT HIT ON ENTERPRISE FROM SECTOR {klingon_ship.sector} ")
            if ship.shields <= 0:
                self.end_game(won=False, quit=False, enterprise_killed=True)
                return
            print(f"      <SHIELDS DOWN TO {ship.shields} UNITS>")
            if h >= 20 and random.random() < 0.60 and h / ship.shields > 0.02:
                device = fnr()
                ship.damage_stats[device] -= h / ship.shields + 0.5 * random.random()
                print(
                    f"DAMAGE CONTROL REPORTS  '{ship.devices[device]} DAMAGED BY THE HIT'"
                )

    def phaser_control(self) -> None:
        """Take phaser control input and fire phasers."""
        world = self.world
        klingon_ships = world.quadrant.klingon_ships
        ship = world.ship

        if ship.damage_stats[3] < 0:
            print("PHASERS INOPERATIVE")
            return

        if self.world.quadrant.nb_klingons <= 0:
            print("SCIENCE OFFICER SPOCK REPORTS  'SENSORS SHOW NO ENEMY SHIPS")
            print("                                IN THIS QUADRANT'")
            return

        if ship.damage_stats[7] < 0:
            print("COMPUTER FAILURE HAMPERS ACCURACY")

        print(f"PHASERS LOCKED ON TARGET;  ENERGY AVAILABLE = {ship.energy} UNITS")
        phaser_firepower: float = 0
        while True:
            while True:
                units_to_fire = input("NUMBER OF UNITS TO FIRE? ")
                if len(units_to_fire) > 0:
                    phaser_firepower = int(units_to_fire)
                    break
            if phaser_firepower <= 0:
                return
            if ship.energy >= phaser_firepower:
                break
            print(f"ENERGY AVAILABLE = {ship.energy} UNITS")

        ship.energy -= phaser_firepower
        if ship.damage_stats[7] < 0:  # bug in original, was d[6]
            phaser_firepower *= random.random()

        phaser_per_klingon = int(phaser_firepower / self.world.quadrant.nb_klingons)
        for i, klingon_ship in enumerate(klingon_ships):
            if klingon_ship.shield <= 0:
                continue

            h = int((phaser_per_klingon / self.fnd(i)) * (random.random() + 2))
            if h <= 0.15 * klingon_ship.shield:
                print(f"SENSORS SHOW NO DAMAGE TO ENEMY AT {klingon_ship.sector}")
            else:
                klingon_ship.shield -= h
                print(f" {h} UNIT HIT ON KLINGON AT SECTOR {klingon_ship.sector}")
                if klingon_ship.shield <= 0:
                    print("*** KLINGON DESTROYED ***")
                    self.world.quadrant.nb_klingons -= 1
                    world.remaining_klingons -= 1
                    world.quadrant.set_value(
                        klingon_ship.sector.x, klingon_ship.sector.y, Entity.void
                    )
                    klingon_ship.shield = 0
                    world.galaxy_map[ship.position.quadrant.x][
                        ship.position.quadrant.y
                    ].klingons -= 1
                    world.charted_galaxy_map[ship.position.quadrant.x][
                        ship.position.quadrant.y
                    ] = world.galaxy_map[ship.position.quadrant.x][
                        ship.position.quadrant.y
                    ]
                    if world.remaining_klingons <= 0:
                        self.end_game(won=True, quit=False)
                        return
                else:
                    print(
                        f"   (SENSORS SHOW {round(klingon_ship.shield,6)} UNITS REMAINING)"
                    )

        self.klingons_fire()

    def photon_torpedoes(self) -> None:
        """Take photon torpedo input and process firing of torpedoes."""
        world = self.world
        klingon_ships = world.quadrant.klingon_ships
        ship = world.ship

        if ship.torpedoes <= 0:
            print("ALL PHOTON TORPEDOES EXPENDED")
            return
        if ship.damage_stats[4] < 0:
            print("PHOTON TUBES ARE NOT OPERATIONAL")
            return

        cd = get_user_float("PHOTON TORPEDO COURSE (1-9)? ")
        if cd == 9:
            cd = 1
        if cd < 1 or cd >= 9:
            print("ENSIGN CHEKOV REPORTS, 'INCORRECT COURSE DATA, SIR!'")
            return

        cdi = int(cd)

        # Interpolate direction:
        dx = dirs[cdi - 1][0] + (dirs[cdi][0] - dirs[cdi - 1][0]) * (cd - cdi)
        dy = dirs[cdi - 1][1] + (dirs[cdi][1] - dirs[cdi - 1][1]) * (cd - cdi)

        ship.energy -= 2
        ship.torpedoes -= 1

        # Exact position
        x: float = ship.position.sector.x
        y: float = ship.position.sector.y

        # Rounded position (to coordinates)
        torpedo_x, torpedo_y = x, y
        print("TORPEDO TRACK:")
        while True:
            x += dx
            y += dy
            torpedo_x, torpedo_y = round(x), round(y)
            if torpedo_x < 0 or torpedo_x > 7 or torpedo_y < 0 or torpedo_y > 7:
                print("TORPEDO MISSED")
                self.klingons_fire()
                return
            print(f"                {torpedo_x + 1} , {torpedo_y + 1}")
            if world.quadrant.get_value(torpedo_x, torpedo_y) != Entity.void:
                break

        if world.quadrant.get_value(torpedo_x, torpedo_y) == Entity.klingon:
            print("*** KLINGON DESTROYED ***")
            self.world.quadrant.nb_klingons -= 1
            world.remaining_klingons -= 1
            if world.remaining_klingons <= 0:
                self.end_game(won=True, quit=False)
                return
            for klingon_ship in klingon_ships:
                if (
                    torpedo_x == klingon_ship.sector.x
                    and torpedo_y == klingon_ship.sector.y
                ):
                    klingon_ship.shield = 0
        elif world.quadrant.get_value(torpedo_x, torpedo_y) == Entity.star:
            print(f"STAR AT {torpedo_x + 1} , {torpedo_y + 1} ABSORBED TORPEDO ENERGY.")
            self.klingons_fire()
            return
        elif world.quadrant.get_value(torpedo_x, torpedo_y) == Entity.starbase:
            print("*** STARBASE DESTROYED ***")
            self.world.quadrant.nb_bases -= 1
            world.bases_in_galaxy -= 1
            if (
                world.bases_in_galaxy == 0
                and world.remaining_klingons
                <= world.stardate - world.initial_stardate - world.mission_duration
            ):
                print("THAT DOES IT, CAPTAIN!! YOU ARE HEREBY RELIEVED OF COMMAND")
                print("AND SENTENCED TO 99 STARDATES AT HARD LABOR ON CYGNUS 12!!")
                self.end_game(won=False)
                return
            print("STARFLEET COMMAND REVIEWING YOUR RECORD TO CONSIDER")
            print("COURT MARTIAL!")
            ship.docked = False

        world.quadrant.set_value(torpedo_x, torpedo_y, Entity.void)
        world.galaxy_map[ship.position.quadrant.x][
            ship.position.quadrant.y
        ] = QuadrantData(
            self.world.quadrant.nb_klingons,
            self.world.quadrant.nb_bases,
            self.world.quadrant.nb_stars,
        )
        world.charted_galaxy_map[ship.position.quadrant.x][
            ship.position.quadrant.y
        ] = world.galaxy_map[ship.position.quadrant.x][ship.position.quadrant.y]
        self.klingons_fire()

    def short_range_scan(self) -> None:
        """Print a short range scan."""
        self.world.ship.docked = False
        ship = self.world.ship
        for x in (
            ship.position.sector.x - 1,
            ship.position.sector.x,
            ship.position.sector.x + 1,
        ):
            for y in (
                ship.position.sector.y - 1,
                ship.position.sector.y,
                ship.position.sector.y + 1,
            ):
                if (
                    0 <= x <= 7
                    and 0 <= y <= 7
                    and self.world.quadrant.get_value(x, y) == Entity.starbase
                ):
                    ship.docked = True
                    cs = "DOCKED"
                    ship.refill()
                    print("SHIELDS DROPPED FOR DOCKING PURPOSES")
                    ship.shields = 0
                    break
            else:
                continue
            break
        else:
            if self.world.quadrant.nb_klingons > 0:
                cs = "*RED*"
            elif ship.energy < Ship.energy_capacity * 0.1:
                cs = "YELLOW"
            else:
                cs = "GREEN"

        if ship.damage_stats[1] < 0:
            print("\n*** SHORT RANGE SENSORS ARE OUT ***\n")
            return

        sep = "---------------------------------"
        print(sep)
        for x in range(8):
            line = ""
            for y in range(8):
                line = line + " " + self.world.quadrant.data[x][y].value

            if x == 0:
                line += f"        STARDATE           {round(int(self.world.stardate * 10) * 0.1, 1)}"
            elif x == 1:
                line += f"        CONDITION          {cs}"
            elif x == 2:
                line += f"        QUADRANT           {ship.position.quadrant}"
            elif x == 3:
                line += f"        SECTOR             {ship.position.sector}"
            elif x == 4:
                line += f"        PHOTON TORPEDOES   {int(ship.torpedoes)}"
            elif x == 5:
                line += f"        TOTAL ENERGY       {int(ship.energy + ship.shields)}"
            elif x == 6:
                line += f"        SHIELDS            {int(ship.shields)}"
            else:
                line += f"        KLINGONS REMAINING {self.world.remaining_klingons}"

            print(line)
        print(sep)

    def long_range_scan(self) -> None:
        """Print a long range scan."""
        if self.world.ship.damage_stats[2] < 0:
            print("LONG RANGE SENSORS ARE INOPERABLE")
            return

        print(f"LONG RANGE SCAN FOR QUADRANT {self.world.ship.position.quadrant}")
        print_scan_results(
            self.world.ship.position.quadrant,
            self.world.galaxy_map,
            self.world.charted_galaxy_map,
        )

    def navigation(self) -> None:
        """
        Take navigation input and move the Enterprise.

        1/8 warp goes 1 sector in the direction dirs[course]
        """
        world = self.world
        ship = world.ship

        cd = get_user_float("COURSE (1-9)? ") - 1  # Convert to 0-8
        if cd == len(dirs) - 1:
            cd = 0
        if cd < 0 or cd >= len(dirs):
            print("   LT. SULU REPORTS, 'INCORRECT COURSE DATA, SIR!'")
            return

        warp = get_user_float(
            f"WARP FACTOR (0-{'0.2' if ship.damage_stats[0] < 0 else '8'})? "
        )
        if ship.damage_stats[0] < 0 and warp > 0.2:
            print("WARP ENGINES ARE DAMAGED. MAXIMUM SPEED = WARP 0.2")
            return
        if warp == 0:
            return
        if warp < 0 or warp > 8:
            print(
                f"   CHIEF ENGINEER SCOTT REPORTS 'THE ENGINES WON'T TAKE WARP {warp}!'"
            )
            return

        warp_rounds = round(warp * 8)
        if ship.energy < warp_rounds:
            print("ENGINEERING REPORTS   'INSUFFICIENT ENERGY AVAILABLE")
            print(f"                       FOR MANEUVERING AT WARP {warp}!'")
            if ship.shields >= warp_rounds - ship.energy and ship.damage_stats[6] >= 0:
                print(
                    f"DEFLECTOR CONTROL ROOM ACKNOWLEDGES {ship.shields} UNITS OF ENERGY"
                )
                print("                         PRESENTLY DEPLOYED TO SHIELDS.")
            return

        # klingons move and fire
        for klingon_ship in self.world.quadrant.klingon_ships:
            if klingon_ship.shield != 0:
                world.quadrant.set_value(
                    klingon_ship.sector.x, klingon_ship.sector.y, Entity.void
                )
                (
                    klingon_ship.sector.x,
                    klingon_ship.sector.y,
                ) = world.quadrant.find_empty_place()
                world.quadrant.set_value(
                    klingon_ship.sector.x, klingon_ship.sector.y, Entity.klingon
                )

        self.klingons_fire()

        # repair damaged devices and print damage report
        line = ""
        for i in range(8):
            if ship.damage_stats[i] < 0:
                ship.damage_stats[i] += min(warp, 1)
                if -0.1 < ship.damage_stats[i] < 0:
                    ship.damage_stats[i] = -0.1
                elif ship.damage_stats[i] >= 0:
                    if len(line) == 0:
                        line = "DAMAGE CONTROL REPORT:"
                    line += f"   {ship.devices[i]} REPAIR COMPLETED\n"
        if len(line) > 0:
            print(line)
        if random.random() <= 0.2:
            device = fnr()
            if random.random() < 0.6:
                ship.damage_stats[device] -= random.random() * 5 + 1
                print(f"DAMAGE CONTROL REPORT:   {ship.devices[device]} DAMAGED\n")
            else:
                ship.damage_stats[device] += random.random() * 3 + 1
                print(
                    f"DAMAGE CONTROL REPORT:   {ship.devices[device]} STATE OF REPAIR IMPROVED\n"
                )

        self.move_ship(warp_rounds, cd)
        world.stardate += 0.1 * int(10 * warp) if warp < 1 else 1
        if world.has_mission_ended():
            self.end_game(won=False, quit=False)
            return

        self.short_range_scan()

    def move_ship(self, warp_rounds: int, cd: float) -> None:
        assert cd >= 0
        assert cd < len(dirs) - 1
        # cd is the course data which points to 'dirs'
        world = self.world
        ship = self.world.ship
        world.quadrant.set_value(
            int(ship.position.sector.x), int(ship.position.sector.y), Entity.void
        )
        cdi = int(cd)

        # Interpolate direction:
        dx = dirs[cdi][0] + (dirs[cdi + 1][0] - dirs[cdi][0]) * (cd - cdi)
        dy = dirs[cdi][1] + (dirs[cdi + 1][1] - dirs[cdi][1]) * (cd - cdi)

        start_quadrant = Point(ship.position.quadrant.x, ship.position.quadrant.y)
        sector_start_x: float = ship.position.sector.x
        sector_start_y: float = ship.position.sector.y

        for _ in range(warp_rounds):
            ship.position.sector.x += dx  # type: ignore
            ship.position.sector.y += dy  # type: ignore

            if (
                ship.position.sector.x < 0
                or ship.position.sector.x > 7
                or ship.position.sector.y < 0
                or ship.position.sector.y > 7
            ):
                # exceeded quadrant limits; calculate final position
                sector_start_x += ship.position.quadrant.x * 8 + warp_rounds * dx
                sector_start_y += ship.position.quadrant.y * 8 + warp_rounds * dy
                ship.position.quadrant.x = int(sector_start_x / 8)
                ship.position.quadrant.y = int(sector_start_y / 8)
                ship.position.sector.x = int(
                    sector_start_x - ship.position.quadrant.x * 8
                )
                ship.position.sector.y = int(
                    sector_start_y - ship.position.quadrant.y * 8
                )
                if ship.position.sector.x < 0:
                    ship.position.quadrant.x -= 1
                    ship.position.sector.x = 7
                if ship.position.sector.y < 0:
                    ship.position.quadrant.y -= 1
                    ship.position.sector.y = 7

                hit_edge = False
                if ship.position.quadrant.x < 0:
                    hit_edge = True
                    ship.position.quadrant.x = ship.position.sector.x = 0
                if ship.position.quadrant.x > 7:
                    hit_edge = True
                    ship.position.quadrant.x = ship.position.sector.x = 7
                if ship.position.quadrant.y < 0:
                    hit_edge = True
                    ship.position.quadrant.y = ship.position.sector.y = 0
                if ship.position.quadrant.y > 7:
                    hit_edge = True
                    ship.position.quadrant.y = ship.position.sector.y = 7
                if hit_edge:
                    print("LT. UHURA REPORTS MESSAGE FROM STARFLEET COMMAND:")
                    print("  'PERMISSION TO ATTEMPT CROSSING OF GALACTIC PERIMETER")
                    print("  IS HEREBY *DENIED*. SHUT DOWN YOUR ENGINES.'")
                    print("CHIEF ENGINEER SCOTT REPORTS  'WARP ENGINES SHUT DOWN")
                    print(
                        f"  AT SECTOR {ship.position.sector} OF "
                        f"QUADRANT {ship.position.quadrant}.'"
                    )
                    if world.has_mission_ended():
                        self.end_game(won=False, quit=False)
                        return

                stayed_in_quadrant = (
                    ship.position.quadrant.x == start_quadrant.x
                    and ship.position.quadrant.y == start_quadrant.y
                )
                if stayed_in_quadrant:
                    break
                world.stardate += 1
                ship.maneuver_energy(warp_rounds)
                self.new_quadrant()
                return
            ship_sector = self.world.ship.position.sector
            ship_x = int(ship_sector.x)
            ship_y = int(ship_sector.y)
            if self.world.quadrant.data[ship_x][ship_y] != Entity.void:
                ship_sector.x = int(ship_sector.x - dx)
                ship_sector.y = int(ship_sector.y - dy)
                print(
                    "WARP ENGINES SHUT DOWN AT SECTOR "
                    f"{ship_sector} DUE TO BAD NAVIGATION"
                )
                break
        else:
            ship.position.sector.x, ship.position.sector.y = int(
                ship.position.sector.x
            ), int(ship.position.sector.y)

        world.quadrant.set_value(
            int(ship.position.sector.x), int(ship.position.sector.y), Entity.ship
        )
        ship.maneuver_energy(warp_rounds)

    def damage_control(self) -> None:
        """Print a damage control report."""
        ship = self.world.ship

        if ship.damage_stats[5] < 0:
            print("DAMAGE CONTROL REPORT NOT AVAILABLE")
        else:
            print("\nDEVICE             STATE OF REPAIR")
            for r1 in range(8):
                print(
                    f"{ship.devices[r1].ljust(26, ' ')}{int(ship.damage_stats[r1] * 100) * 0.01:g}"
                )
            print()

        if not ship.docked:
            return

        damage_sum = sum(0.1 for i in range(8) if ship.damage_stats[i] < 0)
        if damage_sum == 0:
            return

        damage_sum += self.world.quadrant.delay_in_repairs_at_base
        if damage_sum >= 1:
            damage_sum = 0.9
        print("\nTECHNICIANS STANDING BY TO EFFECT REPAIRS TO YOUR SHIP;")
        print(
            f"ESTIMATED TIME TO REPAIR: {round(0.01 * int(100 * damage_sum), 2)} STARDATES"
        )
        if input("WILL YOU AUTHORIZE THE REPAIR ORDER (Y/N)? ").upper().strip() != "Y":
            return

        for i in range(8):
            if ship.damage_stats[i] < 0:
                ship.damage_stats[i] = 0
        self.world.stardate += damage_sum + 0.1

    def computer(self) -> None:
        """Perform the various functions of the library computer."""
        world = self.world
        ship = world.ship

        if ship.damage_stats[7] < 0:
            print("COMPUTER DISABLED")
            return

        while True:
            command = input("COMPUTER ACTIVE AND AWAITING COMMAND? ")
            if len(command) == 0:
                com = 6
            else:
                try:
                    com = int(command)
                except ValueError:
                    com = 6
            if com < 0:
                return

            print()

            if com in [0, 5]:
                if com == 5:
                    print("                        THE GALAXY")
                else:
                    print(
                        "\n        COMPUTER RECORD OF GALAXY FOR "
                        f"QUADRANT {ship.position.quadrant}\n"
                    )

                print("       1     2     3     4     5     6     7     8")
                sep = "     ----- ----- ----- ----- ----- ----- ----- -----"
                print(sep)

                for i in range(8):
                    line = " " + str(i + 1) + " "

                    if com == 5:
                        g2s = Quadrant.quadrant_name(i, 0, True)
                        line += (" " * int(12 - 0.5 * len(g2s))) + g2s
                        g2s = Quadrant.quadrant_name(i, 4, True)
                        line += (" " * int(39 - 0.5 * len(g2s) - len(line))) + g2s
                    else:
                        for j in range(8):
                            line += "   "
                            if world.charted_galaxy_map[i][j].num() == 0:
                                line += "***"
                            else:
                                line += str(
                                    world.charted_galaxy_map[i][j].num() + 1000
                                )[-3:]

                    print(line)
                    print(sep)

                print()
            elif com == 1:
                print("   STATUS REPORT:")
                print(
                    f"KLINGON{'S' if world.remaining_klingons > 1 else ''} LEFT: {world.remaining_klingons}"
                )
                print(
                    "MISSION MUST BE COMPLETED IN "
                    f"{round(0.1 * int(world.remaining_time() * 10), 1)} STARDATES"
                )

                if world.bases_in_galaxy == 0:
                    print("YOUR STUPIDITY HAS LEFT YOU ON YOUR OWN IN")
                    print("  THE GALAXY -- YOU HAVE NO STARBASES LEFT!")
                else:
                    print(
                        f"THE FEDERATION IS MAINTAINING {world.bases_in_galaxy} "
                        f"STARBASE{'S' if world.bases_in_galaxy > 1 else ''} IN THE GALAXY"
                    )

                self.damage_control()
            elif com == 2:
                if self.world.quadrant.nb_klingons <= 0:
                    print(
                        "SCIENCE OFFICER SPOCK REPORTS  'SENSORS SHOW NO ENEMY "
                        "SHIPS\n"
                        "                                IN THIS QUADRANT'"
                    )
                    return

                print(
                    f"FROM ENTERPRISE TO KLINGON BATTLE CRUISER{'S' if self.world.quadrant.nb_klingons > 1 else ''}"
                )

                for klingon_ship in self.world.quadrant.klingon_ships:
                    if klingon_ship.shield > 0:
                        print_direction(
                            Point(ship.position.sector.x, ship.position.sector.y),
                            Point(
                                int(klingon_ship.sector.x),
                                int(klingon_ship.sector.y),
                            ),
                        )
            elif com == 3:
                if self.world.quadrant.nb_bases == 0:
                    print(
                        "MR. SPOCK REPORTS,  'SENSORS SHOW NO STARBASES IN THIS "
                        "QUADRANT.'"
                    )
                    return

                print("FROM ENTERPRISE TO STARBASE:")
                print_direction(
                    Point(ship.position.sector.x, ship.position.sector.y),
                    self.world.quadrant.starbase,
                )
            elif com == 4:
                print("DIRECTION/DISTANCE CALCULATOR:")
                print(
                    f"YOU ARE AT QUADRANT {ship.position.quadrant} "
                    f"SECTOR {ship.position.sector}"
                )
                print("PLEASE ENTER")
                while True:
                    coordinates = input("  INITIAL COORDINATES (X,Y)? ").split(",")
                    if len(coordinates) == 2:
                        from1, from2 = int(coordinates[0]) - 1, int(coordinates[1]) - 1
                        if 0 <= from1 <= 7 and 0 <= from2 <= 7:
                            break
                while True:
                    coordinates = input("  FINAL COORDINATES (X,Y)? ").split(",")
                    if len(coordinates) == 2:
                        to1, to2 = int(coordinates[0]) - 1, int(coordinates[1]) - 1
                        if 0 <= to1 <= 7 and 0 <= to2 <= 7:
                            break
                print_direction(Point(from1, from2), Point(to1, to2))
            else:
                print(
                    "FUNCTIONS AVAILABLE FROM LIBRARY-COMPUTER:\n"
                    "   0 = CUMULATIVE GALACTIC RECORD\n"
                    "   1 = STATUS REPORT\n"
                    "   2 = PHOTON TORPEDO DATA\n"
                    "   3 = STARBASE NAV DATA\n"
                    "   4 = DIRECTION/DISTANCE CALCULATOR\n"
                    "   5 = GALAXY 'REGION NAME' MAP\n"
                )

    def end_game(
        self, won: bool = False, quit: bool = True, enterprise_killed: bool = False
    ) -> None:
        """Handle end-of-game situations."""
        if won:
            print("CONGRATULATIONS, CAPTAIN! THE LAST KLINGON BATTLE CRUISER")
            print("MENACING THE FEDERATION HAS BEEN DESTROYED.\n")
            print(
                f"YOUR EFFICIENCY RATING IS {round(1000 * (self.world.remaining_klingons / (self.world.stardate - self.world.initial_stardate))**2, 4)}\n\n"
            )
        else:
            if not quit:
                if enterprise_killed:
                    print(
                        "\nTHE ENTERPRISE HAS BEEN DESTROYED. THE FEDERATION "
                        "WILL BE CONQUERED."
                    )
                print(f"IT IS STARDATE {round(self.world.stardate, 1)}")

            print(
                f"THERE WERE {self.world.remaining_klingons} KLINGON BATTLE CRUISERS LEFT AT"
            )
            print("THE END OF YOUR MISSION.\n\n")

            if self.world.bases_in_galaxy == 0:
                sys.exit()

        print("THE FEDERATION IS IN NEED OF A NEW STARSHIP COMMANDER")
        print("FOR A SIMILAR MISSION -- IF THERE IS A VOLUNTEER,")
        if input("LET HIM STEP FORWARD AND ENTER 'AYE'? ").upper().strip() != "AYE":
            sys.exit()
        self.restart = True


```

这段代码定义了一个名为 `klingon_shield_strength` 的函数，其作用是计算一个名为 `Final` 的变量，该变量是一个由 8 个向量组成的数组，每个向量代表一个方向，且每个向量都有上下左右四个选项。

数组的第一个元素是一个包含上下左右四个选项的矢量，第二个元素也是一个矢量，只不过它的方向与第一个矢量的方向相反，第三个元素也是一个矢量，第四个元素同样具有上下左右四个选项，但是它的方向与第三个矢量的方向相反，以此类推。

函数定义了一个包含 8 个元素的向量数组 `dirs`，每个元素都是一个包含上下左右四个选项的矢量。这个数组的定义与函数名中的 `Final` 名称相关，因此可以推测 `klingon_shield_strength` 函数将会对 `dirs` 数组进行某种计算或修改，并返回一个值，该值将存储在 `Final` 变量中。


```
klingon_shield_strength: Final = 200
# 8 sectors = 1 quadrant
dirs: Final = [  # (down-up, left,right)
    [0, 1],  # 1: go right (same as #9)
    [-1, 1],  # 2: go up-right
    [-1, 0],  # 3: go up  (lower x-coordines; north)
    [-1, -1],  # 4: go up-left (north-west)
    [0, -1],  # 5: go left (west)
    [1, -1],  # 6: go down-left (south-west)
    [1, 0],  # 7: go down (higher x-coordines; south)
    [1, 1],  # 8: go down-right
    [0, 1],  # 9: go right (east)
]  # vectors in cardinal directions


```

这段代码定义了两个函数：fnr()和print_scan_results()。fnr()函数生成一个随机整数，范围为0到7，包括整数和浮点数。print_scan_results()函数接收三个参数：quadrant表示扫描区域的坐标，galaxy_map是一个包含四个列表的列表，每个列表包含一个二维方格中的数据，以及charted_galaxy_map一个包含四个列表的列表，每个列表包含一个二维方格中的数据。函数内部首先输出一个sep，然后循环遍历当前区域的所有位置，对于每个位置，获取其对应位置在galaxy_map和charted_galaxy_map中的数据，并输出显示。


```
def fnr() -> int:
    """Generate a random integer from 0 to 7 inclusive."""
    return random.randint(0, 7)


def print_scan_results(
    quadrant: Point,
    galaxy_map: List[List[QuadrantData]],
    charted_galaxy_map: List[List[QuadrantData]],
) -> None:
    sep = "-------------------"
    print(sep)
    for x in (quadrant.x - 1, quadrant.x, quadrant.x + 1):
        n: List[Optional[int]] = [None, None, None]

        # Reveal parts of the current map
        for y in (quadrant.y - 1, quadrant.y, quadrant.y + 1):
            if 0 <= x <= 7 and 0 <= y <= 7:
                n[y - quadrant.y + 1] = galaxy_map[x][y].num()
                charted_galaxy_map[x][y] = galaxy_map[x][y]

        line = ": "
        for line_col in n:
            if line_col is None:
                line += "*** : "
            else:
                line += str(line_col + 1000).rjust(4, " ")[-3:] + " : "
        print(line)
        print(sep)


```

这段代码定义了一个名为 `print_direction` 的函数，它接受两个参数 `source` 和 `to`，表示两个地点在网格中的坐标。函数返回 nothing，但函数内部使用了以下语句：

```python
   delta1 = -(to.x - source.x)  # flip so positive is up (heading = 3)
   delta2 = to.y - source.y

   if delta2 > 0:  # bug in original; no check for divide by 0
       if delta1 > 0:
           base = 7
       else:
           base = 1
           delta1, delta2 = delta2, delta1

   delta1, delta2 = abs(delta1), abs(delta2)

   if delta1 > 0 or delta2 > 0:
       if delta1 >= delta2:
           print(f"DIRECTION = {round(base + delta2 / delta1, 6)}")
       else:
           print(f"DIRECTION = {round(base + 2 - delta1 / delta2, 6)}")
```

这段代码的作用是打印出从 `source` 点到 `to` 点的方向和两点的距离，其中距离使用欧几里得距离公式计算。在函数内部，首先计算了 `delta1` 和 `delta2`，然后判断它们的大小关系。如果 `delta2` 大于 `delta1`，则打印方向为向上，距离为 `base`。否则，向上方向相反，距离为 `base`。

此外，代码还计算了方向和距离，并返回它们。


```
def print_direction(source: Point, to: Point) -> None:
    """Print direction and distance between two locations in the grid."""
    delta1 = -(to.x - source.x)  # flip so positive is up (heading = 3)
    delta2 = to.y - source.y

    if delta2 > 0:
        if delta1 < 0:
            base = 7
        else:
            base = 1
            delta1, delta2 = delta2, delta1
    else:
        if delta1 > 0:
            base = 3
        else:
            base = 5
            delta1, delta2 = delta2, delta1

    delta1, delta2 = abs(delta1), abs(delta2)

    if delta1 > 0 or delta2 > 0:  # bug in original; no check for divide by 0
        if delta1 >= delta2:
            print(f"DIRECTION = {round(base + delta2 / delta1, 6)}")
        else:
            print(f"DIRECTION = {round(base + 2 - delta1 / delta2, 6)}")

    print(f"DISTANCE = {round(sqrt(delta1 ** 2 + delta2 ** 2), 6)}")


```

It looks like you've written a Python program that simulates a game of的时代， where you control a spaceship with different weapons and abilities, and the other player controls the AI computer. The program has several commands that allow you to navigate, fire weapons, and raise/lower shields.

The game starts with the spaceship in a random position, and the以及其他玩家的控制滑行控制，如果没有对任何武器进行激活，则使用默认武器进行攻击。程序使用开始游戏，并在新的 quadrant 中显示所有可用的控件。

然后，程序进入一个无限循环，每轮程序显示所有玩家的控件，并提示输入命令以控制游戏。根据玩家输入的命令，程序会调用相应的函数来执行操作。

如果玩家输入的命令不在可用的控件中，则程序会显示一个错误消息并重新显示所有控件。如果玩家输入的命令是“XXX”则游戏结束。

程序在每次循环结束后，将调用函数来检查所有玩家的控件，并确保所有控件都准备好进行下一步操作。然后，程序将调用新的游戏循环，以继续模拟游戏。


```
def main() -> None:
    game = Game()
    world = game.world
    ship = world.ship

    f: Dict[str, Callable[[], None]] = {
        "NAV": game.navigation,
        "SRS": game.short_range_scan,
        "LRS": game.long_range_scan,
        "PHA": game.phaser_control,
        "TOR": game.photon_torpedoes,
        "SHE": ship.shield_control,
        "DAM": game.damage_control,
        "COM": game.computer,
        "XXX": game.end_game,
    }

    while True:
        game.startup()
        game.new_quadrant()
        restart = False

        while not restart:
            if ship.shields + ship.energy <= 10 or (
                ship.energy <= 10 and ship.damage_stats[6] != 0
            ):
                print(
                    "\n** FATAL ERROR **   YOU'VE JUST STRANDED YOUR SHIP "
                    "IN SPACE.\nYOU HAVE INSUFFICIENT MANEUVERING ENERGY, "
                    "AND SHIELD CONTROL\nIS PRESENTLY INCAPABLE OF CROSS-"
                    "CIRCUITING TO ENGINE ROOM!!"
                )

            command = input("COMMAND? ").upper().strip()

            if command in f:
                f[command]()
            else:
                print(
                    "ENTER ONE OF THE FOLLOWING:\n"
                    "  NAV  (TO SET COURSE)\n"
                    "  SRS  (FOR SHORT RANGE SENSOR SCAN)\n"
                    "  LRS  (FOR LONG RANGE SENSOR SCAN)\n"
                    "  PHA  (TO FIRE PHASERS)\n"
                    "  TOR  (TO FIRE PHOTON TORPEDOES)\n"
                    "  SHE  (TO RAISE OR LOWER SHIELDS)\n"
                    "  DAM  (FOR DAMAGE CONTROL REPORTS)\n"
                    "  COM  (TO CALL ON LIBRARY-COMPUTER)\n"
                    "  XXX  (TO RESIGN YOUR COMMAND)\n"
                )


```

这段代码是一个if语句，它会判断当前脚本是否是Python的主程序入口(__main__)。如果当前脚本被运行，即被当做主程序入口，那么程序会执行if语句中的代码块。

if语句块中的代码是一个函数main()，它可能是用来定义程序的函数，或者是用来执行一些设置或配置的代码。因为这个函数是包含在if语句中的，所以它的作用就是在程序作为主程序入口时执行。当程序作为主程序入口时，程序会首先执行if语句块中的代码，如果这个代码块中有任何错误，那么程序就会崩溃。


```
if __name__ == "__main__":
    main()

```