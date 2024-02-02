# `basic-computer-games\84_Super_Star_Trek\java\Enterprise.java`

```py
# 导入 java.util.stream.IntStream 类
import java.util.stream.IntStream;

# 定义 Enterprise 类，代表星际飞船企业号
public class Enterprise {

    # 定义飞船坐标的常量
    public static final int COORD_X = 0;
    public static final int COORD_Y = 1;

    # 设备常量
    static final int DEVICE_WARP_ENGINES = 1;
    static final int DEVICE_SHORT_RANGE_SENSORS = 2;
    static final int DEVICE_LONG_RANGE_SENSORS = 3;
    static final int DEVICE_PHASER_CONTROL = 4;
    static final int DEVICE_PHOTON_TUBES = 5;
    static final int DEVICE_DAMAGE_CONTROL = 6;
    static final int DEVICE_SHIELD_CONTROL = 7;
    static final int DEVICE_LIBRARY_COMPUTER = 8;
    final double[] deviceStatus = new double[9];   # 8 个设备的损坏状态

    # 位置
    final int[][] cardinalDirections = new int[10][3];   # 9 个基本方向的 2 维向量
    int quadrantX;  # 象限 X 坐标
    int quadrantY;  # 象限 Y 坐标
    int sectorX;    # 扇区 X 坐标
    int sectorY;    # 扇区 Y 坐标

    # 飞船状态
    boolean docked = false;  # 是否停靠
    int energy = 3000;       # 能量
    int torpedoes = 10;      # 鱼雷数量
    int shields = 0;         # 护盾能量
    double repairCost;       # 维修成本

    final int initialEnergy = energy;    # 初始能量
    final int initialTorpedoes = torpedoes;  # 初始鱼雷数量
    // 无参构造函数，初始化企业飞船
    public Enterprise() {
        // 随机初始化位置
        this.setQuadrant(new int[]{ Util.fnr(), Util.fnr() });
        this.setSector(new int[]{ Util.fnr(), Util.fnr() });
        // 初始化基本方向
        IntStream.range(1, 9).forEach(i -> {
            cardinalDirections[i][1] = 0;
            cardinalDirections[i][2] = 0;
        });
        cardinalDirections[3][1] = -1;
        cardinalDirections[2][1] = -1;
        cardinalDirections[4][1] = -1;
        cardinalDirections[4][2] = -1;
        cardinalDirections[5][2] = -1;
        cardinalDirections[6][2] = -1;
        cardinalDirections[1][2] = 1;
        cardinalDirections[2][2] = 1;
        cardinalDirections[6][1] = 1;
        cardinalDirections[7][1] = 1;
        cardinalDirections[8][1] = 1;
        cardinalDirections[8][2] = 1;
        cardinalDirections[9][2] = 1;
        // 初始化设备状态
        IntStream.range(1, 8).forEach(i -> deviceStatus[i] = 0);
    }

    // 获取护盾能量
    public int getShields() {
        return shields;
    }

    /**
     * 企业飞船受到敌人攻击
     * @param hits 攻击点数
     */
    public void sufferHitPoints(int hits) {
        this.shields = shields - hits;
    }

    // 获取能量
    public int getEnergy() {
        return energy;
    }

    // 补充能量和鱼雷
    public void replenishSupplies() {
        this.energy = this.initialEnergy;
        this.torpedoes = this.initialTorpedoes;
    }

    // 减少能量
    public void decreaseEnergy(final double amount) {
        this.energy -= amount;
    }

    // 减少鱼雷数量
    public void decreaseTorpedoes(final int amount) {
        torpedoes -= amount;
    }

    // 关闭护盾
    public void dropShields() {
        this.shields = 0;
    }

    // 获取总能量
    public int getTotalEnergy() {
        return (shields + energy);
    }

    // 获取初始能量
    public int getInitialEnergy() {
        return initialEnergy;
    }

    // 获取鱼雷数量
    public int getTorpedoes() {
        return torpedoes;
    }

    // 获取设备状态
    public double[] getDeviceStatus() {
        return deviceStatus;
    }
    // 返回基本方向数组
    public int[][] getCardinalDirections() {
        return cardinalDirections;
    }

    // 设置设备状态
    public void setDeviceStatus(final int device, final double status) {
        this.deviceStatus[device] = status;
    }

    // 返回是否停靠
    public boolean isDocked() {
        return docked;
    }

    // 设置停靠状态
    public void setDocked(boolean docked) {
        this.docked = docked;
    }

    // 返回象限坐标
    public int[] getQuadrant() {
        return new int[] {quadrantX, quadrantY};
    }

    // 设置象限坐标
    public void setQuadrant(final int[] quadrant) {
        this.quadrantX = quadrant[COORD_X];
        this.quadrantY = quadrant[COORD_Y];
    }

    // 返回扇区坐标
    public int[] getSector() {
        return new int[] {sectorX, sectorY};
    }

    // 设置扇区坐标
    public void setSector(final int[] sector) {
        this.sectorX = sector[COORD_X];
        this.sectorY = sector[COORD_Y];
    }

    // 随机设置修复成本
    void randomRepairCost() {
        repairCost = .5 * Util.random();
    }

    // 修复受损设备
    public void repairDamagedDevices(final float warp) {
        // 修复受损设备并打印损坏报告
        for (int i = 1; i <= 8; i++) {
            if (deviceStatus[i] < 0) {
                deviceStatus[i] += Math.min(warp, 1);
                if ((deviceStatus[i] > -.1) && (deviceStatus[i] < 0)) {
                    deviceStatus[i] = -.1;
                    break;
                } else if (deviceStatus[i] >= 0) {
                    Util.println("DAMAGE CONTROL REPORT:  ");
                    Util.println(Util.tab(8) + printDeviceName(i) + " REPAIR COMPLETED.");
                }
            }
        }
    }

    // 操作能量SR
    public void maneuverEnergySR(final int N) {
        energy = energy - N - 10;
        if (energy >= 0) return;
        Util.println("SHIELD CONTROL SUPPLIES ENERGY TO COMPLETE THE MANEUVER.");
        shields = shields + energy;
        energy = 0;
        if (shields <= 0) shields = 0;
    }
    // 控制护盾的方法
    void shieldControl() {
        // 如果护盾控制设备状态小于0，输出信息并返回
        if (deviceStatus[DEVICE_SHIELD_CONTROL] < 0) {
            Util.println("SHIELD CONTROL INOPERABLE");
            return;
        }
        // 输出能量和护盾可用数量
        Util.println("ENERGY AVAILABLE = " + (energy + shields));
        // 获取输入的能量转换为护盾的数量
        int energyToShields = Util.toInt(Util.inputFloat("NUMBER OF UNITS TO SHIELDS"));
        // 如果输入的能量小于0或者与当前护盾数量相等，输出信息并返回
        if (energyToShields < 0 || shields == energyToShields) {
            Util.println("<SHIELDS UNCHANGED>");
            return;
        }
        // 如果输入的能量大于总能量，输出信息并返回
        if (energyToShields > energy + energyToShields) {
            Util.println("SHIELD CONTROL REPORTS  'THIS IS NOT THE FEDERATION TREASURY.'");
            Util.println("<SHIELDS UNCHANGED>");
            return;
        }
        // 更新能量和护盾的数量
        energy = energy + shields - energyToShields;
        shields = energyToShields;
        // 输出护盾控制室的报告
        Util.println("DEFLECTOR CONTROL ROOM REPORT:");
        Util.println("  'SHIELDS NOW AT " + Util.toInt(shields) + " UNITS PER YOUR COMMAND.'");
    }
    // 对设备进行损坏控制
    void damageControl(GameCallback callback) {
        // 如果设备状态小于0，则输出报告不可用
        if (deviceStatus[DEVICE_DAMAGE_CONTROL] < 0) {
            Util.println("DAMAGE CONTROL REPORT NOT AVAILABLE");
        } else {
            // 输出设备状态修复情况表
            Util.println("\nDEVICE             STATE OF REPAIR");
            for (int deviceNr = 1; deviceNr <= 8; deviceNr++) {
                // 输出设备名称和修复状态
                Util.print(printDeviceName(deviceNr) + Util.leftStr(GalaxyMap.QUADRANT_ROW, 25 - Util.strlen(printDeviceName(deviceNr))) + " " + Util.toInt(deviceStatus[deviceNr] * 100) * .01 + "\n");
            }
        }
        // 如果未停靠，则直接返回
        if (!docked) return;

        // 计算需要修复的设备数量
        double deltaToRepair = 0;
        for (int i = 1; i <= 8; i++) {
            if (deviceStatus[i] < 0) deltaToRepair += .1;
        }
        // 如果有需要修复的设备
        if (deltaToRepair > 0) {
            // 计算修复成本
            deltaToRepair += repairCost;
            if (deltaToRepair >= 1) deltaToRepair = .9;
            // 输出修复信息和预计修复时间
            Util.println("TECHNICIANS STANDING BY TO EFFECT REPAIRS TO YOUR SHIP;");
            Util.println("ESTIMATED TIME TO REPAIR:'" + .01 * Util.toInt(100 * deltaToRepair) + " STARDATES");
            // 接受用户输入，是否授权修复
            final String reply = Util.inputStr("WILL YOU AUTHORIZE THE REPAIR ORDER (Y/N)");
            if ("Y".equals(reply)) {
                // 修复设备状态，并增加星日期
                for (int deviceNr = 1; deviceNr <= 8; deviceNr++) {
                    if (deviceStatus[deviceNr] < 0) deviceStatus[deviceNr] = 0;
                }
                callback.incrementStardate(deltaToRepair + .1);
            }
        }
    }
    // 定义一个静态方法，用于打印设备名称，参数为设备编号
    public static String printDeviceName(final int deviceNumber) {  // 8790
        // 根据设备编号进行判断
        switch (deviceNumber) {
            case DEVICE_WARP_ENGINES:
                return "WARP ENGINES";  // 如果设备编号为WARP_ENGINES，则返回"WARP ENGINES"
            case DEVICE_SHORT_RANGE_SENSORS:
                return "SHORT RANGE SENSORS";  // 如果设备编号为SHORT_RANGE_SENSORS，则返回"SHORT RANGE SENSORS"
            case DEVICE_LONG_RANGE_SENSORS:
                return "LONG RANGE SENSORS";  // 如果设备编号为LONG_RANGE_SENSORS，则返回"LONG RANGE SENSORS"
            case DEVICE_PHASER_CONTROL:
                return "PHASER CONTROL";  // 如果设备编号为PHASER_CONTROL，则返回"PHASER CONTROL"
            case DEVICE_PHOTON_TUBES:
                return "PHOTON TUBES";  // 如果设备编号为PHOTON_TUBES，则返回"PHOTON TUBES"
            case DEVICE_DAMAGE_CONTROL:
                return "DAMAGE CONTROL";  // 如果设备编号为DAMAGE_CONTROL，则返回"DAMAGE CONTROL"
            case DEVICE_SHIELD_CONTROL:
                return "SHIELD CONTROL";  // 如果设备编号为SHIELD_CONTROL，则返回"SHIELD CONTROL"
            case DEVICE_LIBRARY_COMPUTER:
                return "LIBRARY-COMPUTER";  // 如果设备编号为LIBRARY_COMPUTER，则返回"LIBRARY-COMPUTER"
        }
        return "";  // 如果设备编号不匹配任何情况，则返回空字符串
    }
# 闭合前面的函数定义
```