# `d:/src/tocomm/basic-computer-games\86_Target\java\Target.java`

```
import java.util.Scanner; // 导入 Scanner 类，用于从控制台读取输入

/**
 * TARGET
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
public class Target {

	private static final double RADIAN = 180 / Math.PI; // 定义一个常量 RADIAN，用于将角度转换为弧度

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in); // 创建一个 Scanner 对象，用于从控制台读取输入

		printIntro(); // 调用 printIntro() 方法，打印游戏介绍

		//continue till the user aborts
		while (true) { // 进入无限循环，直到用户中止游戏
			int numberShots = 0; // 初始化射击次数为 0
			// 生成一个 0 到 2π 之间的随机数，表示与 X 轴的夹角
			final double xAxisInRadians = Math.random() * 2 * Math.PI;
			// 生成一个 0 到 2π 之间的随机数，表示与 Z 轴的夹角
			final double yAxisInRadians = Math.random() * 2 * Math.PI;
			// 打印 X 轴和 Z 轴的夹角
			System.out.printf("RADIANS FROM X AXIS = %.7f     FROM Z AXIS = %.7f\n", xAxisInRadians, yAxisInRadians);

			// 生成一个随机数，表示目标到自己的距离
			final double p1 = 100000 * Math.random() + Math.random();
			// 根据夹角和距离计算目标的坐标
			final double x = Math.sin(yAxisInRadians) * Math.cos(xAxisInRadians) * p1;
			final double y = Math.sin(yAxisInRadians) * Math.sin(xAxisInRadians) * p1;
			final double z = Math.cos(yAxisInRadians) * p1;
			// 打印目标的坐标
			System.out.printf("TARGET SIGHTED: APPROXIMATE COORDINATES:  X=%.3f  Y=%.3f  Z=%.3f\n", x, y, z);
			// 初始化目标或自身是否被摧毁的标志
			boolean targetOrSelfDestroyed = false;
			// 循环直到目标或自身被摧毁
			while (!targetOrSelfDestroyed) {
				// 记录射击次数
				numberShots++;
				// 估计目标距离
				int estimatedDistance = 0;
				// 根据射击次数选择不同的估计距离计算方法
				switch (numberShots) {
					case 1:
						estimatedDistance = (int) (p1 * .05) * 20;
						break;
					case 2:
						estimatedDistance = (int) (p1 * .1) * 10;
						break;
				case 3:
					# 根据不同情况计算估计距离
					estimatedDistance = (int) (p1 * .5) * 2;
					break;
				case 4:
				case 5:
					# 根据不同情况计算估计距离
					estimatedDistance = (int) (p1);
					break;
			}

			# 打印估计距离
			System.out.printf("     ESTIMATED DISTANCE: %s\n\n", estimatedDistance);

			# 读取输入的目标尝试
			final TargetAttempt targetAttempt = readInput(scan);
			# 如果距离小于20，输出信息并设置目标或自身被摧毁的标志为真
			if (targetAttempt.distance < 20) {
				System.out.println("YOU BLEW YOURSELF UP!!");
				targetOrSelfDestroyed = true;
			} else {
				# 计算偏离X轴和Z轴的弧度
				final double a1 = targetAttempt.xDeviation / RADIAN;
				final double b1 = targetAttempt.zDeviation / RADIAN;
				System.out.printf("RADIANS FROM X AXIS = %.7f  FROM Z AXIS = %.7f\n", a1, b1);
					// 计算目标点与射击点之间的三维距离
					final double x1 = targetAttempt.distance * Math.sin(b1) * Math.cos(a1);
					final double y1 = targetAttempt.distance * Math.sin(b1) * Math.sin(a1);
					final double z1 = targetAttempt.distance * Math.cos(b1);

					double distance = Math.sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y) + (z1 - z) * (z1 - z));
					// 如果距离大于20千米
					if (distance > 20) {
						double X2 = x1 - x;
						double Y2 = y1 - y;
						double Z2 = z1 - z;
						// 如果 X2 小于0，表示射击点在目标点后方
						if (X2 < 0) {
							System.out.printf("SHOT BEHIND TARGET %.7f KILOMETERS.\n", -X2);
						} else {
							// 否则表示射击点在目标点前方
							System.out.printf("SHOT IN FRONT OF TARGET %.7f KILOMETERS.\n", X2);
						}
						// 如果 Y2 小于0，表示射击点在目标点右侧
						if (Y2 < 0) {
							System.out.printf("SHOT TO RIGHT OF TARGET %.7f KILOMETERS.\n", -Y2);
						} else {
							// 否则表示射击点在目标点左侧
							System.out.printf("SHOT TO LEFT OF TARGET %.7f KILOMETERS.\n", Y2);
						}
						// 如果 Z2 小于0，表示射击点在目标点下方
						System.out.printf("SHOT BELOW TARGET %.7f KILOMETERS.\n", -Z2); // 打印距离目标下方的距离
						} else {
							System.out.printf("SHOT ABOVE TARGET %.7f KILOMETERS.\n", Z2); // 打印距离目标上方的距离
						}
						System.out.printf("APPROX POSITION OF EXPLOSION:  X=%.7f   Y=%.7f   Z=%.7f\n", x1, y1, z1); // 打印爆炸的大致位置坐标
						System.out.printf("     DISTANCE FROM TARGET =%.7f\n\n\n\n", distance); // 打印距离目标的距离
					} else {
						System.out.println(" * * * HIT * * *   TARGET IS NON-FUNCTIONAL"); // 打印击中目标但目标已失效的提示
						System.out.printf("DISTANCE OF EXPLOSION FROM TARGET WAS %.5f KILOMETERS.\n", distance); // 打印爆炸距离目标的距离
						System.out.printf("MISSION ACCOMPLISHED IN %s SHOTS.\n", numberShots); // 打印完成任务所需的射击次数
						targetOrSelfDestroyed = true; // 将目标或自身摧毁的标志设为true
					}
				}
			}
			System.out.println("\n\n\n\n\nNEXT TARGET...\n"); // 打印下一个目标的提示
		}
	}

	private static TargetAttempt readInput(Scanner scan) {
		System.out.println("INPUT ANGLE DEVIATION FROM X, DEVIATION FROM Z, DISTANCE "); // 打印输入角度偏差和距离的提示
		# 初始化一个布尔变量，用于判断输入是否有效
		boolean validInput = false;
		# 创建一个 TargetAttempt 对象
		TargetAttempt targetAttempt = new TargetAttempt();
		# 循环直到输入有效
		while (!validInput) {
			# 从控制台读取输入
			String input = scan.nextLine();
			# 将输入按逗号分割成数组
			final String[] split = input.split(",");
			# 尝试将分割后的字符串转换成浮点数，并赋值给 targetAttempt 对象的属性
			try {
				targetAttempt.xDeviation = Float.parseFloat(split[0]);
				targetAttempt.zDeviation = Float.parseFloat(split[1]);
				targetAttempt.distance = Float.parseFloat(split[2]);
				# 输入有效，设置 validInput 为 true
				validInput = true;
			} catch (NumberFormatException nfe) {
				# 捕获异常，提示用户重新输入
				System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n? ");
			}

		}
		# 返回 targetAttempt 对象
		return targetAttempt;
	}

	# 打印介绍信息
	private static void printIntro() {
		System.out.println("                                TARGET");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		# 打印创意计算公司的信息
		System.out.println("\n\n");
		# 打印空行
		System.out.println("YOU ARE THE WEAPONS OFFICER ON THE STARSHIP ENTERPRISE");
		# 打印角色信息
		System.out.println("AND THIS IS A TEST TO SEE HOW ACCURATE A SHOT YOU");
		# 打印测试目的
		System.out.println("ARE IN A THREE-DIMENSIONAL RANGE.  YOU WILL BE TOLD");
		# 打印测试说明
		System.out.println("THE RADIAN OFFSET FOR THE X AND Z AXES, THE LOCATION");
		# 打印坐标轴偏移信息
		System.out.println("OF THE TARGET IN THREE DIMENSIONAL RECTANGULAR COORDINATES,");
		# 打印目标坐标信息
		System.out.println("THE APPROXIMATE NUMBER OF DEGREES FROM THE X AND Z");
		# 打印角度信息
		System.out.println("AXES, AND THE APPROXIMATE DISTANCE TO THE TARGET.");
		# 打印距离信息
		System.out.println("YOU WILL THEN PROCEED TO SHOOT AT THE TARGET UNTIL IT IS");
		# 打印测试流程
		System.out.println("DESTROYED!");
		# 打印测试目标
		System.out.println("\nGOOD LUCK!!\n\n");
		# 打印祝福信息
	}

	/**
	 * Represents the user input
	 */
	private static class TargetAttempt {
		# 表示用户输入的类

		double xDeviation;
		# x轴偏移量
		# 定义一个双精度浮点型变量 zDeviation
		double zDeviation;
		# 定义一个双精度浮点型变量 distance
		double distance;
	}
}
```