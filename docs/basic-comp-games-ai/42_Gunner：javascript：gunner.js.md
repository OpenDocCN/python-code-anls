# `42_Gunner\javascript\gunner.js`

```
// 创建一个新的 Promise 对象，用于处理输入操作
// 创建一个 INPUT 元素，用于用户输入
// 在页面上显示提示符 "? "
// 设置 INPUT 元素的类型为文本输入
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回处理后的字符串

print(tab(30) + "GUNNER\n");  # 打印以30个空格开头的字符串 "GUNNER"
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印以15个空格开头的字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
print("\n");  # 打印空行
print("\n");  # 打印空行
print("\n");  # 打印空行
print("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN\n");  # 打印字符串 "YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN"
print("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE\n");  # 打印字符串 "CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE"
print("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS\n");  # 打印字符串 "WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS"
print("OF THE TARGET WILL DESTROY IT.\n");  # 打印字符串 "OF THE TARGET WILL DESTROY IT."
print("\n");  # 打印空行

// Main control section
async function main()  # 定义异步函数 main
{
    while (1) {  # 进入无限循环
        r = Math.floor(40000 * Math.random() + 20000);  # 生成一个20000到60000之间的随机整数并赋值给变量 r
        # 打印枪的最大射程
        print("MAXIMUM RANGE OF YOUR GUN IS " + r + " YARDS.\n");
        # 初始化变量 z
        z = 0;
        # 打印换行
        print("\n");
        # 初始化变量 s1
        s1 = 0;
        # 进入无限循环
        while (1) {
            # 生成目标距离 t，取值范围为 0.1 到 0.9 乘以枪的射程 r
            t = Math.floor(r * (0.1 + 0.8 * Math.random()));
            # 初始化变量 s
            s = 0;
            # 打印目标距离
            print("DISTANCE TO THE TARGET IS " + t + " YARDS.\n");
            # 打印换行
            print("\n");

            # 进入内部无限循环
            while (1) {
                # 打印换行
                print("\n");
                # 打印提示信息
                print("ELEVATION");
                # 将用户输入的值转换为浮点数并赋给变量 b
                b = parseFloat(await input());
                # 如果输入值大于 89，则打印最大仰角为 89 度
                if (b > 89) {
                    print("MAXIMUM ELEVATION IS 89 DEGREES.\n");
                    # 继续内部循环
                    continue;
                }
                # 如果输入值小于 1，则打印最小仰角为 1 度
                if (b < 1) {
                    print("MINIMUM ELEVATION IS ONE DEGREE.\n");
                    continue;  # 继续执行下一次循环
                }
                if (++s >= 6) {  # 如果 s 大于等于 6
                    print("\n");  # 打印换行
                    print("BOOM !!!!   YOU HAVE JUST BEEN DESTROYED BY THE ENEMY.\n");  # 打印提示信息
                    print("\n");  # 打印换行
                    print("\n");  # 打印换行
                    print("\n");  # 打印换行
                    e = 0;  # 将 e 设为 0
                    break;  # 跳出循环
                }
                b2 = 2 * b / 57.3;  # 计算 b2 的值
                i = r * Math.sin(b2);  # 计算 i 的值
                x = t - i;  # 计算 x 的值
                e = Math.floor(x);  # 将 x 向下取整赋值给 e
                if (true) {  # 如果条件为真
                    e = 1;  # 将 e 设为 1
                    break;  # 跳出循环
                }
                if (e > 100) {  # 如果 e 大于 100
# 如果 e 大于 0，表示目标距离不足，打印距离不足的信息
if (e > 0) {
    print("SHORT OF TARGET BY " + Math.abs(e) + " YARDS.\n");
} else {
    # 如果 e 小于 0，表示目标距离过远，打印距离过远的信息
    print("OVER TARGET BY " + Math.abs(e) + " YARDS.\n");
}
# 如果 e 等于 1，表示目标被摧毁，打印目标被摧毁的信息，并累加击中的回合数
if (e == 1) {
    print("*** TARGET DESTROYED *** " + s + " ROUNDS OF AMMUNITION EXPENDED.\n");
    s1 += s;
    # 如果 z 等于 4，表示击中回合数达到上限，打印总共击中回合数并结束循环
    if (z == 4) {
        print("\n");
        print("\n");
        print("TOTAL ROUND EXPENDED WERE: " + s1 + "\n");
        break;
    } else {
        # 如果 z 不等于 4，增加 z 的值，打印新的敌人活动信息
        z++;
        print("\n");
        print("THE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY...\n");
    }
} else {
    # 如果 e 不等于 1，将 s1 设为 19
    s1 = 19;
}
                break;  # 结束当前循环，跳出循环体
            }
        }
        if (s1 > 18) {  # 如果s1大于18
            print("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!\n");  # 打印提示信息
        } else {  # 否则
            print("NICE SHOOTING !!");  # 打印提示信息
        }
        print("\n");  # 打印空行
        print("TRY AGAIN (Y OR N)");  # 打印提示信息
        str = await input();  # 获取用户输入的字符串
        if (str.substr(0, 1) != "Y")  # 如果用户输入的字符串的第一个字符不是Y
            break;  # 结束当前循环，跳出循环体
    }
    print("\n");  # 打印空行
    print("OK.  RETURN TO BASE CAMP.\n");  # 打印提示信息
}

main();  # 调用main函数
```