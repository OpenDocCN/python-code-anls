# `basic-computer-games\13_Bounce\javascript\bounce.js`

```py
// BOUNCE
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入元素类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入元素添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入元素获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "BOUNCE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化数组
    ta = [];
    // 打印提示信息
    print("THIS SIMULATION LETS YOU SPECIFY THE INITIAL VELOCITY\n");
    print("OF A BALL THROWN STRAIGHT UP, AND THE COEFFICIENT OF\n");
    print("ELASTICITY OF THE BALL.  PLEASE USE A DECIMAL FRACTION\n");
    # 打印提示信息：效率小于1
    print("COEFFICIENCY (LESS THAN 1).\n");
    # 打印空行
    print("\n");
    # 打印提示信息：指定时间增量用于“闪烁”球的飞行（初始尝试0.1）
    print("YOU ALSO SPECIFY THE TIME INCREMENT TO BE USED IN\n");
    # 打印空行
    print("\n");
    while (1) {
        # 打印提示信息：时间增量（秒）
        print("TIME INCREMENT (SEC)");
        # 将输入的字符串转换为浮点数
        s2 = parseFloat(await input());
        # 打印空行
        print("\n");
        # 打印提示信息：速度（FPS）
        print("VELOCITY (FPS)");
        # 将输入的字符串转换为浮点数
        v = parseFloat(await input());
        # 打印空行
        print("\n");
        # 打印提示信息：系数
        print("COEFFICIENT");
        # 将输入的字符串转换为浮点数
        c = parseFloat(await input());
        # 打印空行
        print("\n");
        # 打印提示信息：英尺
        print("FEET\n");
        # 打印空行
        print("\n");
        # 计算球飞行的总时间
        s1 = Math.floor(70 / (v / (16 * s2)));
        # 计算球在每个时间增量内的速度
        for (i = 1; i <= s1; i++)
            ta[i] = v * Math.pow(c, i - 1) / 16;
        # 循环计算并打印球的轨迹
        for (h = Math.floor(-16 * Math.pow(v / 32, 2) + Math.pow(v, 2) / 32 + 0.5); h >= 0; h -= 0.5) {
            str = "";
            # 如果高度是整数，则在字符串中添加高度值
            if (Math.floor(h) == h)
                str += " " + h + " ";
            l = 0;
            # 计算每个时间增量内的球的位置，并打印轨迹
            for (i = 1; i <= s1; i++) {
                for (t = 0; t <= ta[i]; t += s2) {
                    l += s2;
                    if (Math.abs(h - (0.5 * (-32) * Math.pow(t, 2) + v * Math.pow(c, i - 1) * t)) <= 0.25) {
                        while (str.length < l / s2)
                            str += " ";
                        str += "0";
                    }
                }
                t = ta[i + 1] / 2;
                if (-16 * Math.pow(t, 2) + v * Math.pow(c, i - 1) * t < h)
                    break;
            }
            # 打印球的轨迹
            print(str + "\n");
        }
        # 打印球的轨迹
        str = " ";
        for (i = 1; i < Math.floor(l + 1) / s2 + 1; i++)
            str += ".";
        print(str + "\n");
        # 打印球的轨迹
        str = " 0";
        for (i = 1; i < Math.floor(l + 0.9995); i++) {
            while (str.length < Math.floor(i / s2))
                str += " ";
            str += i;
        }
        print(str + "\n");
        # 打印球飞行的总时间
        print(tab(Math.floor(l + 1) / (2 * s2) - 2) + "SECONDS\n");
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```