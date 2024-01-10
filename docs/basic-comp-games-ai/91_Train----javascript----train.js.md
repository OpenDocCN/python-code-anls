# `basic-computer-games\91_Train\javascript\train.js`

```
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
                                                      // 从输出元素中移除输入元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析 Promise 对象
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

// 主控制部分，使用 async 关键字定义异步函数
async function main()
{
    // 打印标题
    print(tab(33) + "TRAIN\n");
    // 打印副标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印多个空行
    print("\n");
    print("\n");
    print("\n");
    // 打印题目
    print("TIME - SPEED DISTANCE EXERCISE\n");
    // 打印空格
    print("\n ");
}
    # 进入无限循环，直到条件不满足时退出
    while (1) {
        # 生成一个40-65之间的随机数，表示汽车的速度
        c = Math.floor(25 * Math.random()) + 40;
        # 生成一个5-20之间的随机数，表示火车的速度
        d = Math.floor(15 * Math.random()) + 5;
        # 生成一个20-39之间的随机数，表示行程时间
        t = Math.floor(19 * Math.random()) + 20;
        # 打印汽车以c速度行驶可以比火车快d小时到达目的地
        print(" A CAR TRAVELING " + c + " MPH CAN MAKE A CERTAIN TRIP IN\n");
        print(d + " HOURS LESS THAN A TRAIN TRAVELING AT " + t + " MPH.\n");
        # 打印提示信息，等待用户输入汽车行程时间
        print("HOW LONG DOES THE TRIP TAKE BY CAR");
        a = parseFloat(await input());
        # 根据公式计算出正确的行程时间
        v = d * t / (c - t);
        # 计算用户输入的行程时间与正确行程时间的误差百分比
        e = Math.floor(Math.abs((v - a) * 100 / a) + 0.5);
        # 如果误差大于5%，打印错误提示信息
        if (e > 5) {
            print("SORRY.  YOU WERE OFF BY " + e + " PERCENT.\n");
        } else {
            # 如果误差小于等于5%，打印正确提示信息
            print("GOOD! ANSWER WITHIN " + e + " PERCENT.\n");
        }
        # 打印正确的行程时间
        print("CORRECT ANSWER IS " + v + " HOURS.\n");
        # 打印空行
        print("\n");
        # 提示用户是否继续下一个问题
        print("ANOTHER PROBLEM (YES OR NO)\n");
        str = await input();
        # 打印空行
        print("\n");
        # 如果用户输入的第一个字符不是Y，则退出循环
        if (str.substr(0, 1) != "Y")
            break;
    }
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```