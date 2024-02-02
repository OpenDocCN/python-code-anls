# `basic-computer-games\68_Orbit\javascript\orbit.js`

```py
// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，移除输入元素，打印输入的字符串，并解析 Promise
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

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个空数组
var a = [];

// 主程序，使用 async 关键字定义一个异步函数
async function main()
{
    // 打印标题
    print(tab(33) + "ORBIT\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP.\n");
    print("\n");
    print("THE SHIP IS IN A CONSTANT POLAR ORBIT.  ITS\n");
    print("DISTANCE FROM THE CENTER OF YOUR PLANET IS FROM\n");
    # 打印一段文本
    print("10,000 TO 30,000 MILES AND AT ITS PRESENT VELOCITY CAN\n");
    # 打印一段文本
    print("CIRCLE YOUR PLANET ONCE EVERY 12 TO 36 HOURS.\n");
    # 打印空行
    print("\n");
    # 打印一段文本
    print("UNFORTUNATELY, THEY ARE USING A CLOAKING DEVICE SO\n");
    # 打印一段文本
    print("YOU ARE UNABLE TO SEE THEM, BUT WITH A SPECIAL\n");
    # 打印一段文本
    print("INSTRUMENT YOU CAN TELL HOW NEAR THEIR SHIP YOUR\n");
    # 打印一段文本
    print("PHOTON BOMB EXPLODED.  YOU HAVE SEVEN HOURS UNTIL THEY\n");
    # 打印一段文本
    print("HAVE BUILT UP SUFFICIENT POWER IN ORDER TO ESCAPE\n");
    # 打印一段文本
    print("YOUR PLANET'S GRAVITY.\n");
    # 打印空行
    print("\n");
    # 打印一段文本
    print("YOUR PLANET HAS ENOUGH POWER TO FIRE ONE BOMB AN HOUR.\n");
    # 打印空行
    print("\n");
    # 打印一段文本
    print("AT THE BEGINNING OF EACH HOUR YOU WILL BE ASKED TO GIVE AN\n");
    # 打印一段文本
    print("ANGLE (BETWEEN 0 AND 360) AND A DISTANCE IN UNITS OF\n");
    # 打印一段文本
    print("100 MILES (BETWEEN 100 AND 300), AFTER WHICH YOUR BOMB'S\n");
    # 打印一段文本
    print("DISTANCE FROM THE ENEMY SHIP WILL BE GIVEN.\n");
    # 打印空行
    print("\n");
    # 打印一段文本
    print("AN EXPLOSION WITHIN 5,000 MILES OF THE ROMULAN SHIP\n");
    # 打印一段文本
    print("WILL DESTROY IT.\n");
    # 打印空行
    print("\n");
    # 打印一段文本
    print("BELOW IS A DIAGRAM TO HELP YOU VISUALIZE YOUR PLIGHT.\n");
    # 打印空行
    print("\n");
    # 打印空行
    print("\n");
    # 打印一段文本
    print("                          90\n");
    # 打印一段文本
    print("                    0000000000000\n");
    # 打印一段文本
    print("                 0000000000000000000\n");
    # 打印一段文本
    print("               000000           000000\n");
    # 打印一段文本
    print("             00000                 00000\n");
    # 打印一段文本
    print("            00000    XXXXXXXXXXX    00000\n");
    # 打印一段文本
    print("           00000    XXXXXXXXXXXXX    00000\n");
    # 打印一段文本
    print("          0000     XXXXXXXXXXXXXXX     0000\n");
    # 打印一段文本
    print("         0000     XXXXXXXXXXXXXXXXX     0000\n");
    # 打印一段文本
    print("        0000     XXXXXXXXXXXXXXXXXXX     0000\n");
    # 打印一段文本
    print("180<== 00000     XXXXXXXXXXXXXXXXXXX     00000 ==>0\n");
    # 打印一段文本
    print("        0000     XXXXXXXXXXXXXXXXXXX     0000\n");
    # 打印一段文本
    print("         0000     XXXXXXXXXXXXXXXXX     0000\n");
    # 打印一段文本
    print("          0000     XXXXXXXXXXXXXXX     0000\n");
    # 打印太空图形
    print("           00000    XXXXXXXXXXXXX    00000\n");
    print("            00000    XXXXXXXXXXX    00000\n");
    print("             00000                 00000\n");
    print("               000000           000000\n");
    print("                 0000000000000000000\n");
    print("                    0000000000000\n");
    print("                         270\n");
    print("\n");
    print("X - YOUR PLANET\n");
    print("O - THE ORBIT OF THE ROMULAN SHIP\n");
    print("\n");
    print("ON THE ABOVE DIAGRAM, THE ROMULAN SHIP IS CIRCLING\n");
    print("COUNTERCLOCKWISE AROUND YOUR PLANET.  DON'T FORGET THAT\n");
    print("WITHOUT SUFFICIENT POWER THE ROMULAN SHIP'S ALTITUDE\n");
    print("AND ORBITAL RATE WILL REMAIN CONSTANT.\n");
    print("\n");
    print("GOOD LUCK.  THE FEDERATION IS COUNTING ON YOU.\n");
    # 进入无限循环
    while (1) {
        # 生成随机角度
        a = Math.floor(360 * Math.random());
        # 生成随机距离
        d = Math.floor(200 * Math.random() + 200);
        # 生成随机半径
        r = Math.floor(20 * Math.random() + 10);
        # 初始化小时数
        h = 0;
        # 在小时数小于7的情况下执行循环
        while (h < 7) {
            # 打印提示信息
            print("\n");
            print("\n");
            print("THIS IS HOUR " + (h + 1) + ", AT WHAT ANGLE DO YOU WISH TO SEND\n");
            print("YOUR PHOTON BOMB");
            # 获取用户输入的角度
            a1 = parseFloat(await input());
            print("HOW FAR OUT DO YOU WISH TO DETONATE IT");
            # 获取用户输入的距离
            d1 = parseFloat(await input());
            print("\n");
            print("\n");
            # 更新角度
            a += r;
            if (a >= 360)
                a -= 360;
            # 计算角度差
            t = Math.abs(a - a1);
            if (t >= 180)
                t = 360 - t;
            # 计算距离
            c = Math.sqrt(d * d + d1 * d1 - 2 * d * d1 * Math.cos(t * Math.PI / 180));
            # 打印结果
            print("YOUR PHOTON BOMB EXPLODED " + c + "*10^2 MILES FROM THE\n");
            print("ROMULAN SHIP.\n");
            # 如果距离小于等于50，跳出循环
            if (c <= 50)
                break;
            # 更新小时数
            h++;
        }
        # 如果小时数等于7，打印失败信息
        if (h == 7) {
            print("YOU HAVE ALLOWED THE ROMULANS TO ESCAPE.\n");
        } else {
            # 否则打印成功信息
            print("YOU HAVE SUCCESSFULLY COMPLETED YOUR MISSION.\n");
        }
        # 打印提示信息
        print("ANOTHER ROMULAN SHIP HAS GONE INTO ORBIT.\n");
        print("DO YOU WISH TO TRY TO DESTROY IT");
        # 获取用户输入
        str = await input();
        # 如果输入不是"YES"，跳出循环
        if (str != "YES")
            break;
    }
    # 打印结束信息
    print("GOOD BYE.\n");
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```