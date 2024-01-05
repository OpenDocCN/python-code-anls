# `68_Orbit\javascript\orbit.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 INPUT 元素，用于接收用户输入
// 在页面上打印提示符 "? "
// 设置 INPUT 元素的类型为文本输入类型
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
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

var a = [];  // 创建一个空数组

// Main program
async function main()
{
    print(tab(33) + "ORBIT\n");  // 打印带有缩进的字符串 "ORBIT"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有缩进的字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP.\n");  // 打印字符串 "SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP."
    print("\n");  // 打印空行
    print("THE SHIP IS IN A CONSTANT POLAR ORBIT.  ITS\n");  // 打印字符串 "THE SHIP IS IN A CONSTANT POLAR ORBIT.  ITS"
    print("DISTANCE FROM THE CENTER OF YOUR PLANET IS FROM\n");  // 打印字符串 "DISTANCE FROM THE CENTER OF YOUR PLANET IS FROM"
    print("10,000 TO 30,000 MILES AND AT ITS PRESENT VELOCITY CAN\n");  // 打印字符串 "10,000 TO 30,000 MILES AND AT ITS PRESENT VELOCITY CAN"
    print("CIRCLE YOUR PLANET ONCE EVERY 12 TO 36 HOURS.\n");  // 打印字符串 "CIRCLE YOUR PLANET ONCE EVERY 12 TO 36 HOURS."
    # 打印空行
    print("\n");
    # 打印提示信息
    print("UNFORTUNATELY, THEY ARE USING A CLOAKING DEVICE SO\n");
    print("YOU ARE UNABLE TO SEE THEM, BUT WITH A SPECIAL\n");
    print("INSTRUMENT YOU CAN TELL HOW NEAR THEIR SHIP YOUR\n");
    print("PHOTON BOMB EXPLODED.  YOU HAVE SEVEN HOURS UNTIL THEY\n");
    print("HAVE BUILT UP SUFFICIENT POWER IN ORDER TO ESCAPE\n");
    print("YOUR PLANET'S GRAVITY.\n");
    print("\n");
    print("YOUR PLANET HAS ENOUGH POWER TO FIRE ONE BOMB AN HOUR.\n");
    print("\n");
    print("AT THE BEGINNING OF EACH HOUR YOU WILL BE ASKED TO GIVE AN\n");
    print("ANGLE (BETWEEN 0 AND 360) AND A DISTANCE IN UNITS OF\n");
    print("100 MILES (BETWEEN 100 AND 300), AFTER WHICH YOUR BOMB'S\n");
    print("DISTANCE FROM THE ENEMY SHIP WILL BE GIVEN.\n");
    print("\n");
    print("AN EXPLOSION WITHIN 5,000 MILES OF THE ROMULAN SHIP\n");
    print("WILL DESTROY IT.\n");
    print("\n");
    print("BELOW IS A DIAGRAM TO HELP YOU VISUALIZE YOUR PLIGHT.\n");
    print("\n");
    # 打印空行
    print("\n");
    # 打印一行字符串 "90"
    print("                          90\n");
    # 打印一行字符串 "0000000000000"
    print("                    0000000000000\n");
    # 打印一行字符串 "0000000000000000000"
    print("                 0000000000000000000\n");
    # 打印一行字符串 "000000           000000"
    print("               000000           000000\n");
    # 打印一行字符串 "00000                 00000"
    print("             00000                 00000\n");
    # 打印一行字符串 "00000    XXXXXXXXXXX    00000"
    print("            00000    XXXXXXXXXXX    00000\n");
    # 打印一行字符串 "00000    XXXXXXXXXXXXX    00000"
    print("           00000    XXXXXXXXXXXXX    00000\n");
    # 打印一行字符串 "0000     XXXXXXXXXXXXXXX     0000"
    print("          0000     XXXXXXXXXXXXXXX     0000\n");
    # 打印一行字符串 "0000     XXXXXXXXXXXXXXXXX     0000"
    print("         0000     XXXXXXXXXXXXXXXXX     0000\n");
    # 打印一行字符串 "0000     XXXXXXXXXXXXXXXXXXX     0000"
    print("        0000     XXXXXXXXXXXXXXXXXXX     0000\n");
    # 打印一行字符串 "180<== 00000     XXXXXXXXXXXXXXXXXXX     00000 ==>0"
    print("180<== 00000     XXXXXXXXXXXXXXXXXXX     00000 ==>0\n");
    # 打印一行字符串 "0000     XXXXXXXXXXXXXXXXXXX     0000"
    print("        0000     XXXXXXXXXXXXXXXXXXX     0000\n");
    # 打印一行字符串 "0000     XXXXXXXXXXXXXXXXX     0000"
    print("         0000     XXXXXXXXXXXXXXXXX     0000\n");
    # 打印一行字符串 "0000     XXXXXXXXXXXXXXX     0000"
    print("          0000     XXXXXXXXXXXXXXX     0000\n");
    # 打印一行字符串 "00000    XXXXXXXXXXXXX    00000"
    print("           00000    XXXXXXXXXXXXX    00000\n");
    # 打印一行字符串 "00000    XXXXXXXXXXX    00000"
    print("            00000    XXXXXXXXXXX    00000\n");
    # 打印一行字符串 "00000                 00000"
    print("             00000                 00000\n");
    # 打印一行字符串 "000000           000000"
    print("               000000           000000\n");
    # 打印一行字符串 "0000000000000000000"
    print("                 0000000000000000000\n");
    # 打印图形的顶部部分
    print("                    0000000000000\n");
    # 打印图形的中间部分
    print("                         270\n");
    # 打印空行
    print("\n");
    # 打印提示信息
    print("X - YOUR PLANET\n");
    # 打印提示信息
    print("O - THE ORBIT OF THE ROMULAN SHIP\n");
    # 打印空行
    print("\n");
    # 打印提示信息
    print("ON THE ABOVE DIAGRAM, THE ROMULAN SHIP IS CIRCLING\n");
    # 打印提示信息
    print("COUNTERCLOCKWISE AROUND YOUR PLANET.  DON'T FORGET THAT\n");
    # 打印提示信息
    print("WITHOUT SUFFICIENT POWER THE ROMULAN SHIP'S ALTITUDE\n");
    # 打印提示信息
    print("AND ORBITAL RATE WILL REMAIN CONSTANT.\n");
    # 打印空行
    print("\n");
    # 打印提示信息
    print("GOOD LUCK.  THE FEDERATION IS COUNTING ON YOU.\n");
    # 进入无限循环
    while (1) {
        # 生成一个随机的角度
        a = Math.floor(360 * Math.random());
        # 生成一个随机的距离
        d = Math.floor(200 * Math.random() + 200);
        # 生成一个随机的半径
        r = Math.floor(20 * Math.random() + 10);
        # 初始化高度
        h = 0;
        # 进入内部循环，循环次数小于7
        while (h < 7) {
            # 打印两个空行
            print("\n");
            print("\n");
# 打印提示信息，询问用户在当前小时的角度上想要发送光子炸弹
print("THIS IS HOUR " + (h + 1) + ", AT WHAT ANGLE DO YOU WISH TO SEND\n");
# 打印提示信息，询问用户希望在多远的距离上引爆光子炸弹
print("YOUR PHOTON BOMB");
# 从用户输入中获取角度值
a1 = parseFloat(await input());
# 打印提示信息，询问用户希望在多远的距离上引爆光子炸弹
print("HOW FAR OUT DO YOU WISH TO DETONATE IT");
# 从用户输入中获取距离值
d1 = parseFloat(await input());
# 打印空行
print("\n");
# 打印空行
print("\n");
# 将当前角度值增加 r
a += r;
# 如果增加后的角度值大于等于 360，则减去 360
if (a >= 360)
    a -= 360;
# 计算角度差的绝对值
t = Math.abs(a - a1);
# 如果角度差大于等于 180，则将角度差设为 360 减去角度差
if (t >= 180)
    t = 360 - t;
# 计算光子炸弹爆炸的距离
c = Math.sqrt(d * d + d1 * d1 - 2 * d * d1 * Math.cos(t * Math.PI / 180));
# 打印光子炸弹爆炸的距离信息
print("YOUR PHOTON BOMB EXPLODED " + c + "*10^2 MILES FROM THE\n");
# 打印光子炸弹爆炸的距离信息
print("ROMULAN SHIP.\n");
# 如果光子炸弹爆炸的距离小于等于 50，则跳出循环
if (c <= 50)
    break;
# 小时数加一
h++;
        if (h == 7) {  # 如果 h 的值等于 7
            print("YOU HAVE ALLOWED THE ROMULANS TO ESCAPE.\n");  # 打印“你已经让罗穆兰人逃跑了。”
        } else {  # 否则
            print("YOU HAVE SUCCESSFULLY COMPLETED YOUR MISSION.\n");  # 打印“你已成功完成了你的任务。”
        }
        print("ANOTHER ROMULAN SHIP HAS GONE INTO ORBIT.\n");  # 打印“另一艘罗穆兰飞船已进入轨道。”
        print("DO YOU WISH TO TRY TO DESTROY IT");  # 打印“你想试图摧毁它吗？”
        str = await input();  # 等待用户输入，并将输入的值赋给变量 str
        if (str != "YES")  # 如果输入的值不等于“YES”
            break;  # 跳出循环
    }
    print("GOOD BYE.\n");  # 打印“再见。”
}

main();  # 调用主函数
```