# `yolov5-DNF\image_grab.py`

```
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:45:04 2020

@author: analoganddigital   ( GitHub )
"""

# 导入所需的库
import os
import time
import cv2
import getkeys
import grabscreen

# 设置等待时间、L_t、保存步数、数据路径和窗口大小
wait_time = 5
L_t = 3
save_step = 200
data_path = 'datasets/guiqi/material'
window_size = (0,0,1280,800)#384,344  192,172 96,86

# 初始化训练数据列表
training_data = []

# 设置保存标志
save = True

# 倒计时等待
for i in list(range(wait_time))[::-1]:
    print(i+1)
    time.sleep(1)

# 初始化计时器和原始数据数量
last_time = time.time()
counter = 0
org_num = len(os.listdir(data_path))

# 循环收集训练数据
while(True):
    # 获取按键输入
    output_key = getkeys.get_key(getkeys.key_check())

    # 如果按键为100，保存训练数据并退出循环
    if output_key == 100:
        if save:
            print(len(training_data) + counter*save_step)
            for i, d in enumerate(training_data):
                file_name = os.path.join(data_path, str(org_num + counter*save_step + i) + "_" + str(d[1]) + '.jpg')
                cv2.imwrite(file_name, d[0])
            print("save finish")
        break

    # 获取屏幕截图并转换为灰度图像
    screen_gray = cv2.cvtColor(grabscreen.grab_screen(window_size),cv2.COLOR_BGRA2BGR)
    screen_reshape = cv2.resize(screen_gray,(1280,800)) # 1200, 750   600, 375

    # 将截图和按键输入添加到训练数据中
    training_data.append([screen_reshape,output_key])

    # 如果训练数据数量达到保存步数的倍数，保存训练数据并清空列表
    if len(training_data) % save_step == 0 and save:
        print(len(training_data))
        for i, d in enumerate(training_data):
            file_name = os.path.join(data_path, str(org_num + counter*save_step + i) + "_" + str(d[1]) + '.jpg')
            cv2.imwrite(file_name, d[0])
        training_data.clear()
        counter += 1
    cv2.imshow('window1',screen_reshape)

    # 打印每帧用时和瞬时fps
    print('每帧用时 {} 秒'.format(time.time()-last_time))
    print("瞬时fps：", 1/(time.time()-last_time))
    last_time = time.time()

    # 如果按下键盘上的q键，退出循环
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
# 等待用户按下任意键，用于在视频结束后等待用户操作
cv2.waitKey()
# 销毁所有窗口，释放资源
cv2.destroyAllWindows()
```