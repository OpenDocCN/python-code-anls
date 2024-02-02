# `yolov5-DNF\json2yolo.py`

```py
import json  # 导入json模块
import os  # 导入os模块

# 将标签名映射为数字ID的字典
name2id = {'hero': 0, 'small_map': 1, "monster": 2, 'money': 3, 'material': 4, 'door': 5, 'BOSS': 6, 'box': 7, 'options': 8}
               
# 将边界框坐标转换为Yolo格式
def convert(img_size, box):
    dw = 1./(img_size[0])  # 计算宽度缩放比例
    dh = 1./(img_size[1])  # 计算高度缩放比例
    x = (box[0] + box[2])/2.0 - 1  # 计算中心点x坐标
    y = (box[1] + box[3])/2.0 - 1  # 计算中心点y坐标
    w = box[2] - box[0]  # 计算边界框宽度
    h = box[3] - box[1]  # 计算边界框高度
    x = x*dw  # 缩放中心点x坐标
    w = w*dw  # 缩放边界框宽度
    y = y*dh  # 缩放中心点y坐标
    h = h*dh  # 缩放边界框高度
    return (x,y,w,h)  # 返回转换后的边界框坐标

# 解析JSON文件并将标注信息写入对应的txt文件
def decode_json(json_floder_path,json_name):
    txt_name = 'E:\\Computer_vision\\object_DNF\\datasets\\guiqi\\yolo5_datasets\\labels\\' + json_name[0:-5] + '.txt'  # 生成对应的txt文件名
    txt_file = open(txt_name, 'w')  # 打开txt文件用于写入
    
    json_path = os.path.join(json_floder_path, json_name)  # 构建JSON文件的完整路径
    data = json.load(open(json_path, 'r', encoding='gb2312'))  # 读取JSON文件内容
    
    img_w = data['imageWidth']  # 获取图像宽度
    img_h = data['imageHeight']  # 获取图像高度
    
    for i in data['shapes']:  # 遍历JSON文件中的标注信息
        label_name = i['label']  # 获取标签名
        if (i['shape_type'] == 'rectangle'):  # 判断标注类型是否为矩形
            x1 = int(i['points'][0][0])  # 获取矩形左上角x坐标
            y1 = int(i['points'][0][1])  # 获取矩形左上角y坐标
            x2 = int(i['points'][1][0])  # 获取矩形右下角x坐标
            y2 = int(i['points'][1][1])  # 获取矩形右下角y坐标
            
            bb = (x1,y1,x2,y2)  # 构建边界框坐标元组
            bbox = convert((img_w,img_h),bb)  # 转换边界框坐标为Yolo格式
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')  # 将标注信息写入txt文件
    
if __name__ == "__main__":
    json_floder_path = r'E:\Computer_vision\object_DNF\datasets\guiqi\yolo5_datasets\labels_json'  # JSON文件夹路径
    json_names = os.listdir(json_floder_path)  # 获取JSON文件夹下的所有文件名
    for json_name in json_names:  # 遍历JSON文件夹下的所有文件
        decode_json(json_floder_path,json_name)  # 解析JSON文件并将标注信息写入对应的txt文件
```