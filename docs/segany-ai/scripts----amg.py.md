# `.\segment-anything\scripts\amg.py`

```py
# 导入所需的库
import cv2  # type: ignore
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import json
import os
from typing import Any, Dict, List

# 创建参数解析器
parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

# 添加输入路径参数
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

# 添加输出路径参数
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

# 添加模型类型参数
parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

# 添加检查点路径参数
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

# 添加设备参数
parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

# 添加是否转换为 RLE 参数
parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

# 创建 AMG 设置参数组
amg_settings = parser.add_argument_group("AMG Settings")

# 添加每边点数参数
amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

# 添加每批次点数参数
amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    # 定义命令行参数，用于指定一次处理的输入点的数量
# 添加参数设置：预测 IOU 阈值，排除模型预测得分低于此阈值的掩模
amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

# 添加参数设置：稳定性分数阈值，排除稳定性分数低于此阈值的掩模
amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

# 添加参数设置：稳定性分数偏移量，较大的值在测量稳定性分数时会扰动掩模更多
amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

# 添加参数设置：框 NMS 阈值，用于排除重复的掩模
amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

# 添加参数设置：裁剪 N 层，如果大于 0，则在图像的较小裁剪上运行掩模生成以生成更多掩模
amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

# 添加参数设置：裁剪 NMS 阈值，用于在不同裁剪之间排除重复掩模的重叠阈值
amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

# 添加参数设置：裁剪重叠比例，较大的数字意味着图像裁剪将更多地重叠
amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

# 添加参数设置：裁剪 N 点缩放因子，每个裁剪层的每边的点数减少此因子
amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

# 添加参数设置：最小掩模区域面积，通过后处理删除像素值小于此值的断开掩模区域或孔
amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)

# 定义函数：将掩模写入文件夹
def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    # 定义 CSV 文件头部信息
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    # 遍历 masks 列表，获取索引 i 和每个 mask 数据
    for i, mask_data in enumerate(masks):
        # 获取 mask 数据中的分割信息
        mask = mask_data["segmentation"]
        # 构建文件名，格式为 "{索引}.png"
        filename = f"{i}.png"
        # 将 mask 数据保存为图片文件，路径为 path，像素值乘以 255
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        # 构建 mask 的元数据信息列表
        mask_metadata = [
            str(i),  # 索引
            str(mask_data["area"]),  # 区域面积
            *[str(x) for x in mask_data["bbox"]],  # 边界框信息
            *[str(x) for x in mask_data["point_coords"][0]],  # 点坐标信息
            str(mask_data["predicted_iou"]),  # 预测的 IoU
            str(mask_data["stability_score"]),  # 稳定性评分
            *[str(x) for x in mask_data["crop_box"]],  # 裁剪框信息
        ]
        # 将 mask 的元数据信息列表转换为逗号分隔的字符串
        row = ",".join(mask_metadata)
        # 将该行元数据信息添加到 metadata 列表中
        metadata.append(row)
    # 构建 metadata.csv 文件的路径
    metadata_path = os.path.join(path, "metadata.csv")
    # 将 metadata 列表中的内容写入 metadata.csv 文件
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    # 函数执行完毕，返回
    return
# 根据传入的参数args，构建AMG的关键字参数字典
def get_amg_kwargs(args):
    # 初始化AMG的关键字参数字典
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    # 过滤掉值为None的关键字参数
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    # 返回AMG的关键字参数字典
    return amg_kwargs

# 主函数
def main(args: argparse.Namespace) -> None:
    # 打印加载模型信息
    print("Loading model...")
    # 根据模型类型从sam_model_registry中获取模型实例
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    # 将模型移动到指定设备
    _ = sam.to(device=args.device)
    # 根据参数决定输出模式
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    # 获取AMG的关键字参数字典
    amg_kwargs = get_amg_kwargs(args)
    # 创建SamAutomaticMaskGenerator实例
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    # 判断输入路径是文件还是目录
    if not os.path.isdir(args.input):
        # 如果是文件，将其作为目标列表的唯一元素
        targets = [args.input]
    else:
        # 如果是目录，获取目录下所有文件名（不包括子目录）
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        # 将文件名与输入目录拼接成完整路径
        targets = [os.path.join(args.input, f) for f in targets]

    # 创建输出目录（如果不存在）
    os.makedirs(args.output, exist_ok=True)
    # 遍历目标文件列表
    for t in targets:
        # 打印正在处理的目标文件名
        print(f"Processing '{t}'...")
        # 读取目标文件作为图像
        image = cv2.imread(t)
        # 如果图像为空，打印无法加载图像的信息并跳过当前文件
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        # 将图像从BGR颜色空间转换为RGB颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用生成器生成掩模
        masks = generator.generate(image)

        # 获取目标文件的基本名称
        base = os.path.basename(t)
        # 去除基本名称的扩展名
        base = os.path.splitext(base)[0]
        # 设置保存文件的基本路径
        save_base = os.path.join(args.output, base)
        # 如果输出模式为"binary_mask"
        if output_mode == "binary_mask":
            # 创建保存路径，如果路径不存在则报错
            os.makedirs(save_base, exist_ok=False)
            # 将掩模保存到文件夹中
            write_masks_to_folder(masks, save_base)
        else:
            # 设置保存文件的完整路径
            save_file = save_base + ".json"
            # 打开保存文件，以写入模式
            with open(save_file, "w") as f:
                # 将掩模数据以JSON格式写入文件
                json.dump(masks, f)
    # 打印处理完成的信息
    print("Done!")
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 解析命令行参数并存储在args变量中
    args = parser.parse_args()
    # 调用main函数并传入参数args
    main(args)
```