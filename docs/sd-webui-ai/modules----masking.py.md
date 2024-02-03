# `stable-diffusion-webui\modules\masking.py`

```py
from PIL import Image, ImageFilter, ImageOps

# 找到包含图像中所有掩码区域的矩形区域。返回矩形的 (x1, y1, x2, y2) 坐标。
# 例如，如果用户在一个 512x512 图像的右上部分涂色，则结果可能是 (256, 0, 512, 256)
def get_crop_region(mask, pad=0):
    h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    return (
        int(max(crop_left-pad, 0)),
        int(max(crop_top-pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h))
    )

# 将 get_crop_region() 返回的裁剪区域扩展到与图像处理时的比例相匹配；返回扩展后的区域
# 例如，如果用户在 128x32 区域绘制了掩码，处理时的尺寸为 512x512，则区域将扩展为 128x128
def expand_crop_region(crop_region, processing_width, processing_height, image_width, image_height):
    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height
    # 如果裁剪区域的宽高比大于处理比例
    if ratio_crop_region > ratio_processing:
        # 计算期望的高度
        desired_height = (x2 - x1) / ratio_processing
        # 计算期望高度与当前高度的差值
        desired_height_diff = int(desired_height - (y2-y1))
        # 调整裁剪区域的上下边界
        y1 -= desired_height_diff//2
        y2 += desired_height_diff - desired_height_diff//2
        # 如果裁剪区域的下边界超过图像高度
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        # 如果裁剪区域的上边界小于0
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        # 再次检查裁剪区域的下边界是否超过图像高度
        if y2 >= image_height:
            y2 = image_height
    # 如果裁剪区域的宽高比小于等于处理比例
    else:
        # 计算期望的宽度
        desired_width = (y2 - y1) * ratio_processing
        # 计算期望宽度与当前宽度的差值
        desired_width_diff = int(desired_width - (x2-x1))
        # 调整裁剪区域的左右边界
        x1 -= desired_width_diff//2
        x2 += desired_width_diff - desired_width_diff//2
        # 如果裁剪区域的右边界超过图像宽度
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        # 如果裁剪区域的左边界小于0
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        # 再次检查裁剪区域的右边界是否超过图像宽度
        if x2 >= image_width:
            x2 = image_width

    # 返回调整后的裁剪区域坐标
    return x1, y1, x2, y2
# 使用图像中的颜色填充掩码区域，使用模糊效果。效果不是非常有效。
def fill(image, mask):
    # 创建一个与原始图像相同大小的新 RGBA 图像
    image_mod = Image.new('RGBA', (image.width, image.height))

    # 创建一个与原始图像相同大小的 RGBa 图像
    image_masked = Image.new('RGBa', (image.width, image.height))
    # 将原始图像转换为 RGBA 格式，再转换为 RGBa 格式，并粘贴到 RGBa 图像中，使用反转的掩码
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

    # 将 RGBa 图像转换为 RGBa 格式
    image_masked = image_masked.convert('RGBa')

    # 对不同半径和重复次数的高斯模糊进行循环处理
    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        # 对 RGBa 图像进行高斯模糊处理，并转换为 RGBA 格式
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        # 根据重复次数将模糊图像合成到新图像中
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    # 将新图像转换为 RGB 格式并返回
    return image_mod.convert("RGB")
```