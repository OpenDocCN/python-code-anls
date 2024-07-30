# `.\comic-translate\modules\utils\download.py`

```py
# 导入必要的模块：os、sys、hashlib
import os, sys, hashlib
# 从torch.hub中导入download_url_to_file函数
from torch.hub import download_url_to_file
# 导入loguru中的logger对象
from loguru import logger

# 计算文件的SHA256校验和
def calculate_sha256_checksum(file_path):
    # 创建SHA256哈希对象
    sha256_hash = hashlib.sha256()
    # 以二进制只读方式打开文件
    with open(file_path, "rb") as f:
        # 以4K为单位迭代读取文件并更新哈希值
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    # 返回SHA256校验和的十六进制表示
    return sha256_hash.hexdigest()

# 获取模型数据
def get_models(data):
    # 检查保存目录是否存在，不存在则创建
    save_dir = data['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # 遍历文件列表
    for i, file_name in enumerate(data['files']):
        # 构造文件的完整URL
        file_url = f"{data['url']}{file_name}"
        # 构造文件的完整路径
        file_path = os.path.join(data['save_dir'], file_name)
        # 获取预计的SHA256校验和
        expected_checksum = data['sha256_pre_calculated'][i]

        # 如果文件已存在
        if os.path.exists(file_path):
            # 如果有预期的校验和，验证文件的完整性
            if expected_checksum is not None:
                calculated_checksum = calculate_sha256_checksum(file_path)
                # 如果校验和匹配，则继续下一个文件
                if calculated_checksum == expected_checksum:
                    continue
                else:
                    # 如果校验和不匹配，输出错误信息并重新下载文件
                    print(f"Checksum mismatch for {file_name}. Expected {expected_checksum}, got {calculated_checksum}. Redownloading...")

        # 输出下载信息到标准错误输出
        sys.stderr.write('Downloading: "{}" to {}\n'.format(file_url, save_dir))
        # 使用下载函数下载文件到指定路径，不使用哈希前缀，显示下载进度
        download_url_to_file(file_url, file_path, hash_prefix=None, progress=True)

        # 如果有预期的校验和
        if expected_checksum:
            # 计算下载后文件的SHA256校验和
            calculated_checksum = calculate_sha256_checksum(file_path)
            # 如果校验和匹配，则记录下载成功的信息
            if calculated_checksum == expected_checksum:
                logger.info(f"Download model success, sha256: {calculated_checksum}")
            else:
                # 如果校验和不匹配，则删除错误的模型文件并记录错误信息
                try:
                    os.remove(file_path)
                    logger.error(
                        f"Model sha256: {calculated_checksum}, expected sha256: {expected_checksum}, wrong model deleted. Please restart comic-translate."
                    )
                except:
                    logger.error(
                        f"Model sha256: {calculated_checksum}, expected sha256: {expected_checksum}, please delete {file_path} and restart comic-translate."
                    )
                # 程序退出，返回错误状态码
                exit(-1)

# 默认的漫画OCR模型数据
manga_ocr_data = {
    'url': 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/',
    'files': [
        'pytorch_model.bin', 'config.json', 'preprocessor_config.json',
        'README.md', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt'
    ],
    # 预先计算的文件SHA256校验和列表
    'sha256_pre_calculated': [
        'c63e0bb5b3ff798c5991de18a8e0956c7ee6d1563aca6729029815eda6f5c2eb', 
        '8c0e395de8fa699daaac21aee33a4ba9bd1309cfbff03147813d2a025f39f349', 
        'af4eb4d79cf61b47010fc0bc9352ee967579c417423b4917188d809b7e048948', 
        '32f413afcc4295151e77d25202c5c5d81ef621b46f947da1c3bde13256dc0d5f', 
        '303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3', 
        'd775ad1deac162dc56b84e9b8638f95ed8a1f263d0f56f4f40834e26e205e266', 
        '344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d'
    ],



# 预先计算的 SHA-256 值列表，用于验证下载文件的完整性
'sha256_pre_calculated': [
    'c63e0bb5b3ff798c5991de18a8e0956c7ee6d1563aca6729029815eda6f5c2eb', 
    '8c0e395de8fa699daaac21aee33a4ba9bd1309cfbff03147813d2a025f39f349', 
    'af4eb4d79cf61b47010fc0bc9352ee967579c417423b4917188d809b7e048948', 
    '32f413afcc4295151e77d25202c5c5d81ef621b46f947da1c3bde13256dc0d5f', 
    '303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3', 
    'd775ad1deac162dc56b84e9b8638f95ed8a1f263d0f56f4f40834e26e205e266', 
    '344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d'
],



    'save_dir': 'models/ocr/manga-ocr-base'



# 模型保存目录路径，用于指定保存 OCR 模型文件的基础目录
'save_dir': 'models/ocr/manga-ocr-base'
}

# comic_text_segmenter_data 字典定义，包含模型的相关信息
comic_text_segmenter_data = {
    'url': 'https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m/resolve/main/',
    'files': [
        'comic-text-segmenter.pt'],  # 模型文件名列表
    'sha256_pre_calculated': [
        'f2dded0d2f5aaa25eed49f1c34a4720f1c1cd40da8bc3138fde1abb202de625e',  # 预先计算的 SHA-256 值列表
    ],
    'save_dir': 'models/detection'  # 模型保存目录
}

# inpaint_lama_finetuned_data 字典定义，包含模型的相关信息
inpaint_lama_finetuned_data = {
    'url': 'https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/',
    'files': [
        'lama_large_512px.ckpt'  # 模型文件名列表
    ],
    'sha256_pre_calculated': [
        "11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935"  # 预先计算的 SHA-256 值列表
    ],
    'save_dir': 'models/inpainting'  # 模型保存目录
}

# comic_bubble_detector_data 字典定义，包含模型的相关信息
comic_bubble_detector_data = {
    'url': 'https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m/resolve/main/',
    'files': [
        'comic-speech-bubble-detector.pt'  # 模型文件名列表
    ],
    'sha256_pre_calculated': [
        '10bc9f702698148e079fb4462a6b910fcd69753e04838b54087ef91d5633097b'  # 预先计算的 SHA-256 值列表
    ],
    'save_dir': 'models/detection'  # 模型保存目录  
}

# pororo_data 字典定义，包含模型的相关信息
pororo_data = {
    'url': 'https://twg.kakaocdn.net/pororo/ko/models/misc/',
    'files': [
        'craft.pt', 'brainocr.pt', 'ocr-opt.txt'  # 模型文件名列表
    ],
    'sha256_pre_calculated': [
        '4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17',  # 预先计算的 SHA-256 值列表
        '125820ba8ae4fa5d9fd8b8a2d4d4a7afe96a70c32b1aa01d4129001a6f61baec',
        'dd471474e91d78e54b179333439fea58158ad1a605df010ea0936dcf4387a8c2'
    ],
    'save_dir': 'models/ocr/pororo'  # 模型保存目录  
}

# mandatory_models 列表定义，包含需要下载的模型数据字典
mandatory_models = [comic_text_segmenter_data, inpaint_lama_finetuned_data, comic_bubble_detector_data]
```