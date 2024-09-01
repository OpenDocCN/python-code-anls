# `.\flux\src\flux\api.py`

```py
# 导入标准库中的 io 模块，用于处理
    ):
        """
        Manages an image generation request to the API.

        Args:
            prompt: Prompt to sample
            width: Width of the image in pixel
            height: Height of the image in pixel
            name: Name of the model
            num_steps: Number of network evaluations
            prompt_upsampling: Use prompt upsampling
            seed: Fix the generation seed
            validate: Run input validation
            launch: Directly launches request
            api_key: Your API key if not provided by the environment

        Raises:
            ValueError: For invalid input
            ApiException: For errors raised from the API
        """
        # 如果需要验证输入
        if validate:
            # 检查模型名称是否有效
            if name not in ["flux.1-pro"]:
                raise ValueError(f"Invalid model {name}")
            # 检查宽度是否是 32 的倍数
            elif width % 32 != 0:
                raise ValueError(f"width must be divisible by 32, got {width}")
            # 检查宽度是否在合法范围内
            elif not (256 <= width <= 1440):
                raise ValueError(f"width must be between 256 and 1440, got {width}")
            # 检查高度是否是 32 的倍数
            elif height % 32 != 0:
                raise ValueError(f"height must be divisible by 32, got {height}")
            # 检查高度是否在合法范围内
            elif not (256 <= height <= 1440):
                raise ValueError(f"height must be between 256 and 1440, got {height}")
            # 检查步骤数量是否在合法范围内
            elif not (1 <= num_steps <= 50):
                raise ValueError(f"steps must be between 1 and 50, got {num_steps}")

        # 创建请求 JSON 对象，包含所有必需的参数
        self.request_json = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "variant": name,
            "steps": num_steps,
            "prompt_upsampling": prompt_upsampling,
        }
        # 如果指定了种子，将其添加到请求 JSON 中
        if seed is not None:
            self.request_json["seed"] = seed

        # 初始化实例变量
        self.request_id: str | None = None
        self.result: dict | None = None
        self._image_bytes: bytes | None = None
        self._url: str | None = None
        # 如果没有提供 API 密钥，则从环境变量中获取
        if api_key is None:
            self.api_key = os.environ.get("BFL_API_KEY")
        else:
            # 否则使用提供的 API 密钥
            self.api_key = api_key

        # 如果需要立即发起请求
        if launch:
            self.request()

    def request(self):
        """
        Request to generate the image.
        """
        # 如果已经有请求 ID，则不再发起请求
        if self.request_id is not None:
            return
        # 发起 POST 请求以生成图像
        response = requests.post(
            f"{API_ENDPOINT}/v1/image",
            headers={
                "accept": "application/json",
                "x-key": self.api_key,
                "Content-Type": "application/json",
            },
            json=self.request_json,
        )
        # 解析响应为 JSON
        result = response.json()
        # 如果响应状态码不是 200，抛出 API 异常
        if response.status_code != 200:
            raise ApiException(status_code=response.status_code, detail=result.get("detail"))
        # 存储请求 ID
        self.request_id = response.json()["id"]
    # 定义一个方法来等待生成完成并检索响应结果
    def retrieve(self) -> dict:
        """
        等待生成完成并检索响应
        """
        # 如果 request_id 为空，则调用请求方法生成请求 ID
        if self.request_id is None:
            self.request()
        # 循环等待直到结果可用
        while self.result is None:
            # 发送 GET 请求以获取结果
            response = requests.get(
                f"{API_ENDPOINT}/v1/get_result",
                headers={
                    "accept": "application/json",
                    "x-key": self.api_key,
                },
                params={
                    "id": self.request_id,
                },
            )
            # 将响应内容转换为 JSON 格式
            result = response.json()
            # 检查返回结果中是否包含状态字段
            if "status" not in result:
                # 如果没有状态字段，抛出 API 异常
                raise ApiException(status_code=response.status_code, detail=result.get("detail"))
            # 如果状态是“Ready”，则将结果保存到实例变量
            elif result["status"] == "Ready":
                self.result = result["result"]
            # 如果状态是“Pending”，则等待 0.5 秒再重试
            elif result["status"] == "Pending":
                time.sleep(0.5)
            # 如果状态是其他值，抛出 API 异常
            else:
                raise ApiException(status_code=200, detail=f"API returned status '{result['status']}'")
        # 返回最终结果
        return self.result

    # 定义一个属性方法，返回生成的图像字节
    @property
    def bytes(self) -> bytes:
        """
        生成的图像字节
        """
        # 如果图像字节为空，则从 URL 获取图像数据
        if self._image_bytes is None:
            response = requests.get(self.url)
            # 如果响应状态码是 200，则保存图像字节
            if response.status_code == 200:
                self._image_bytes = response.content
            # 否则抛出 API 异常
            else:
                raise ApiException(status_code=response.status_code)
        # 返回图像字节
        return self._image_bytes

    # 定义一个属性方法，返回图像的公共 URL
    @property
    def url(self) -> str:
        """
        检索图像的公共 URL
        """
        # 如果 URL 为空，则调用 retrieve 方法获取结果并保存 URL
        if self._url is None:
            result = self.retrieve()
            self._url = result["sample"]
        # 返回图像的 URL
        return self._url

    # 定义一个属性方法，返回 PIL 图像对象
    @property
    def image(self) -> Image.Image:
        """
        加载图像为 PIL Image 对象
        """
        return Image.open(io.BytesIO(self.bytes))

    # 定义一个方法来将生成的图像保存到本地路径
    def save(self, path: str):
        """
        将生成的图像保存到本地路径
        """
        # 获取 URL 的文件扩展名
        suffix = Path(self.url).suffix
        # 如果路径没有扩展名，则将扩展名添加到路径中
        if not path.endswith(suffix):
            path = path + suffix
        # 创建保存路径的父目录（如果不存在）
        Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
        # 将图像字节写入指定路径
        with open(path, "wb") as file:
            file.write(self.bytes)
# 确保只有在直接运行该脚本时才执行以下代码
if __name__ == "__main__":
    # 从 fire 库中导入 Fire 类
    from fire import Fire

    # 使用 Fire 类启动命令行界面，传入 ImageRequest 作为处理对象
    Fire(ImageRequest)
```