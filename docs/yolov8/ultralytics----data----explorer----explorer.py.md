# `.\yolov8\ultralytics\data\explorer\explorer.py`

```py
        data: Union[str, Path] = "coco128.yaml",
        model: str = "yolov8n.pt",
        uri: str = USER_CONFIG_DIR / "explorer",

初始化方法，接受数据配置文件路径或字符串，默认为"coco128.yaml"；模型文件名，默认为"yolov8n.pt"；URI路径，默认为用户配置目录下的"explorer"。


        self.data = Path(data)
        self.model = Path(model)
        self.uri = Path(uri)

将传入的数据路径、模型路径和URI路径转换为`Path`对象，并分别赋值给实例变量`self.data`、`self.model`和`self.uri`。


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

根据当前系统是否支持CUDA，选择使用GPU（如果可用）或CPU，并将设备类型赋值给实例变量`self.device`。


        self.model = YOLO(self.model).to(self.device).eval()

使用`YOLO`类加载指定的YOLO模型文件，并将其移动到之前确定的设备（GPU或CPU），然后设置为评估模式（eval），覆盖之前定义的`self.model`。


        self.data = ExplorerDataset(self.data)

使用`ExplorerDataset`类加载指定的数据配置文件，并赋值给实例变量`self.data`，以供后续数据集探索和操作使用。


    def embed_images(self, images: List[Union[np.ndarray, str, Path]]) -> List[np.ndarray]:
        """Embeds a list of images into feature vectors using the initialized YOLO model."""

定义一个方法`embed_images`，接受一个包含图像的列表（可以是`np.ndarray`、字符串路径或`Path`对象），返回一个包含特征向量的`np.ndarray`列表。


        embeddings = []
        for image in tqdm(images, desc="Embedding images"):
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))  # BGR
            if image is None:
                LOGGER.error(f"Image Not Found {image}")
                embeddings.append(None)
                continue
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).to(self.device).float() / 255.0
            else:
                embeddings.append(None)
                continue
            if image.ndimension() == 3:
                image = image.unsqueeze(0)
            with torch.no_grad():
                features = self.model(image)[0].cpu().numpy()
            embeddings.append(features)
        return embeddings

遍历图像列表，对每张图像进行以下操作：如果图像是字符串路径或`Path`对象，使用OpenCV加载图像（格式为BGR）；如果加载失败，记录错误并在嵌入列表中添加`None`；如果图像是`np.ndarray`，将其转换为`torch.Tensor`并移动到设备上，然后进行归一化处理；最后使用YOLO模型提取图像特征，将特征向量添加到嵌入列表中。


    def create_table(self, schema: dict) -> bool:
        """Creates a table in LanceDB using the provided schema."""

定义一个方法`create_table`，接受一个表示表结构的字典作为参数，返回布尔值表示是否成功创建表。


        success = False
        try:
            success = get_table_schema(self.uri, schema)
        except Exception as e:
            LOGGER.error(f"Error creating table: {e}")
        return success

尝试调用`get_table_schema`函数，使用提供的URI路径和表结构字典创建表格。如果出现异常，记录错误信息，并返回`False`；否则返回函数调用结果。


    def query_similarity(self, image: Union[np.ndarray, str, Path], threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Queries LanceDB for images similar to the provided image, using YOLO features."""

定义一个方法`query_similarity`，接受一个图像（可以是`np.ndarray`、字符串路径或`Path`对象）和相似度阈值作为参数，返回一个包含（文件名，相似度得分）元组的列表。


        schema = get_sim_index_schema()

调用`get_sim_index_schema`函数，获取相似度索引的模式。


        results = []
        try:
            image_embed = self.embed_images([image])[0]

调用`embed_images`方法，将提供的图像转换为特征向量。


            if image_embed is None:
                return results

如果特征向量为空，直接返回空结果列表。


            query_result = prompt_sql_query(self.uri, schema, image_embed, threshold)

使用提供的URI路径、模式、特征向量和阈值，调用`prompt_sql_query`函数执行相似度查询。


            results = [(r[0], float(r[1])) for r in query_result]
        except Exception as e:
            LOGGER.error(f"Error querying similarity: {e}")
        return results

遍历查询结果，将文件名和相似度得分组成元组，并将它们添加到结果列表中。如果出现异常，记录错误信息，并返回空的结果列表。
    ) -> None:
        """初始化 Explorer 类，设置数据集路径、模型和数据库连接的 URI。"""
        # 注意 duckdb==0.10.0 的 bug https://github.com/ultralytics/ultralytics/pull/8181
        checks.check_requirements(["lancedb>=0.4.3", "duckdb<=0.9.2"])
        import lancedb

        # 建立与数据库的连接
        self.connection = lancedb.connect(uri)
        # 设定表格名称，使用数据路径和模型名称的小写形式
        self.table_name = f"{Path(data).name.lower()}_{model.lower()}"
        # 设定相似度索引的基础名称，用于重用表格并添加阈值和 top_k 参数
        self.sim_idx_base_name = (
            f"{self.table_name}_sim_idx".lower()
        )  # 使用这个名称并附加阈值和 top_k 以重用表格
        # 初始化 YOLO 模型
        self.model = YOLO(model)
        # 数据路径
        self.data = data  # None
        # 选择集合为空
        self.choice_set = None

        # 表格为空
        self.table = None
        # 进度为 0
        self.progress = 0

    def create_embeddings_table(self, force: bool = False, split: str = "train") -> None:
        """
        创建包含数据集中图像嵌入的 LanceDB 表格。如果表格已经存在，则会重用它。传入 force=True 来覆盖现有表格。

        Args:
            force (bool): 是否覆盖现有表格。默认为 False。
            split (str): 要使用的数据集拆分。默认为 'train'。

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            ```py
        """
        # 如果表格已存在且不强制覆盖，则返回
        if self.table is not None and not force:
            LOGGER.info("表格已存在。正在重用。传入 force=True 来覆盖它。")
            return
        # 如果表格名称在连接的表格列表中且不强制覆盖，则重用表格
        if self.table_name in self.connection.table_names() and not force:
            LOGGER.info(f"表格 {self.table_name} 已存在。正在重用。传入 force=True 来覆盖它。")
            self.table = self.connection.open_table(self.table_name)
            self.progress = 1
            return
        # 如果数据为空，则抛出 ValueError
        if self.data is None:
            raise ValueError("必须提供数据以创建嵌入表格")

        # 检查数据集的详细信息
        data_info = check_det_dataset(self.data)
        # 如果拆分参数不在数据集信息中，则抛出 ValueError
        if split not in data_info:
            raise ValueError(
                f"数据集中找不到拆分 {split}。数据集中可用的键为 {list(data_info.keys())}"
            )

        # 获取选择集并确保其为列表形式
        choice_set = data_info[split]
        choice_set = choice_set if isinstance(choice_set, list) else [choice_set]
        self.choice_set = choice_set
        # 创建 ExplorerDataset 实例
        dataset = ExplorerDataset(img_path=choice_set, data=data_info, augment=False, cache=False, task=self.model.task)

        # 创建表格模式
        batch = dataset[0]
        # 获取嵌入向量的大小
        vector_size = self.model.embed(batch["im_file"], verbose=False)[0].shape[0]
        # 创建表格
        table = self.connection.create_table(self.table_name, schema=get_table_schema(vector_size), mode="overwrite")
        # 向表格添加数据
        table.add(
            self._yield_batches(
                dataset,
                data_info,
                self.model,
                exclude_keys=["img", "ratio_pad", "resized_shape", "ori_shape", "batch_idx"],
            )
        )

        self.table = table
    def _yield_batches(self, dataset: ExplorerDataset, data_info: dict, model: YOLO, exclude_keys: List[str]):
        """Generates batches of data for embedding, excluding specified keys."""
        # 遍历数据集中的每个样本
        for i in tqdm(range(len(dataset))):
            # 更新进度条
            self.progress = float(i + 1) / len(dataset)
            # 获取当前样本数据
            batch = dataset[i]
            # 排除指定的键
            for k in exclude_keys:
                batch.pop(k, None)
            # 对批次数据进行清洗
            batch = sanitize_batch(batch, data_info)
            # 使用模型对图像文件进行嵌入
            batch["vector"] = model.embed(batch["im_file"], verbose=False)[0].detach().tolist()
            # 生成包含当前批次的列表，并进行 yield
            yield [batch]

    def query(
        self, imgs: Union[str, np.ndarray, List[str], List[np.ndarray]] = None, limit: int = 25
    ) -> Any:  # pyarrow.Table
        """
        Query the table for similar images. Accepts a single image or a list of images.

        Args:
            imgs (str or list): Path to the image or a list of paths to the images.
            limit (int): Number of results to return.

        Returns:
            (pyarrow.Table): An arrow table containing the results. Supports converting to:
                - pandas dataframe: `result.to_pandas()`
                - dict of lists: `result.to_pydict()`

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            similar = exp.query(imgs=['https://ultralytics.com/images/zidane.jpg'])
            ```py
        """
        # 检查表格是否已创建
        if self.table is None:
            raise ValueError("Table is not created. Please create the table first.")
        # 如果 imgs 是单个字符串，则转换为列表
        if isinstance(imgs, str):
            imgs = [imgs]
        # 断言 imgs 类型为列表
        assert isinstance(imgs, list), f"img must be a string or a list of strings. Got {type(imgs)}"
        # 使用模型嵌入图像数据
        embeds = self.model.embed(imgs)
        # 如果传入多张图像，则计算平均嵌入向量
        embeds = torch.mean(torch.stack(embeds), 0).cpu().numpy() if len(embeds) > 1 else embeds[0].cpu().numpy()
        # 使用嵌入向量进行查询，并限制结果数量
        return self.table.search(embeds).limit(limit).to_arrow()

    def sql_query(
        self, query: str, return_type: str = "pandas"
    ):
        """
        Execute an SQL query on the embedded data.

        Args:
            query (str): SQL query string.
            return_type (str): Type of the return data. Default is "pandas".

        Returns:
            Depending on return_type:
                - "pandas": Returns a pandas dataframe.
                - "arrow": Returns a pyarrow Table.
                - "dict": Returns a dictionary.

        Example:
            ```python
            exp = Explorer()
            query_result = exp.sql_query("SELECT * FROM embeddings WHERE category='person'", return_type='arrow')
            ```py
        """
        # 执行 SQL 查询，并根据返回类型返回相应的数据结构
        if return_type == "pandas":
            return pd.read_sql_query(query, self.conn)
        elif return_type == "arrow":
            return pa.Table.from_pandas(pd.read_sql_query(query, self.conn))
        elif return_type == "dict":
            return pd.read_sql_query(query, self.conn).to_dict(orient='list')
        else:
            raise ValueError(f"Unsupported return_type: {return_type}. Choose from 'pandas', 'arrow', or 'dict'.")
    ) -> Union[Any, None]:  # pandas.DataFrame or pyarrow.Table
        """
        Run a SQL-Like query on the table. Utilizes LanceDB predicate pushdown.

        Args:
            query (str): SQL query to run.
            return_type (str): Type of the result to return. Can be either 'pandas' or 'arrow'. Defaults to 'pandas'.

        Returns:
            (pyarrow.Table): An arrow table containing the results.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            query = "SELECT * FROM 'table' WHERE labels LIKE '%person%'"
            result = exp.sql_query(query)
            ```py
        """
        # Ensure the return_type is either 'pandas' or 'arrow'
        assert return_type in {
            "pandas",
            "arrow",
        }, f"Return type should be either `pandas` or `arrow`, but got {return_type}"
        
        import duckdb
        
        # Raise an error if the table is not created
        if self.table is None:
            raise ValueError("Table is not created. Please create the table first.")

        # Note: using filter pushdown would be a better long term solution. Temporarily using duckdb for this.
        # Convert the internal table representation to Arrow format
        table = self.table.to_arrow()  # noqa NOTE: Don't comment this. This line is used by DuckDB
        
        # Check if the query starts with correct SQL keywords
        if not query.startswith("SELECT") and not query.startswith("WHERE"):
            raise ValueError(
                f"Query must start with SELECT or WHERE. You can either pass the entire query or just the WHERE "
                f"clause. found {query}"
            )
        
        # If the query starts with WHERE, prepend it with SELECT * FROM 'table'
        if query.startswith("WHERE"):
            query = f"SELECT * FROM 'table' {query}"
        
        # Log the query being executed
        LOGGER.info(f"Running query: {query}")

        # Execute the SQL query using duckdb
        rs = duckdb.sql(query)
        
        # Return the result based on the specified return_type
        if return_type == "arrow":
            return rs.arrow()
        elif return_type == "pandas":
            return rs.df()

    def plot_sql_query(self, query: str, labels: bool = True) -> Image.Image:
        """
        Plot the results of a SQL-Like query on the table.
        
        Args:
            query (str): SQL query to run.
            labels (bool): Whether to plot the labels or not.

        Returns:
            (PIL.Image): Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            query = "SELECT * FROM 'table' WHERE labels LIKE '%person%'"
            result = exp.plot_sql_query(query)
            ```py
        """
        # Execute the SQL query with return_type='arrow' to get the result as an Arrow table
        result = self.sql_query(query, return_type="arrow")
        
        # If no results are found, log and return None
        if len(result) == 0:
            LOGGER.info("No results found.")
            return None
        
        # Generate a plot based on the query result and return it as a PIL Image
        img = plot_query_result(result, plot_labels=labels)
        return Image.fromarray(img)

    def get_similar(
        self,
        img: Union[str, np.ndarray, List[str], List[np.ndarray]] = None,
        idx: Union[int, List[int]] = None,
        limit: int = 25,
        return_type: str = "pandas",
    ) -> Any:  # pandas.DataFrame or pyarrow.Table
        """
        Query the table for similar images. Accepts a single image or a list of images.

        Args:
            img (str or list): Path to the image or a list of paths to the images.
            idx (int or list): Index of the image in the table or a list of indexes.
            limit (int): Number of results to return. Defaults to 25.
            return_type (str): Type of the result to return. Can be either 'pandas' or 'arrow'. Defaults to 'pandas'.

        Returns:
            (pandas.DataFrame or pyarrow.Table): Depending on return_type, either a DataFrame or a Table.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            similar = exp.get_similar(img='https://ultralytics.com/images/zidane.jpg')
            ```py
        """
        assert return_type in {"pandas", "arrow"}, f"Return type should be `pandas` or `arrow`, but got {return_type}"
        # Check if img argument is valid and normalize it
        img = self._check_imgs_or_idxs(img, idx)
        # Query for similar images using the normalized img argument
        similar = self.query(img, limit=limit)

        if return_type == "arrow":
            # Return the query result as a pyarrow.Table
            return similar
        elif return_type == "pandas":
            # Convert the query result to a pandas DataFrame and return
            return similar.to_pandas()

    def plot_similar(
        self,
        img: Union[str, np.ndarray, List[str], List[np.ndarray]] = None,
        idx: Union[int, List[int]] = None,
        limit: int = 25,
        labels: bool = True,
    ) -> Image.Image:
        """
        Plot the similar images. Accepts images or indexes.

        Args:
            img (str or list): Path to the image or a list of paths to the images.
            idx (int or list): Index of the image in the table or a list of indexes.
            labels (bool): Whether to plot the labels or not.
            limit (int): Number of results to return. Defaults to 25.

        Returns:
            (PIL.Image): Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            similar = exp.plot_similar(img='https://ultralytics.com/images/zidane.jpg')
            ```py
        """
        # Retrieve similar images data in arrow format
        similar = self.get_similar(img, idx, limit, return_type="arrow")
        # If no similar images found, log and return None
        if len(similar) == 0:
            LOGGER.info("No results found.")
            return None
        # Plot the query result and return as a PIL.Image
        img = plot_query_result(similar, plot_labels=labels)
        return Image.fromarray(img)
    def similarity_index(self, max_dist: float = 0.2, top_k: float = None, force: bool = False) -> Any:  # pd.DataFrame
        """
        Calculate the similarity index of all the images in the table. Here, the index will contain the data points that
        are max_dist or closer to the image in the embedding space at a given index.

        Args:
            max_dist (float): maximum L2 distance between the embeddings to consider. Defaults to 0.2.
            top_k (float): Percentage of the closest data points to consider when counting. Used to apply limit.
                           vector search. Defaults: None.
            force (bool): Whether to overwrite the existing similarity index or not. Defaults to True.

        Returns:
            (pandas.DataFrame): A dataframe containing the similarity index. Each row corresponds to an image,
                and columns include indices of similar images and their respective distances.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            sim_idx = exp.similarity_index()
            ```py
        """
        # 如果表不存在，则抛出值错误异常
        if self.table is None:
            raise ValueError("Table is not created. Please create the table first.")
        # 构建相似性索引表名，包括最大距离和top_k参数
        sim_idx_table_name = f"{self.sim_idx_base_name}_thres_{max_dist}_top_{top_k}".lower()
        # 如果指定的相似性索引表名已经存在且不强制覆盖，则记录日志并返回现有表的 pandas 数据帧
        if sim_idx_table_name in self.connection.table_names() and not force:
            LOGGER.info("Similarity matrix already exists. Reusing it. Pass force=True to overwrite it.")
            return self.connection.open_table(sim_idx_table_name).to_pandas()

        # 如果指定了top_k参数且不在0到1之间，则抛出值错误异常
        if top_k and not (1.0 >= top_k >= 0.0):
            raise ValueError(f"top_k must be between 0.0 and 1.0. Got {top_k}")
        # 如果max_dist小于0，则抛出值错误异常
        if max_dist < 0.0:
            raise ValueError(f"max_dist must be greater than 0. Got {max_dist}")

        # 计算实际的top_k数量，确保不小于1
        top_k = int(top_k * len(self.table)) if top_k else len(self.table)
        top_k = max(top_k, 1)
        # 从表中提取特征向量和图像文件名
        features = self.table.to_lance().to_table(columns=["vector", "im_file"]).to_pydict()
        im_files = features["im_file"]
        embeddings = features["vector"]

        # 创建相似性索引表，使用指定的表名和模式
        sim_table = self.connection.create_table(sim_idx_table_name, schema=get_sim_index_schema(), mode="overwrite")

        def _yield_sim_idx():
            """Generates a dataframe with similarity indices and distances for images."""
            # 使用进度条遍历嵌入向量列表
            for i in tqdm(range(len(embeddings))):
                # 在表中搜索与当前嵌入向量最相似的top_k项，并限制距离小于等于max_dist的项
                sim_idx = self.table.search(embeddings[i]).limit(top_k).to_pandas().query(f"_distance <= {max_dist}")
                # 生成包含相似性索引信息的列表
                yield [
                    {
                        "idx": i,
                        "im_file": im_files[i],
                        "count": len(sim_idx),
                        "sim_im_files": sim_idx["im_file"].tolist(),
                    }
                ]

        # 将相似性索引信息添加到相似性索引表中
        sim_table.add(_yield_sim_idx())
        # 更新对象的相似性索引属性
        self.sim_index = sim_table
        # 返回相似性索引表的 pandas 数据帧
        return sim_table.to_pandas()
    def plot_similarity_index(self, max_dist: float = 0.2, top_k: float = None, force: bool = False) -> Image:
        """
        Plot the similarity index of all the images in the table. Here, the index will contain the data points that are
        max_dist or closer to the image in the embedding space at a given index.

        Args:
            max_dist (float): maximum L2 distance between the embeddings to consider. Defaults to 0.2.
            top_k (float): Percentage of closest data points to consider when counting. Used to apply limit when
                running vector search. Defaults to 0.01.
            force (bool): Whether to overwrite the existing similarity index or not. Defaults to True.

        Returns:
            (PIL.Image): Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()

            similarity_idx_plot = exp.plot_similarity_index()
            similarity_idx_plot.show() # view image preview
            similarity_idx_plot.save('path/to/save/similarity_index_plot.png') # save contents to file
            ```py
        """
        # Retrieve similarity index based on provided parameters
        sim_idx = self.similarity_index(max_dist=max_dist, top_k=top_k, force=force)
        
        # Extract counts of similar images from the similarity index
        sim_count = sim_idx["count"].tolist()
        sim_count = np.array(sim_count)

        # Generate indices for the bar plot
        indices = np.arange(len(sim_count))

        # Create the bar plot using matplotlib
        plt.bar(indices, sim_count)

        # Customize the plot with labels and title
        plt.xlabel("data idx")
        plt.ylabel("Count")
        plt.title("Similarity Count")
        
        # Save the plot to a PNG image in memory
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Use Pillow to open the image from the buffer and return it
        return Image.fromarray(np.array(Image.open(buffer)))


    def _check_imgs_or_idxs(
        self, img: Union[str, np.ndarray, List[str], List[np.ndarray], None], idx: Union[None, int, List[int]]
    ) -> List[np.ndarray]:
        """Determines whether to fetch images or indexes based on provided arguments and returns image paths."""
        # Check if both img and idx are None, which is not allowed
        if img is None and idx is None:
            raise ValueError("Either img or idx must be provided.")
        
        # Check if both img and idx are provided, which is also not allowed
        if img is not None and idx is not None:
            raise ValueError("Only one of img or idx must be provided.")
        
        # If idx is provided, fetch corresponding image paths from the table
        if idx is not None:
            idx = idx if isinstance(idx, list) else [idx]
            img = self.table.to_lance().take(idx, columns=["im_file"]).to_pydict()["im_file"]

        # Return a list of image paths as numpy arrays
        return img if isinstance(img, list) else [img]
    # 定义一个方法，用于向AI提出问题并获取结果
    def ask_ai(self, query):
        """
        Ask AI a question.

        Args:
            query (str): Question to ask.

        Returns:
            (pandas.DataFrame): A dataframe containing filtered results to the SQL query.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            answer = exp.ask_ai('Show images with 1 person and 2 dogs')
            ```py
        """
        # 使用提供的查询字符串调用prompt_sql_query函数，并获取结果
        result = prompt_sql_query(query)
        try:
            # 尝试使用结果调用sql_query方法，返回处理后的数据帧
            return self.sql_query(result)
        except Exception as e:
            # 如果出现异常，记录错误信息到日志，并返回None
            LOGGER.error("AI generated query is not valid. Please try again with a different prompt")
            LOGGER.error(e)
            return None

    # 定义一个方法，用于可视化查询结果，但当前未实现任何功能
    def visualize(self, result):
        """
        Visualize the results of a query. TODO.

        Args:
            result (pyarrow.Table): Table containing the results of a query.
        """
        # 目前这个方法没有实现任何功能，因此pass

    # 定义一个方法，用于生成数据集的报告，但当前未实现任何功能
    def generate_report(self, result):
        """
        Generate a report of the dataset.

        TODO
        """
        # 目前这个方法没有实现任何功能，因此pass
```