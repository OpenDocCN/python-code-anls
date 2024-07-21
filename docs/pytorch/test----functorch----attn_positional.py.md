# `.\pytorch\test\functorch\attn_positional.py`

```
        # 初始化函数，定义了 BertSelfAttention 类，继承自 nn.Module
        def __init__(
            self,
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            position_embedding_type=None,
            max_position_embeddings=None,
        ):
            super().__init__()
            # 检查隐藏大小是否是注意力头数的整数倍
            if hidden_size % num_attention_heads != 0:
                raise ValueError(
                    f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                    f"heads ({num_attention_heads})"
                )

            # 设置注意力头数和每个头的大小
            self.num_attention_heads = num_attention_heads
            self.attention_head_size = int(hidden_size / num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            # 定义查询、键、值的线性层
            self.query = nn.Linear(hidden_size, self.all_head_size)
            self.key = nn.Linear(hidden_size, self.all_head_size)
            self.value = nn.Linear(hidden_size, self.all_head_size)

            # 定义 dropout 层
            self.dropout = nn.Dropout(attention_probs_dropout_prob)
            self.position_embedding_type = position_embedding_type

            # 如果指定了位置嵌入类型，则初始化位置嵌入相关的组件
            if self.position_embedding_type is not None:
                assert max_position_embeddings is not None
                self.max_position_embeddings = max_position_embeddings
                # 使用 nn.Embedding 初始化距离嵌入
                self.distance_embedding = nn.Embedding(
                    2 * max_position_embeddings - 1, self.attention_head_size
                )

        def transpose_for_scores(self, x):
            # 改变 x 的形状以便进行注意力计算
            new_x_shape = x.size()[:-1] + (
                self.num_attention_heads,
                self.attention_head_size,
            )
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)

        def forward(
            self,
            hidden_states,
            past_key_value=None,
        ):
            # Compute query, key, and value matrices from hidden_states using learned weights
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)

            # Transpose q, k, and v matrices for subsequent matrix multiplication
            q = self.transpose_for_scores(q)
            k = self.transpose_for_scores(k)
            v = self.transpose_for_scores(v)

            # Concatenate past_key_value with k and v if past_key_value is provided
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)

            # Calculate attention scores as dot product of q and transposed k
            attention_scores = torch.matmul(q, k.transpose(-1, -2))
            # Scale attention_scores by the square root of the attention head size
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # Calculate positional embeddings if position_embedding_type is specified
            if self.position_embedding_type is not None:
                seq_length = hidden_states.size()[1]
                # Generate left and right position ids
                position_ids_l = torch.arange(
                    seq_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
                position_ids_r = torch.arange(
                    seq_length, dtype=torch.long, device=hidden_states.device
                ).view(1, -1)
                # Calculate distance between position_ids_l and position_ids_r
                distance = position_ids_l - position_ids_r
                # Get positional embeddings based on distance and max_position_embeddings
                positional_embedding = self.distance_embedding(
                    distance + self.max_position_embeddings - 1
                )
                positional_embedding = positional_embedding.to(
                    dtype=q.dtype
                )  # Ensure dtype compatibility

                # Add relative position scores to attention_scores based on position_embedding_type
                if self.position_embedding_type == "relative_key":
                    relative_position_scores = torch.einsum(
                        "bhld,lrd->bhlr", q, positional_embedding
                    )
                    attention_scores = attention_scores + relative_position_scores
                elif self.position_embedding_type == "relative_key_query":
                    relative_position_scores_query = torch.einsum(
                        "bhld,lrd->bhlr", q, positional_embedding
                    )
                    relative_position_scores_key = torch.einsum(
                        "bhrd,lrd->bhlr", k, positional_embedding
                    )
                    attention_scores = (
                        attention_scores
                        + relative_position_scores_query
                        + relative_position_scores_key
                    )

            # Softmax normalization of attention_scores to get attention_probs
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            # Apply dropout to attention_probs
            attention_probs = self.dropout(attention_probs)

            # Compute context_layer by matrix multiplication of attention_probs and v
            context_layer = torch.matmul(attention_probs, v)

            # Transpose and reshape context_layer for output
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            return context_layer
```