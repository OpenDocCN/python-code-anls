<!--yml
category: 未分类
date: 2023-10-10 23:18:11
-->

# ChatGLM2 源码解析：`GLMTransformer`-CSDN博客

> 来源：[https://blog.csdn.net/wizardforcel/article/details/132686803](https://blog.csdn.net/wizardforcel/article/details/132686803)

![](img/2ea40e9af6ba0cea7ff6349c92adf627.png)

```py
 class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        self.num_layers = config.num_layers

        def build_layer(layer_number):
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm

            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                 dtype=config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):

        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]

        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None

        all_hidden_states = () if output_hidden_states else None

        for index in range(self.num_layers):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )

            hidden_states, kv_cache = layer_ret

            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions 
```