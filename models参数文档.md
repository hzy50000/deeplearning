# models.py Informer主类参数解读

### `Informer` 类

```python
class Informer(nn.Module):
```

- **定义**：创建一个名为 `Informer` 的类，继承自 PyTorch 的 `nn.Module`，用于构建自定义神经网络模型。

```python
def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
            factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
            dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
            output_attention=False, distil=True, mix=True,
            device=torch.device('cuda:0')):
```

- 初始化方法：定义模型的参数。

  - `enc_in`: 编码器输入的特征维度。
  - `dec_in`: 解码器输入的特征维度。
  - `c_out`: 输出的特征维度。
  - `seq_len`: 输入序列的长度。
  - `label_len`: 标签序列的长度。
  - `out_len`: 输出序列的长度。
  - `factor`: 注意力机制中的因子，默认值为5。
  - `d_model`: 模型的隐藏层维度，默认512。
  - `n_heads`: 多头注意力的头数，默认8。
  - `e_layers`: 编码器的层数，默认3。
  - `d_layers`: 解码器的层数，默认2。
  - `d_ff`: 前馈网络的隐藏层维度，默认512。
  - `dropout`: Dropout率，默认0.0。
  - `attn`: 注意力类型，默认'prob'。
  - `embed`: 嵌入类型，默认'fixed'。
  - `freq`: 频率，默认小时'h'。
  - `activation`: 激活函数，默认'gelu'。
  - `output_attention`: 是否输出注意力，默认False。
  - `distil`: 是否使用蒸馏，默认True。
  - `mix`: 是否混合注意力，默认True。
  - `device`: 设备，默认使用CUDA设备。

```python
super(Informer, self).__init__()
```

- **调用父类初始化**：初始化父类 `nn.Module`。

```python
self.pred_len = out_len
self.attn = attn
self.output_attention = output_attention
```

- 赋值参数：
  - `self.pred_len`: 预测长度，对应输出序列长度。
  - `self.attn`: 注意力类型。
  - `self.output_attention`: 是否输出注意力。

```python
self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
```

- 嵌入层：
  - `self.enc_embedding`: 编码器的嵌入层，将输入特征维度扩展到 `d_model` 维度。
  - `self.dec_embedding`: 解码器的嵌入层，功能类似。

```python
Attn = ProbAttention if attn=='prob' else FullAttention
```

- **选择注意力机制**：根据 `attn` 参数选择 `ProbAttention` 或 `FullAttention`。

```python
self.encoder = Encoder(
    [
        EncoderLayer(
            AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                        d_model, n_heads, mix=False),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        ) for l in range(e_layers)
    ],
    [
        ConvLayer(
            d_model
        ) for l in range(e_layers-1)
    ] if distil else None,
    norm_layer=torch.nn.LayerNorm(d_model)
)
```

- 编码器：
  - 创建 `e_layers` 个 `EncoderLayer`，每个包含一个 `AttentionLayer`，使用选择的注意力机制 `Attn`。
  - 如果 `distil` 为 `True`，在编码器层之间添加 `ConvLayer` 进行蒸馏。
  - 使用 `LayerNorm` 进行归一化。

```python
self.decoder = Decoder(
    [
        DecoderLayer(
            AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                        d_model, n_heads, mix=mix),
            AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                        d_model, n_heads, mix=False),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation,
        )
        for l in range(d_layers)
    ],
    norm_layer=torch.nn.LayerNorm(d_model)
)
```

- 解码器：

  - 创建`d_layers`个`DecoderLayer`，每个包含两个`AttentionLayer`：

    - 第一个使用选择的注意力机制 `Attn`，并根据 `mix` 参数决定是否混合。
    - 第二个使用 `FullAttention`。
  
- 使用 `LayerNorm` 进行归一化。

```python
self.projection = nn.Linear(d_model, c_out, bias=True)
```

- **投影层**：将解码器的输出从 `d_model` 维度投影到 `c_out` 维度，作为最终输出。

```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
```

- 前向传播方法

  ：定义数据流动的过程。

  - `x_enc`: 编码器的输入数据。
  - `x_mark_enc`: 编码器输入的时间标记。
  - `x_dec`: 解码器的输入数据。
  - `x_mark_dec`: 解码器输入的时间标记。
  - `enc_self_mask`: 编码器自注意力的掩码。
  - `dec_self_mask`: 解码器自注意力的掩码。
  - `dec_enc_mask`: 解码器和编码器之间的掩码。

```python
enc_out = self.enc_embedding(x_enc, x_mark_enc)
enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
```

- 编码过程：
  - 将编码器的输入数据和时间标记通过嵌入层，得到嵌入输出 `enc_out`。
  - 将嵌入后的输出传递给编码器，得到编码器的最终输出 `enc_out` 和注意力权重 `attns`。

```python
dec_out = self.dec_embedding(x_dec, x_mark_dec)
dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
dec_out = self.projection(dec_out)
```

- 解码过程：
  - 将解码器的输入数据和时间标记通过嵌入层，得到嵌入输出 `dec_out`。
  - 将嵌入后的输出和编码器的输出传递给解码器，结合相应的掩码，得到解码器的输出 `dec_out`。
  - 通过投影层将解码器的输出维度转换为 `c_out`。

```python
if self.output_attention:
    return dec_out[:, -self.pred_len:, :], attns
else:
    return dec_out[:, -self.pred_len:, :]  # [B, L, D]
```

- 输出：
  - 如果 `output_attention` 为 `True`，返回预测长度的输出和注意力权重。
  - 否则，仅返回预测长度的输出。

### `InformerStack` 类

```python
class InformerStack(nn.Module):
```

- **定义**：创建一个名为 `InformerStack` 的类，继承自 PyTorch 的 `nn.Module`，用于构建堆叠式的 `Informer` 模型。

```python
def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
            factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
            dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
            output_attention=False, distil=True, mix=True,
            device=torch.device('cuda:0')):
```

- **初始化方法**：定义模型的参数，与 `Informer` 类类似，只是 `e_layers` 这里是一个列表 `[3,2,1]`，表示不同编码器堆叠的层数。

```python
super(InformerStack, self).__init__()
```

- **调用父类初始化**：初始化父类 `nn.Module`。

```python
self.pred_len = out_len
self.attn = attn
self.output_attention = output_attention
```

- 赋值参数：
  - `self.pred_len`: 预测长度。
  - `self.attn`: 注意力类型。
  - `self.output_attention`: 是否输出注意力。

```python
self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
```

- 嵌入层：
  - `self.enc_embedding`: 编码器的嵌入层。
  - `self.dec_embedding`: 解码器的嵌入层。

```python
Attn = ProbAttention if attn=='prob' else FullAttention
```

- **选择注意力机制**：与 `Informer` 类相同，根据 `attn` 参数选择 `ProbAttention` 或 `FullAttention`。

```python
inp_lens = list(range(len(e_layers)))  # [0,1,2,...] 可自定义
encoders = [
    Encoder(
        [
            EncoderLayer(
                AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                            d_model, n_heads, mix=False),
                d_model,
                d_ff,
                dropout=dropout,
                activation=activation
            ) for l in range(el)
        ],
        [
            ConvLayer(
                d_model
            ) for l in range(el-1)
        ] if distil else None,
        norm_layer=torch.nn.LayerNorm(d_model)
    ) for el in e_layers]
self.encoder = EncoderStack(encoders, inp_lens)
```

- 编码器堆叠：
  - `inp_lens`: 输入长度列表，用于 `EncoderStack`。
  - 创建多个 `Encoder` 实例，每个 `Encoder` 的层数由 `e_layers` 列表指定。
  - 每个 `Encoder` 包含对应数量的 `EncoderLayer` 和可选的 `ConvLayer`。
  - 使用 `LayerNorm` 进行归一化。
  - 将所有编码器实例堆叠到 `EncoderStack` 中。

```python
self.decoder = Decoder(
    [
        DecoderLayer(
            AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                        d_model, n_heads, mix=mix),
            AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                        d_model, n_heads, mix=False),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation,
        )
        for l in range(d_layers)
    ],
    norm_layer=torch.nn.LayerNorm(d_model)
)
```

- **解码器**：与 `Informer` 类中的解码器定义相同。

```python
self.projection = nn.Linear(d_model, c_out, bias=True)
```

- **投影层**：将解码器的输出维度转换为 `c_out`，作为最终输出。

```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
```

- **前向传播方法**：定义数据流动的过程，与 `Informer` 类类似。

```python
enc_out = self.enc_embedding(x_enc, x_mark_enc)
enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
dec_out = self.dec_embedding(x_dec, x_mark_dec)
dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
dec_out = self.projection(dec_out)
```

- 数据处理：
  - 编码器部分：嵌入 -> 编码。
  - 解码器部分：嵌入 -> 解码 -> 投影。

 

```python
if self.output_attention:
    return dec_out[:, -self.pred_len:, :], attns
else:
    return dec_out[:, -self.pred_len:, :]  # [B, L, D]
```

- 输出：
  - 如果 `output_attention` 为 `True`，返回预测长度的输出和注意力权重。
  - 否则，仅返回预测长度的输出。

### 总结

- **`Informer` 类**：实现了一个基于 Transformer 架构的时间序列预测模型，包含编码器和解码器部分，使用自注意力机制进行特征提取和序列建模。
- **`InformerStack` 类**：在 `Informer` 基础上，通过堆叠多个编码器，实现更深层次的特征提取，适用于更复杂的模型需求。
- 关键组件：
  - **嵌入层 (`DataEmbedding`)**：将输入特征嵌入到高维空间。
  - **注意力机制 (`ProbAttention` 或 `FullAttention`)**：用于捕捉序列中的依赖关系。
  - **编码器和解码器 (`Encoder`, `Decoder`)**：负责特征提取和序列生成。
  - **投影层 (`nn.Linear`)**：将模型输出转换为目标维度。

