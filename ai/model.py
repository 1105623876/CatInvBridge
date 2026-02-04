import torch
import torch.nn as nn

class CatInvTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, input_dim=33):
        super().__init__()
        
        # 1. 特征嵌入 (Embedding)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. 位置编码 (Positional Encoding)
        self.pos_emb = nn.Parameter(torch.zeros(1, 100, d_model))
        
        # 3. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 回归头 (输出同样维度的参数，但要更精确)
        self.output_head = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (Batch, SeqLen, 33)
        b, s, d = x.shape
        x = self.input_projection(x) + self.pos_emb[:, :s, :]
        
        # 通过 Transformer 提取上下文语义
        latent = self.transformer_encoder(x)
        
        # 预测精细化后的参数
        out = self.output_head(latent)
        return out