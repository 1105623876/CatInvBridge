import torch
import torch.nn as nn

class CatInvTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, input_dim=33, max_seq_len=100):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 改进：用随机初始化的位置编码，或者 Sine/Cosine PE
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Linear(d_model, input_dim)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (Batch, SeqLen, 33)
        src_key_padding_mask: (Batch, SeqLen) 布尔值，True 表示该位置是填充，不参与计算
        """
        b, s, d = x.shape
        # 投影并加上位置信息
        x = self.input_projection(x) + self.pos_emb[:, :s, :]
        
        # 传入 mask，让 Transformer 忽略填充位
        latent = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        out = self.output_head(latent)
        return out