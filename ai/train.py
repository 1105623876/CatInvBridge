import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler # 导入混合精度工具

from model import CatInvTransformer
from data_loader import CADRefineDataset

# --- 强制检查 GPU ---
if not torch.cuda.is_available():
    print("❌ 错误: 未检测到 GPU！请检查 PyTorch 是否安装了 CUDA 版本。")
    print("提示: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    exit()

DEVICE = torch.device("cuda")
print(f"✅ 成功识别显卡: {torch.cuda.get_device_name(0)}")

# --- 配置参数 ---
BATCH_SIZE = 128   # 有了 4060 和 AMP，Batch 可以调大一点
LR = 1e-4
EPOCHS = 100
MAX_SEQ_LEN = 100
INPUT_DIM = 33
SAVE_DIR = "checkpoints"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def train():
    # 1. 准备数据
    train_ds = CADRefineDataset("train.txt", max_len=MAX_SEQ_LEN)
    val_ds = CADRefineDataset("val.txt", max_len=MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 2. 初始化模型和优化器
    model = CatInvTransformer(input_dim=INPUT_DIM, max_seq_len=MAX_SEQ_LEN).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2) # 使用 AdamW
    
    # 3. 损失函数和混合精度缩放器
    criterion = nn.MSELoss(reduction='none')
    scaler = GradScaler() # 自动缩放梯度，防止 FP16 溢出

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            inputs = batch["input"].to(DEVICE)
            targets = batch["target"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # --- 核心优化：使用 autocast 进行混合精度前向传播 ---
            with autocast():
                outputs = model(inputs, src_key_padding_mask=masks)
                
                # 计算有效位的 Loss
                active_mask = (~masks).float() # (B, S)
                loss_matrix = criterion(outputs, targets).mean(dim=-1) # (B, S)
                loss = (loss_matrix * active_mask).sum() / active_mask.sum()
            
            # --- 核心优化：梯度缩放和反向传播 ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(DEVICE)
                targets = batch["target"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                
                with autocast():
                    outputs = model(inputs, src_key_padding_mask=masks)
                    active_mask = (~masks).float()
                    loss_matrix = criterion(outputs, targets).mean(dim=-1)
                    loss = (loss_matrix * active_mask).sum() / active_mask.sum()
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"✨ Epoch {epoch+1} 完成! Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 4. 自动保存最优模型
        if (epoch + 1) % 10 == 0:
            save_path = f"{SAVE_DIR}/catinv_model_e{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 模型已保存至 {save_path}")

if __name__ == "__main__":
    train()