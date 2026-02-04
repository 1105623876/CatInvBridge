import os
import random

def split_dataset(root_dir, train_ratio=0.9):
    # --- 调试代码：检查路径是否存在 ---
    if not os.path.exists(root_dir):
        print(f"❌ 错误：路径不存在 -> {root_dir}")
        return

    print(f"🔍 正在扫描路径: {root_dir}")
    
    all_h5_files = []
    # 递归遍历所有子文件夹
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 改进：忽略大小写检查后缀
            if file.lower().endswith(".h5"):
                full_path = os.path.join(root, file)
                all_h5_files.append(full_path)
    
    if len(all_h5_files) == 0:
        print("❓ 警告：没找到任何 .h5 文件。请检查路径下是否真的有文件。")
        # 打印一下扫描到的第一个文件夹名看看
        return

    random.shuffle(all_h5_files)
    split_idx = int(len(all_h5_files) * train_ratio)
    
    train_files = all_h5_files[:split_idx]
    val_files = all_h5_files[split_idx:]
    
    # 获取当前脚本所在目录，确保 txt 存在 ai 文件夹下
    current_dir = os.path.dirname(__file__)
    train_txt = os.path.join(current_dir, "train.txt")
    val_txt = os.path.join(current_dir, "val.txt")

    with open(train_txt, "w", encoding='utf-8') as f:
        f.write("\n".join(train_files))
    with open(val_txt, "w", encoding='utf-8') as f:
        f.write("\n".join(val_files))
        
    print(f"✅ 成功！")
    print(f"总计: {len(all_h5_files)} 个文件")
    print(f"训练集: {len(train_files)} -> {train_txt}")
    print(f"验证集: {len(val_files)} -> {val_txt}")

if __name__ == "__main__":
    # !!! 请务必确认这个路径是你电脑上 H5 文件夹所在的真实路径 !!!
    # 示例：如果是 D:\data\0001\1.h5，这里就写 D:\data
    target_path = r"D:\0_WYW_0\WHU\WHUCAD-lab\CatInvBridge\test_dataset" 
    split_dataset(target_path)