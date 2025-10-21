import os
import glob
import shutil

# --- 1. 定义源和目标路径 ---

# 包含 npz 文件的源文件夹 (原音频文件所在的文件夹)
SOURCE_DIR = r"C:\Users\肖梓涵\Desktop\全组音频_未处理\新建文件夹"

# 目标文件夹 (exp2 预处理结果的集中地)
DESTINATION_DIR = r"C:\Users\肖梓涵\Desktop\exp2预处理"


def move_npz_files():
    """
    搜索源文件夹中的所有 .npz 文件，并将其剪切到目标文件夹。
    """
    
    # 检查源文件夹是否存在
    if not os.path.isdir(SOURCE_DIR):
        print(f"错误：源文件夹不存在 -> {SOURCE_DIR}")
        return

    # 检查并创建目标文件夹
    if not os.path.exists(DESTINATION_DIR):
        print(f"目标文件夹 {DESTINATION_DIR} 不存在，正在创建...")
        try:
            os.makedirs(DESTINATION_DIR)
            print("创建成功。")
        except Exception as e:
            print(f"错误：无法创建目标文件夹 -> {e}")
            return
    
    print("-" * 40)
    print(f"正在搜索源文件夹: {SOURCE_DIR}")
    
    # 2. 搜索所有 .npz 文件
    # 使用 os.path.join 确保路径兼容性，使用 glob 查找文件
    search_pattern = os.path.join(SOURCE_DIR, "*.npz")
    npz_files = glob.glob(search_pattern)
    
    if not npz_files:
        print("未找到任何 .npz 文件，无需移动。")
        return
    
    print(f"找到 {len(npz_files)} 个 .npz 文件。")
    print(f"开始移动到目标文件夹: {DESTINATION_DIR}")
    print("-" * 40)
    
    move_count = 0
    for file_path in npz_files:
        # 3. 确定目标路径
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(DESTINATION_DIR, file_name)
        
        # 4. 执行剪切操作 (shutil.move)
        try:
            # move 函数实现了剪切/重命名操作
            shutil.move(file_path, destination_path)
            print(f"成功移动: {file_name}")
            move_count += 1
        except Exception as e:
            print(f"错误：移动文件 {file_name} 失败: {e}")
            
    print("-" * 40)
    print(f"操作完成！成功移动 {move_count} 个文件。")


if __name__ == "__main__":
    move_npz_files()