import os
from PIL import Image
import sys

def convert_bmp_to_png(folder_path):
    """
    将指定文件夹中的所有BMP图片转换为PNG格式
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在！")
        return
    
    # 统计变量
    converted_count = 0
    error_count = 0
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为BMP格式
        if filename.lower().endswith('.bmp'):
            bmp_path = os.path.join(folder_path, filename)
            
            # 生成PNG文件名
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(folder_path, png_filename)
            
            try:
                # 打开BMP图片并保存为PNG
                with Image.open(bmp_path) as img:
                    # 确保图片是RGB模式（如果是索引模式需要转换）
                    if img.mode == 'P':
                        img = img.convert('RGB')
                    elif img.mode == '1':
                        img = img.convert('L')
                    elif img.mode == 'RGBA':
                        # 如果是RGBA模式，保持透明度
                        pass
                    elif img.mode == 'CMYK':
                        img = img.convert('RGB')
                    
                    # 保存为PNG格式
                    img.save(png_path, 'PNG')
                
                # 删除原始BMP文件（可选，取消注释以启用）
                os.remove(bmp_path)
                
                print(f"转换成功：{filename} -> {png_filename}")
                converted_count += 1
                
            except Exception as e:
                print(f"转换失败 {filename}: {str(e)}")
                error_count += 1
    
    # 输出统计信息
    print(f"\n转换完成！")
    print(f"成功转换: {converted_count} 个文件")
    print(f"失败: {error_count} 个文件")
    
    if converted_count == 0:
        print("提示：未找到任何BMP格式的图片文件")

def main():
    # 指定要处理的文件夹路径
    folder_path = r"D:\PPT\ISTA-BM3D-Comparison\data\Set14"
    
    # 检查是否提供了自定义路径（通过命令行参数）
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    
    # 询问用户是否要删除原始BMP文件
    print("BMP转PNG转换工具")
    print(f"目标文件夹: {folder_path}")
    print("\n注意：本程序默认只会转换文件，不会删除原始BMP文件。")
    
    delete_original = input("\n是否要删除原始BMP文件？(y/N): ").strip().lower()
    
    # 执行转换
    convert_bmp_to_png(folder_path)
    
    if delete_original == 'y':
        # 如果需要删除原始BMP文件，可以在这里添加删除逻辑
        print("\n提示：如需删除原始BMP文件，请取消脚本中的注释代码后重新运行。")

if __name__ == "__main__":
    main()