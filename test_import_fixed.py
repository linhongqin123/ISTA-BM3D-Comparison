import sys
import os

print("测试数据加载器导入...")
print("=" * 50)

# 方法1: 添加src目录到路径
sys.path.insert(0, 'src')

try:
    # 尝试导入
    from utils.data_loader import Set14Loader
    print("✅ 成功导入 Set14Loader")
    
    # 创建实例
    loader = Set14Loader('data/Set14')
    print("✅ 成功创建 Set14Loader 实例")
    
    # 测试加载图像
    print("\n尝试加载图像...")
    images = loader.load_images(as_gray=True)
    
    if images:
        print(f"✅ 成功加载 {len(images)} 张图像")
        print(f"图像名称列表: {list(images.keys())}")
        
        # 查找ppt3
        ppt3_name, ppt3_img = loader.find_ppt3()
        if ppt3_name:
            print(f"✅ 找到关键图像 '{ppt3_name}': {ppt3_img.shape}")
        else:
            print("⚠️  未找到名称中包含'ppt3'的图像")
            
    else:
        print("⚠️  未加载到任何图像")
        print("请检查 data/Set14 目录是否包含图像文件")
        
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("\n可能原因:")
    print("1. 文件路径不正确")
    print("2. utils目录不是Python包（缺少__init__.py）")
    print("3. 文件中有语法错误")
    
    # 检查utils目录是否有__init__.py
    utils_init = "src/utils/__init__.py"
    if os.path.exists(utils_init):
        print(f"✅ {utils_init} 存在")
    else:
        print(f"❌ {utils_init} 不存在，正在创建...")
        open(utils_init, 'w').close()
        print(f"✅ 已创建 {utils_init}")
        
except Exception as e:
    print(f"❌ 其他错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
