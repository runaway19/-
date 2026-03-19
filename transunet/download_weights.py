import os
import urllib.request
from tqdm import tqdm

# --- 配置区 ---
# 你指定的绝对路径
SAVE_DIR = r"C:\Users\123\.cache\transunet_weights"

# 需要下载的模型名称列表 (你可以根据需要增删)
MODELS_TO_DOWNLOAD = [
    "R50+ViT-B_16",  # <--- 这是你训练主要需要的混合模型
    # "ViT-B_16",    # 如果需要纯 ViT 模型，可以取消注释
    # "ViT-L_16",
]

# Google 官方存放权重的云盘基础地址
BASE_URL = "https://storage.googleapis.com/vit_models/imagenet21k/"


class DownloadProgressBar(tqdm):
    """
    一个带进度条的下载钩子，用于 urllib.request.urlretrieve
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_weights():
    print(f"{'=' * 50}")
    print(f"TransUNet 官方预训练权重下载器")
    print(f"目标保存目录: {SAVE_DIR}")
    print(f"{'=' * 50}\n")

    # 1. 创建目标目录
    if not os.path.exists(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR)
            print(f"已创建目录: {SAVE_DIR}")
        except Exception as e:
            print(f"创建目录失败: {e}")
            return

    # 2. 遍历下载模型
    for model_name in MODELS_TO_DOWNLOAD:
        file_name = f"{model_name}.npz"
        file_path = os.path.join(SAVE_DIR, file_name)
        download_url = f"{BASE_URL}{file_name}"

        print(f"\n检查权重文件: [{file_name}] ...")

        if os.path.exists(file_path):
            print(f"✅ 文件已存在，无需重复下载: {file_path}")
            continue

        print(f"⏳ 本地未找到该文件，准备从 Google 服务器下载...")
        print(f"🔗 下载链接: {download_url}")

        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=file_name) as t:
                urllib.request.urlretrieve(download_url, filename=file_path, reporthook=t.update_to)
            print(f"🎉 下载成功！已保存至: {file_path}")
        except urllib.error.URLError as e:
            print(f"❌ 下载失败，网络连接错误: {e}")
            print("提示: 国内访问 storage.googleapis.com 可能需要开启系统代理 (VPN)。")
            if os.path.exists(file_path):
                os.remove(file_path)  # 删除可能下载残缺的损坏文件
        except Exception as e:
            print(f"❌ 发生未知错误: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)


if __name__ == "__main__":
    download_weights()
    print("\n所有任务执行完毕。请按回车键退出...")
    input()