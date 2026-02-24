# SoundSpaces 项目

完整的环境配置，包含所有必要的依赖和安装说明。
自取，多了必要的habitat-sim和habitat-lab,当时下载挺费时，也许这样会简单点

**注意：此仓库包含了原始的 [facebookresearch/sound-spaces](https://github.com/facebookresearch/sound-spaces) 以及所有必要的依赖项目。**

这是 SoundSpaces 音频视觉导航项目的完整环境。

## 📁 项目结构

```
soundspaces-project/
├── habitat-sim/        # 3D 模拟器
├── habitat-lab/        # 任务定义和训练框架
├── sound-spaces/       # SoundSpaces 核心代码
└── README.md           # 本文件
```

## 🚀 安装步骤

### 1. 创建 Conda 环境
```bash
conda create -n ss python=3.9 cmake=3.14.0 -y
conda activate ss
```

### 2. 安装 habitat-sim
```bash
cd habitat-sim
git checkout RLRAudioPropagationUpdate
python setup.py install --headless --audio
cd ..
```

### 3. 安装 habitat-lab
```bash
cd habitat-lab
git checkout v0.2.2
pip install -e .
cd ..
```

### 4. 安装 sound-spaces
```bash
cd sound-spaces
pip install -e .
cd ..
```

## 📖 使用说明

详细的安装和使用说明请参考：
- `sound-spaces/INSTALLATION.md`
- `sound-spaces/README.md`

## ⚠️ 注意事项

- **需要 GPU 支持**：habitat-sim 需要 NVIDIA GPU 或配置正确的 headless 模式
- **下载场景数据**：需要下载完整的场景数据集（Replica, Matterport3D 等）
- **VMware 虚拟机**：如果使用 VMware，需要配置 GPU 直通或使用 Docker

## 🔧 当前状态

- ✅ habitat-sim (RLRAudioPropagationUpdate 分支)
- ✅ habitat-lab (v0.2.2 版本)
- ✅ sound-spaces (main 分支)
- ✅ Conda 环境 `ss` 已创建
- ⚠️ 需要下载完整场景数据集
- ⚠️ 需要 GPU 配置

## 📚 参考链接

- [SoundSpaces GitHub](https://github.com/facebookresearch/sound-spaces)
- [habitat-sim GitHub](https://github.com/facebookresearch/habitat-sim)
- [habitat-lab GitHub](https://github.com/facebookresearch/habitat-lab)
