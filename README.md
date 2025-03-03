# MatAI_Benchmark
MatAI_Benchmark 是一款面向 **材料学（Materials Science）** 场景的多模型 Benchmark 框架，涵盖了 **图神经网络（GNN）**、**Transformer**、**Diffusion** 等深度学习模型。该项目最初参考了 [shchur/gnn-benchmark](https://github.com/shchur/gnn-benchmark) 的结构与思路，旨在为材料学中的多种前沿模型提供统一的训练、测试与评估环境。

------

## 特性

- **多模型支持**
  - 同时支持 GNN（GCN、GraphSAGE、GAT、GIN 等）、Transformer、Diffusion 等模型结构。
  - 在材料学特定场景（如晶体结构、分子结构等）下，为多种深度学习模型提供统一的接口与示例。
- **材料学场景优化**
  - 提供了材料学数据处理示例，兼容多种材料学数据格式（如 CIF、POSCAR、XYZ 等）。
  - 针对材料结构的特点，支持自定义的图表示方法与评估指标。
- **统一的实验框架**
  - 参考并扩展 shchur/gnn-benchmark 的设计思路，拥有便于管理的脚本与配置文件结构。
  - 通过统一的数据加载、预处理、训练、评估接口，方便研究者快速上手并进行对比实验。
- **易于扩展**
  - 支持往项目中添加新的材料学数据集、模型或评估流程。
  - 保持统一的标准接口，大幅减少学习成本并提升复用性。

------

## 目录结构

```
bash复制编辑MatAI_Benchmark/
│
├── datasets/                # 存放数据相关的脚本、下载及转换工具
│   ├── __init__.py
│   ├── dataset_X/           # 示例数据集X的相关文件
│   ├── dataset_Y/           # 示例数据集Y的相关文件
│   └── ...
│
├── models/                  # 存放模型定义（GNN, Transformer, Diffusion等）
│   ├── __init__.py
│   ├── gcn.py
│   ├── graphsage.py
│   ├── gat.py
│   ├── gin.py
│   ├── transformer.py
│   ├── diffusion_model.py
│   └── ...
│
├── scripts/                 # 脚本：训练、评估、可视化等
│   ├── train.py
│   ├── evaluate.py
│   ├── prepare_data.py
│   └── ...
│
├── utils/                   # 工具函数：数据预处理、评估指标、可视化等
│   ├── __init__.py
│   └── ...
│
├── configs/                 # 配置文件：超参数、运行环境配置等
│   └── default_config.yaml
│
├── requirements.txt         # Python 依赖列表
├── README.md                # 项目说明文档（当前文件）
└── LICENSE                  # 许可证
```

------

## 安装与环境依赖

1. **克隆仓库：**

   ```
   bash复制编辑git clone https://github.com/YourUserName/MatAI_Benchmark.git
   cd MatAI_Benchmark
   ```

2. **安装 Python 依赖：**

   ```
   bash
   
   
   复制编辑
   pip install -r requirements.txt
   ```

   - 如果有其他依赖需求，可修改/扩充 `requirements.txt`。

3. **数据准备：**

   - 根据需要进入 `datasets/` 文件夹，或使用 `scripts/prepare_data.py` 脚本来下载和转换所需的数据集。
   - 如果需要添加自定义数据集，请在 `datasets/` 中创建新子目录并编写相应的处理脚本。

------

## 快速开始

以下示例以 `dataset_X` 为例，展示如何训练一个 **GCN** 模型并进行评估。

1. **准备数据：**

   ```
   bash
   
   
   复制编辑
   python scripts/prepare_data.py --dataset_name dataset_X
   ```

   此脚本会自动下载或转换 `dataset_X` 并放在 `datasets/dataset_X/` 下。

2. **训练模型：**

   ```
   bash复制编辑python scripts/train.py \
       --model gcn \
       --dataset dataset_X \
       --config configs/default_config.yaml \
       --epochs 50 \
       --batch_size 32
   ```

   - 该命令会启动 GCN 模型的训练，具体模型日志与结果会保存在配置文件中指定的输出目录（默认在 `./runs` 或类似路径）。
   - 你可以通过传递更多参数或修改配置文件，来选择其他模型（如 `transformer`、`diffusion_model` 等），或者调整超参数。

3. **评估模型：**

   ```
   bash复制编辑python scripts/evaluate.py \
       --model_path path/to/model_checkpoint.pt \
       --dataset dataset_X
   ```

   此脚本会加载训练好的模型权重，并在验证集/测试集上评估其性能。

------

## 自定义与扩展

1. **添加新的材料学数据集：**
   - 在 `datasets/` 文件夹下新建目录，并编写数据集的下载、转换脚本。
   - 在 `prepare_data.py` 或其他数据预处理脚本中注册新的数据集名称与处理流程。
2. **添加新的模型：**
   - 在 `models/` 文件夹中添加对应的模型文件（如 `my_transformer.py` 或 `my_diffusion.py`）。
   - 在 `train.py`、`evaluate.py` 等脚本中适配该模型的加载和调用逻辑；也可在 `configs/` 中添加相应的配置项。
3. **自定义训练流程和评估指标：**
   - 可在 `scripts/` 文件夹中编写新的训练或评估脚本；也可在 `utils/` 中实现专门的损失函数、评估指标。
   - 针对材料学特定需求（如对晶格常数、能带结构、生成材料的可行性指标等），可以自行定义新的指标或可视化脚本。

------

## 参考与致谢

- **shchur/gnn-benchmark**
  本项目在架构和思想上参考了 [shchur/gnn-benchmark](https://github.com/shchur/gnn-benchmark)。感谢原作者及其社区所做的贡献。
- **材料学开源社区**
  感谢所有为材料学数据开放、模型研究、基准测试做出贡献的开发者与研究者。

------

## 贡献指南

我们非常欢迎社区通过 [Issues](https://github.com/YourUserName/MatAI_Benchmark/issues) 与 [Pull Requests](https://github.com/YourUserName/MatAI_Benchmark/pulls) 提出问题、提交改进或贡献新的数据集、模型和功能。任何疑问与建议都可以在 issue 中讨论。

请在提交 Pull Request 前，确保遵循以下流程：

1. **代码规范**：遵从 Python 通用编码风格（PEP 8）及项目已有的代码结构。
2. **单元测试**：若修复 bug 或新增功能，尽可能编写或更新测试脚本，以确保稳定性和可复现性。
3. **文档更新**：完善或更新相应的文档、示例，便于他人理解和使用。

------

## 许可证

本项目在 MIT 协议下开源，请在使用本项目代码或二次开发时保留原作者版权信息。

如果本项目对你的研究或产品有所帮助，欢迎在论文或引用中注明来源！祝大家在材料学与深度学习研究和应用中取得更多成果！
