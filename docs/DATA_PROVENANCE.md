# 数据集全流程溯源文档 (DATA PROVENANCE)

> 本文档完整记录 MetalScan 四分类训练集 415 张图片的来源、筛选、映射与校验流程，确保数据可复现、可追溯。

## 1. 数据来源

| 项目 | 值 |
|------|-----|
| 镜像仓库 | GitHub `gg123654/all-Dataset`（main 分支） |
| 子数据集 1 | `NEU-DET/`（东北大学热轧钢带缺陷库，1620 张，6 类） |
| 子数据集 2 | `GC10-DET/`（钢材表面缺陷库，2001 张，7 类） |
| 下载方式 | GitHub REST API 获取文件树 (`git/trees?recursive=1`) → raw.githubusercontent.com 并发下载 |
| 下载规模 | 约 3000 个文件（图片 + YOLO 格式标注） |

原始数据集论文：
- NEU-DET: K. Song, Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*, 2013.
- GC10-DET: Lv et al., "Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network," *arXiv:1906.06475*, 2019.

## 2. 类别映射规则

| 来源 | 原始类别 | 标注 ID (原始) | → 本系统类别 | 新标注 ID |
|------|----------|---------------|-------------|-----------|
| NEU-DET | scratches | 5 | scratch | 0 |
| GC10-DET | crescent_gap | 0 | dent | 1 |
| GC10-DET | punching | 2 | dent | 1 |
| GC10-DET | oil_spot | 1 | stain | 2 |
| GC10-DET | water_spot | 5 | stain | 2 |
| 双数据集 | 缺陷框外无缺陷区域裁剪 | — | normal_part | 3（空标签/负样本） |

## 3. 筛选与校验流程（防错关键点）

1. **标注内容级筛选（非文件名）**
   - ⚠️ 该镜像仓库 GC10 部分的文件名类别码与标注框内容**不一致**（上传前被重排）。
   - 因此先下载全部 2001 个标注文件（每个 <1KB），按**标注文件中的真实类别 ID** 筛选候选图，再按需下载对应图片。
2. **可视化人工抽检**
   - 将标注框叠加到原图（红框 + 类别 ID 黄字），对 dent（id0/id2）与 stain（id1/id5）样本逐一目视确认缺陷类型相符；
   - 抽检确认：id0 = 月牙缺口 ✓、id5 = 水渍条痕 ✓；初选中标注不符的样本已剔除。
3. **图像完整性校验**
   - 每张入选图片经 PIL `Image.open().verify()` 校验，损坏文件丢弃。
4. **标签合法性校验**
   - 仅保留含目标类别标注框的图片；框坐标为 YOLO 归一化格式（cx, cy, w, h），非法行丢弃。

## 4. normal_part（正常件）生成规则

- 从 GC10 / NEU 源图（含标注框）的**缺陷框以外的区域**随机裁剪：
  - 裁剪边长 200~420px（GC10 大图，每图最多 6 块）或 90~150px（NEU 200×200 小图，每图最多 4 块）；
  - 与任何缺陷框保持 ≥12px 间距，避免负样本混入缺陷；
  - 随机种子 `seed=42`，结果可复现。
- 共生成 170 张，标注为空文件（YOLO 负样本标准做法）。

## 5. 最终数据集构成

| 类别 | 数量 | 真实来源 |
|------|------|----------|
| scratch | 85 | NEU-DET scratches |
| dent | 80 | GC10-DET crescent_gap + punching |
| stain | 80 | GC10-DET oil_spot + water_spot |
| normal_part | 170 | 双数据集无缺陷区域裁剪 |
| **合计** | **415** | 缺陷 245 + 正常 170 |

## 6. train/val 划分

- 比例 8:2，逐类分层随机划分，`seed=42`；
- 划分后文件名加类前缀防止跨类重名；
- `dataset/data.yaml` 指向划分后目录。

## 7. 已知局限

1. 两个来源均为**钢材表面**灰度/近灰度图像；若需强调"多材质"，建议补拍铝件/塑料件各 20~30 张；
2. NEU 图为 200×200 低分辨率，训练时依赖 YOLO 内置增强提升多样性；
3. `stain` 类 AP50 (0.769) 相对较低，与油斑/水斑形态多变有关，可通过扩充该类样本改善。
