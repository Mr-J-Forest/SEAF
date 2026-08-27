# APEX 实验执行计划

正式研究问题是：APEX 能否学习 anomaly persistence 未能描述的可预测海洋异常演化？

## 冻结模型

APEX 仅由固定 AP skip、低频谱编码器、残差成员与空间 ensemble gate 组成。输入包括 TEMP/SALT 的因果一步 tendency 和配置中声明的外部动力场。代码中不保留 thermohaline memory、独立 3-D branch、fusion transformer、global token bank 或 local parallel branch。

## 冻结消融

`configs/oras5_ablation_matrix.json` 预声明七个实验：

1. full APEX；
2. no AP residual；
3. direct full field；
4. no tendency；
5. no external dynamics；
6. no spectral branch；
7. no ensemble gate。

所有模型联合预测 TEMP/SALT，不训练或报告正式的单变量版本。筛选和消融只使用 validation；最终配置冻结后才运行 test。

## 必须报告

- Climatology、Persistence、Anomaly Persistence、Damped Anomaly Persistence；
- FourCastNet/AFNO、ClimaX、Swin comparison adapters；
- `SS_AP = 1 - MSE_model / MSE_AP`；
- TEMP/SALT × lead × depth，以及可行时的 season/region 分解；
- 消融的配对 forecast-origin bootstrap 与预声明 contrast。

现有旧架构权重和结果不得作为 APEX 结果复用；正式 APEX 代码同步后需要重新训练。
