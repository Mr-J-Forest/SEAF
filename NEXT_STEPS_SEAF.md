# SEAF 实验执行计划

正式研究问题是：低频谱建模、空间 ensemble gate、因果 tendency 和外部动力信息能否直接预测未来海洋温盐异常？

## 冻结模型

SEAF 直接输出未来 TEMP/SALT anomaly。正式网络仅包含低频谱编码器、联合异常预测成员和空间 ensemble gate；输入固定包含因果 tendency 与外部动力场。模型中不存在 AP skip、persistence projection 或 persistence scale。

## 冻结消融

`configs/oras5_ablation_matrix.json` 预声明六个实验：

1. full SEAF；
2. direct full field；
3. no tendency；
4. no external dynamics；
5. no spectral branch；
6. no ensemble gate。

原 `no_ap_residual` 已成为 full SEAF，不再作为消融。所有模型联合预测 TEMP/SALT；筛选和消融只使用 validation，最终配置冻结后才运行 test。

## 结果有效性

旧 `no_ap_residual` 运行与 SEAF 主模型的计算图一致，可作为历史 SEAF screen 结果。旧 full/no-spectral/no-ensemble 和三个 learned baseline 均包含 AP skip，不能作为新 SEAF 证据复用；正式组件消融与 learned baselines 必须在 direct-anomaly 协议下重跑。

## 必须报告

- Climatology、Persistence、Anomaly Persistence、Damped Anomaly Persistence；
- 使用同一 direct-anomaly 目标的 FourCastNet/AFNO、ClimaX、Swin comparison adapters；
- `SS_AP = 1 - MSE_model / MSE_AP`；
- TEMP/SALT × lead × depth，以及可行时的 season/region 分解；
- 新 SEAF 消融的配对 forecast-origin bootstrap 与预声明 contrast。
