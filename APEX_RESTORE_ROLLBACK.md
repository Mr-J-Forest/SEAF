# Restored APEX rollback record

本次恢复版只新增 `apex_restored_model.py`、一个实验配置和一个独立矩阵，
并在 `model_factory.py` / `config.py` / `train.py` 中增加了 `apex_restored`
的分支。现有 `seaf_model.py`、`model_type=seaf`、正式 SEAF 结果目录和正式矩阵
没有被替换。

## 当前版本的回退方式

继续使用原来的 SEAF 配置即可：

```text
model_type = seaf
configs/experiments/oras5_seaf.json
```

恢复版输出只写入独立 campaign 目录，不会被 SEAF 队列复用。若要彻底移除
恢复版代码，删除 `apex_restored_model.py`、
`configs/experiments/oras5_apex_restored.json`、
`configs/oras5_apex_restored_matrix.json`，然后从三个现有文件中删除
`apex_restored` 的新增分支即可。

本地修改前的两个宿主文件快照保存在：

```text
outputs/apex_restore_rollback/pre_restore/config.py
outputs/apex_restore_rollback/pre_restore/model_factory.py
```

快照仅用于回退本次新增分支，不应覆盖其后用户在这两个文件中的其他修改；
优先使用 `model_type=seaf` 回退模型行为。
