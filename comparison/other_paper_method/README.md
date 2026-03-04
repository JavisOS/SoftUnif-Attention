# Other-Paper Method File Group

这个目录用于集中列出“他人论文方法”相关文件，方便你做对比实验以及后续一键删除。

## 说明

- `code/`：他人论文方法的全部代码（已从 `clutrr/` 下迁出）。
- `code/scl/`：他人方法相关的 SCL 逻辑规则文件（已从 `clutrr/scl/` 迁出）。
- `FILES.txt`：`code/` 中主要脚本清单。
- `remove_other_method_files.sh`：按清单删除这些文件（可选）。

## TSRA 可独立运行说明

TSRA 训练主入口不依赖这里列出的文件，主命令：

- `python -m clutrr.cli.train ...`

RuleTaker 主入口：

- `python -m ruletaker.cli.train ...`
