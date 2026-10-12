"""命令行入口。

  python -m stapipe run [VIDEO [OUT_DIR]] --config config.yaml [--set key=value ...]
      标准输出：每个已发布样本的目录（内含 pure/ 与 work/）；日志走标准错误。
      退出码：0 已发布样本，2 skipped（无样本），1 blocked。
  python -m stapipe lock --config config.yaml      # 模板或模型改动后重新写入模板哈希
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .models import refresh_lock
from .pipeline import run, summarize


def main() -> int:
    parser = argparse.ArgumentParser(prog="stapipe")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "lock"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="config.yaml")
    sub.choices["run"].add_argument("video", nargs="?", help="raw video path (absolute, or relative to the "
                                    "current directory); defaults to input_video in the config")
    sub.choices["run"].add_argument("out", nargs="?", help="directory to create results in (created if missing); "
                                    "defaults to output_root in the config, i.e. <workspace>/outputs")
    sub.choices["run"].add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                                    help="override a config value, e.g. --set input_video=inputs/x.mp4")
    args = parser.parse_args()
    if args.command == "lock":
        config = Path(args.config).resolve()
        lock_name = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("models_lock", "models.lock.json")
        lock = refresh_lock(config.parent / lock_name)
        print(f"locked {len(lock['nodes'])} node templates -> {config.parent / lock_name}")
        return 0
    overrides = list(args.set)
    if args.video:
        overrides.append(f"input_video={Path(args.video).resolve()}")
    if args.out:
        overrides.append(f"output_root={Path(args.out).resolve()}")
    result = run(args.config, overrides)
    print(summarize(result), file=sys.stderr)
    for sample in result.get("samples", []):          # 标准输出只有已发布样本的目录，便于脚本取用
        print(sample["path"])
    if result.get("samples"):
        return 0
    return 2 if result["status"] == "skipped" else 1


if __name__ == "__main__":
    sys.exit(main())
