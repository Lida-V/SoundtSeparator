"""エージェント用スキル（soundsep）を、ユーザーのスキルフォルダへ導入する。MCP を使わない導入方法。

このフォルダを開いたエージェントは .claude/skills/soundsep/ をそのまま読める。
別のフォルダで作業するエージェントから使うときに、これを実行する。
スキルの本文に、このフォルダの場所を書き込んでコピーする（スキル1ファイルだけを書く）。

  .venv/Scripts/python.exe -X utf8 tools/install_skill.py                  → ~/.claude/skills/soundsep/
  .venv/Scripts/python.exe -X utf8 tools/install_skill.py --agent codex    → ~/.codex/skills/soundsep/
  .venv/Scripts/python.exe -X utf8 tools/install_skill.py --dest <スキルフォルダ>
  .venv/Scripts/python.exe -X utf8 tools/install_skill.py --remove         （導入したスキルを消す）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
SOURCE = APP_ROOT / ".claude" / "skills" / "soundsep" / "SKILL.md"
ROOT_LINE = '$root = "."'
MARK = "<!-- installed by tools/install_skill.py -->"
AGENT_DIRS = {"claude": Path.home() / ".claude" / "skills", "codex": Path.home() / ".codex" / "skills"}


def render(root=APP_ROOT):
    """スキル本文の `$root = "."` を、このフォルダの絶対パスに置き換える。"""
    text = SOURCE.read_text(encoding="utf-8")
    if text.count(ROOT_LINE) != 1:
        raise SystemExit(f"スキルの書式が想定と違います（{ROOT_LINE} が1か所だけ必要）: {SOURCE}")
    return text.replace(ROOT_LINE, f'$root = "{root}"').rstrip("\n") + f"\n\n{MARK}\n"


def install(skills_dir, force=False):
    target = Path(skills_dir) / "soundsep" / "SKILL.md"
    if target.exists() and not force and MARK not in target.read_text(encoding="utf-8", errors="replace"):
        raise SystemExit(f"別のスキルが既にあります（上書きしません。置き換えるなら --force）: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(), encoding="utf-8", newline="\n")
    return target


def remove(skills_dir):
    target = Path(skills_dir) / "soundsep" / "SKILL.md"
    if not target.exists():
        return None
    if MARK not in target.read_text(encoding="utf-8", errors="replace"):
        raise SystemExit(f"このスクリプトが導入したスキルではありません（消しません）: {target}")
    target.unlink()
    if not any(target.parent.iterdir()):
        target.parent.rmdir()
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=sorted(AGENT_DIRS), default="claude")
    parser.add_argument("--dest", help="スキルフォルダを直接指定する（--agent より優先）")
    parser.add_argument("--force", action="store_true", help="同名の別のスキルがあっても置き換える")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    skills_dir = Path(args.dest) if args.dest else AGENT_DIRS[args.agent]
    if args.remove:
        done = remove(skills_dir)
        print(f"削除しました: {done}" if done else "導入されていません")
    else:
        print(f"導入しました: {install(skills_dir, args.force)}")
        print("新しいセッションから soundsep スキルとして使えます。このフォルダを移動したら、もう一度実行してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
