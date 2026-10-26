# DBS-TIGER 推論コード

公式 [Dynamic-Band-Splitting](https://github.com/JusperLee/Dynamic-Band-Splitting)
の commit `ff526463a58ee0278e7bd05c4b8bce112b743e34` から、DBS-TIGER
の推論に必要な `models/tiger.py`・`modules/band_split.py`・`modules/dbs.py`
をそのまま同梱しています。取得日: 2026-10-26。

Apache-2.0 の `LICENSE` と、派生元の MIT 表記を含む `THIRD_PARTY.md`
を保持しています。

SoundtSeparator 固有の差分:

- `models/__init__.py` は TIGER のみを公開します。
- `hub.py` は Hub mixin を空のシムに置き換え、認証・学習用依存を不要にします。
- ダウンロードとチャンク推論は `app/dbs_adapter.py` が担当します。
  標準品質は公式と同じ6秒窓・3秒ホップの一様重み overlap-add です。
  6秒以下で3秒を超える入力はそのまま、3秒以下は6秒までゼロ埋めします。

モデルは [JusperLee/DBS-TIGER-DnR](https://huggingface.co/JusperLee/DBS-TIGER-DnR)
の revision `22da7d4cd3ea63f9365181008bff1c0d2ab6fb2b` を使います。
重みはこのリポジトリへ同梱せず、初回選択時にダウンロードします。
入力・出力は44.1kHzモノラルで、出力順は music / sfx / speech です。
