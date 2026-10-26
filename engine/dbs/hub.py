"""ローカルの設定・重みだけを使う推論用 Hub 互換シム。"""

# NOTE (SoundtSeparator patch): Hugging Face Hub のダウンロード・認証機能は
# 使用しない。app/dbs_adapter.py が固定 URL のファイルを検証して読み込む。
# モデル本体の nn.Module 継承・重み名・forward は公式実装を保持する。
HUB_KWARGS = {}


class PyTorchModelHubMixin:
    pass
