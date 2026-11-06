# RE-USE 推論実装

`codec_module_time_d4.py`、`generator_SEMamba_time_d4.py`、
`mamba_block2_SEMamba.py`、`stfts.py` および `config.json` は、
NVIDIA NemotronLabs RE-USE の下記リポジトリの固定リビジョンに由来します。

https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/tree/31f82f2efad1095034492f236e8dac9239ba713b

原著作権表示と OpenMDW-1.1 ライセンスは各ファイルおよび同梱 LICENSE を参照。
`generator_SEMamba_time_d4.py` の Hub 自動取得用 mixin を除去し、
`mamba_block2_SEMamba.py` の Mamba を独自の PyTorch 参照実装に置き換えています。
`stfts.py` は ISTFT の出力長を指定できるよう変更しています。
参照実装は公式の高速な CUDA / Triton カーネルではなく、動作速度が異なります。
モデル重みは含まず、明示的にダウンロードする必要があります。
