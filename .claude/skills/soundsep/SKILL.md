---
name: soundsep
description: Sound Separator の分離エンジンを、画面もMCPも使わずコマンドで呼ぶ。曲の6ステム分割、ボーカル抽出、話し声/効果音/BGM分け、リード/バック分け、ドラムの細分化、ステムからのMIDI採譜。「ステム分割して」「ボーカルを抜いて」「Sound Separatorで分けて」「ステムをMIDIにして」と頼まれたら使う。
---

# Sound Separator をコマンドで使う

Sound Separator のフォルダにある `tools/soundsep_control.py` を、同じフォルダの venv の Python で呼ぶ。サーバー（`run.bat`）の起動もMCPの登録も要らない。結果はJSONで標準出力に出る。進捗は標準エラーに出る。

## 呼び出し方

```powershell
$root = "."   # Sound Separator のフォルダ
$py  = "$root\.venv\Scripts\python.exe"
$cli = "$root\tools\soundsep_control.py"
& $py -X utf8 $cli status
& $py -X utf8 $cli separate --input <曲.wav> --out <新しいフォルダ> --mode music --quality high --exact
& $py -X utf8 $cli subseparate --input <vocals.wav> --out <新しいフォルダ> --kind vocals
& $py -X utf8 $cli midi --input <bass.wav> --out <bass.mid>
```

bash から呼ぶときも同じ2つのパスを使う（`"<フォルダ>/.venv/Scripts/python.exe" -X utf8 "<フォルダ>/tools/soundsep_control.py" status`）。

| 処理 | 出力 |
|---|---|
| `separate --mode music` | `vocals` `drums` `bass` `guitar` `piano` `other` |
| `separate --mode vocals` | `vocals` `instrumental` |
| `separate --mode dme` | `dialog` `effect` `music`（映像の音向け） |
| `separate --mode mega --stems ...` | 53カテゴリから選んだもの（カテゴリは重なる） |
| `subseparate --kind vocals` | `lead` `back` |
| `subseparate --kind drums` | `kick` `snare` `cymbals` `toms` |
| `midi` | basic-pitch の採譜結果（`.mid`） |

`--quality` は `fast` / `standard` / `high`。出力は入力と同じサンプルレート・長さの32bit float WAVと `manifest.json`。同名のファイルがあれば上書きせずに止まるので、出力先には新しいフォルダを指定する。

## 進め方

1. `status` でGPUの空きとモデルの取得状況を見る。`.venv` が無ければ、ユーザーに `setup.bat` の実行を頼む。
2. 必要なモデルが `false` なら実行しない。CLIはモデルを取得しない。ユーザーに、アプリ（`run.bat`）の「AIモデルの管理」で取得してもらう。
3. 曲1本の分離は数分かかる。コマンドの制限時間を長めに取るか、バックグラウンドで実行して終了を待つ。分離は1本ずつ実行する（GPUメモリを取り合う）。
4. ミックスや差し替えに使う分割は `--exact` を付ける。モデルは44.1kHzで動くので、48kHzの曲は全ステムの和が元の音からずれる。`--exact` は「元の音 − 全ステムの和」を `other` に足し、和を元の音に一致させる。`music` で全ステムを出すときだけ使える。ずれの大きさは `manifest.json` の `sum_error_before` に残る。
5. ボーカルをリードとバックに分けたら、`manifest.json` の `window_rms`（5秒窓の音量）を読む。`lead` と `back` の割り当ては曲と区間で入れ替わる。ファイル名だけで決めない。
6. `midi` は和音も拾うが、倍音やにじみを音符にする。採譜結果をそのまま正解として扱わない。
7. 終わったら `manifest.json` の数値（和のずれ、各ステムのピークとRMS、所要時間）を報告に入れる。聴いて確かめていないことは、そう書く。

## してはいけないこと

- `audio-separator` を入れない（torchをCPU版へ置き換えようとする）。venv、`models/`、`engine/` を書き換えない。
- 分類名を楽器の判定として扱わない。`guitar` や `piano` には別の楽器が入ることがある。
- 分離した音を、元の曲の権利と切り離して扱わない。
