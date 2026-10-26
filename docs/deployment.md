# GitHubとCloudflareの公開

- ソース: https://github.com/Lida-V/SoundtSeparator （公開リポジトリ）
- 紹介・ダウンロードページ: https://sound-separator.pages.dev/
- Cloudflare Pagesプロジェクト: `sound-separator`
- 公開ブランチ: `main`
- 公開ディレクトリ: `site/`
- デプロイ方式: Direct Upload。GitHubへのpushだけでは自動デプロイしません。

このページは静的な紹介・配布ページです。AI分離は各利用者が `setup.bat` / `run.bat` で用意したPCで行います。ローカルPCのAPIをインターネットへ公開するトンネルや外部GPUへの接続は作成していません。

## 更新手順

Node.js 22以降と、対象CloudflareアカウントのPagesへの権限が必要です。Pythonアプリを使うだけならNode.jsは不要です。

```powershell
# 必要な場合のみ、Cloudflareの公式認証画面でログインする。
npx --yes wrangler@4.153.0 login --scopes account:read pages:write

# リポジトリのルートで実行。既存プロジェクトへ静的ファイルだけを公開。
npx --yes wrangler@4.153.0 pages deploy site --project-name sound-separator --branch main
```

ビルドコマンドは不要です。ページはHTML/CSS/SVGで構成し、外部CDN・解析スクリプト・ファイルアップロードを使いません。`site/_headers` にはCSP等の応答ヘッダーを設定しています。

APIトークン、OAuth設定、`.env`、`.wrangler/`、`node_modules/` はGitへ含めません。Cloudflareの認証は公式CLIが管理します。パスワードやトークンをREADMEやコマンド引数へ記載しないでください。

## 配布時の改行

GitHubのソースZIPはGit内のバイトを使うため、`*.bat -text` でCRLFを保持しています。Windows起動ファイルを編集した場合は、配布ZIP内でもCRLFであることを確認してください。

`models/dbs_tiger_dnr_config.json` は公式ファイルのSHA-256を照合します。Windowsのcloneでも内容を変えないよう、属性でLFを固定しています。

## 公開対象

GitHubにはアプリ、推論コード、ライセンス、モデル設定、テスト、紹介サイトを保存します。モデル重み・音声・動画・分離結果・仮想環境・認証情報・個人ワークスペースの `AGENTS.md` は含めません。

Cloudflareへアップロードするのは `site/` のみです。GPUモデルやPythonアプリはCloudflareで実行しません。
