# Piper ONNX Runtime

ONNXRuntimeを使用したPiper TTSの実装

## 概要

このプロジェクトは、Piper TTSモデルをONNX Runtimeで実行するスタンドアロンの実装です。依存関係なしでPiper TTSを使用できます。

## 必要条件

- Python >= 3.10
- uv (パッケージ管理)

## インストール

```bash
uv pip install espeakng-loader onnxruntime phonemizer-fork soundfile
```

## モデルファイルの取得

英語モデル (Ryan, Medium) をダウンロード:

```bash
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json
```

## 使用方法

### 基本的な使用例

```python
from standalone_piper import StandalonePiper
import soundfile as sf

# モデルを初期化
piper = StandalonePiper('en_US-ryan-medium.onnx', 'en_US-ryan-medium.onnx.json')

# 音声を生成
samples, sample_rate = piper.create('Hello World, I am Stack Chan!')

# WAVファイルに保存
sf.write('output.wav', samples, sample_rate)
```

### コマンドラインから実行

```bash
uv run ./standalone_piper.py
```

実行例:
```
Piper TTS 音声生成を開始...
テキスト: Hello World, I am Stack Chan!
音声ファイル 'standalone_audio.wav' を生成しました
処理時間:
  音素変換: 1.824秒
  前処理: 0.000秒
  ONNX推論: 0.248秒
  合計: 2.072秒
サンプリングレート: 22050Hz
音声長: 1.49秒
```

## API

### StandalonePiper クラス

#### 初期化

```python
piper = StandalonePiper(model_path, config_path)
```

- `model_path`: ONNXモデルファイルのパス
- `config_path`: 設定JSONファイルのパス

#### メソッド

##### create(text, speaker_id=None, is_phonemes=False, length_scale=None, noise_scale=None, noise_w=None)

テキストから音声を生成します。

**パラメータ:**
- `text`: 音声に変換するテキスト
- `speaker_id`: 話者ID (文字列または数値)
- `is_phonemes`: テキストが音素かどうか
- `length_scale`: 音声の長さスケール
- `noise_scale`: ノイズスケール
- `noise_w`: ノイズ重み

**戻り値:**
- `samples`: 音声データ (numpy配列)
- `sample_rate`: サンプリングレート

##### get_voices()

利用可能な話者を取得します。

**戻り値:**
- 話者IDマップ (辞書)

## 依存関係

- `espeakng-loader>=0.2.4`
- `onnxruntime>=1.20.0`
- `phonemizer-fork>=3.3.2`
- `soundfile>=0.13.1`

## ライセンス

このプロジェクトはPiper TTSプロジェクトに基づいています。
