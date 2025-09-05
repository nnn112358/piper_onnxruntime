"""
Standalone Piper TTS implementation without piper_onnx dependency

    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json

Setup:
    uv pip install numpy phonemizer onnxruntime soundfile espeakng-loader

Usage:
    uv run python standalone_piper.py
"""

import numpy as np
from numpy.typing import NDArray
import json
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize
import espeakng_loader
import onnxruntime as ort
import soundfile as sf
import time


_BOS = "^"
_EOS = "$"
_PAD = "_"


class StandalonePiper:
    """Piper TTSモデルを使用してテキストから音声を生成するクラス"""
    def __init__(
            self, 
            model_path: str, 
            config_path: str,
        ):
        """Piper TTSインスタンスを初期化
        
        Args:
            model_path: ONNXモデルファイルのパス
            config_path: 設定JSONファイルのパス
        """
        self.setup(model_path, config_path)
        
    def setup(self, model_path, config_path, session=None):
        """モデルとeSpeak設定を初期化
        
        Args:
            model_path: ONNXモデルファイルのパス
            config_path: 設定JSONファイルのパス  
            session: 既存のONNXセッション（オプション）
        """
        # 設定ファイルを読み込み
        with open(config_path) as fp:
            self.config: dict = json.load(fp)
        self.sample_rate: int = self.config['audio']['sample_rate']
        self.phoneme_id_map: dict = self.config['phoneme_id_map']
        self._voices: dict = self.config.get('speaker_id_map')

        # eSpeakライブラリとデータパスを設定
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        
        # ONNX Runtimeセッションを初期化
        self.sess = session or ort.InferenceSession(
            model_path, 
            sess_options=ort.SessionOptions(),
            providers=['CPUExecutionProvider']
        )
        self.sess_inputs_names = [i.name for i in self.sess.get_inputs()]

    @classmethod
    def from_session(
        cls,
        session: ort.InferenceSession,
        config_path: str,
    ):
        """既存のONNXセッションからインスタンスを作成
        
        Args:
            session: 既存のONNX Runtimeセッション
            config_path: 設定JSONファイルのパス
            
        Returns:
            StandalonePiperインスタンス
        """
        instance = cls.__new__(cls)
        instance.setup(model_path='', config_path=config_path, session=session)
        return instance

    def create(
            self, 
            text: str, 
            speaker_id: str | int = None, 
            is_phonemes=False,
            length_scale: int = None,
            noise_scale: int = None,
            noise_w: int = None,
        ) -> tuple[NDArray[np.float32], int, dict]:
        """テキストから音声を生成
        
        Args:
            text: 音声に変換するテキスト
            speaker_id: 話者ID（文字列または数値）
            is_phonemes: テキストが音素表記かどうか
            length_scale: 音声の長さスケール
            noise_scale: ノイズスケール
            noise_w: ノイズ重み
            
        Returns:
            samples: 音声データ（numpy配列）
            sample_rate: サンプリングレート
            timing_info: 処理時間の詳細情報
        """
        start_time = time.time()
        timing_info = {}
        
        # 推論設定を取得
        inference_cfg = self.config['inference']
        length_scale = length_scale or inference_cfg['length_scale']
        noise_scale = noise_scale or inference_cfg['noise_scale']
        noise_w = noise_w or inference_cfg['noise_w']

        # 話者IDを設定
        sid = 0
        if isinstance(speaker_id, str) and speaker_id in self._voices:
            sid = self._voices[speaker_id]
        elif isinstance(speaker_id, int):
            sid = speaker_id
        
        # テキストを音素に変換
        phonemize_start = time.time()
        phonemes = text if is_phonemes else phonemize(text)
        phonemes = list(phonemes)
        phonemes.insert(0, _BOS)  # 開始記号を追加
        timing_info['phonemize_time'] = time.time() - phonemize_start

        # 音素をIDに変換
        preprocess_start = time.time()
        ids = self._phoneme_to_ids(phonemes)
        inputs = self._create_input(ids, length_scale, noise_w, noise_scale, sid)
        timing_info['preprocess_time'] = time.time() - preprocess_start
        
        # ONNX推論を実行
        inference_start = time.time()
        samples = self.sess.run(None, inputs)[0].squeeze((0,1)).squeeze()
        timing_info['inference_time'] = time.time() - inference_start
        
        timing_info['total_time'] = time.time() - start_time
        return samples, self.sample_rate, timing_info
    
    def get_voices(self) -> dict | None:
        """利用可能な話者リストを取得
        
        Returns:
            話者IDマップの辞書（話者名 -> ID）
        """
        return self._voices

    def _phoneme_to_ids(self, phonemes: str) -> list[int]:
        """音素リストを数値IDに変換
        
        Args:
            phonemes: 音素のリスト
            
        Returns:
            音素IDのリスト
        """
        ids = []
        for p in phonemes:
            if p in self.phoneme_id_map:
                ids.extend(self.phoneme_id_map[p])  # 音素IDを追加
                ids.extend(self.phoneme_id_map[_PAD])  # パディング記号を追加
        ids.extend(self.phoneme_id_map[_EOS])  # 終了記号を追加
        return ids
    
    def _create_input(self, ids: list[int], length_scale: int, noise_w: int, noise_scale: int, sid: int) -> dict:
        """ONNX推論用の入力データを作成
        
        Args:
            ids: 音素IDのリスト
            length_scale: 音声の長さスケール
            noise_w: ノイズ重み
            noise_scale: ノイズスケール
            sid: 話者ID
            
        Returns:
            ONNX推論用の入力辞書
        """
        # 音素IDを配列に変換してバッチ次元を追加
        ids = np.expand_dims(np.array(ids, dtype=np.int64), 0)
        length = np.array([ids.shape[1]], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)
        
        # 話者IDを配列に変換
        sid = np.array([sid], dtype=np.int64) if sid is not None else None
        input = {
            'input': ids,
            'input_lengths': length,
            'scales': scales,
        }
        # モデルが話者IDをサポートしている場合は追加
        if 'sid' in self.sess_inputs_names:
            input['sid'] = sid
        return input


if __name__ == "__main__":
    # 使用例（モデルファイルが必要）
    print("Piper TTS 音声生成を開始...")
    piper = StandalonePiper('en_US-ryan-medium.onnx', 'en_US-ryan-medium.onnx.json')
    
    text = 'Hello World, I am Stack Chan!'
    print(f"テキスト: {text}")
    
    # 音声生成と時間測定
    samples, sample_rate, timing_info = piper.create(text)
    
    # 音声ファイルを保存
    sf.write('standalone_audio.wav', samples, sample_rate)
    
    # 処理時間を表示
    print(f"音声ファイル 'standalone_audio.wav' を生成しました")
    print(f"処理時間:")
    print(f"  音素変換: {timing_info['phonemize_time']:.3f}秒")
    print(f"  前処理: {timing_info['preprocess_time']:.3f}秒")
    print(f"  推論: {timing_info['inference_time']:.3f}秒")
    print(f"  合計: {timing_info['total_time']:.3f}秒")
    print(f"サンプリングレート: {sample_rate}Hz")
    print(f"音声長: {len(samples) / sample_rate:.2f}秒")