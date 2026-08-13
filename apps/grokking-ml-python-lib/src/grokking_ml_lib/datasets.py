"""共有データセット（`apps/grokking-ml-datasets`）へのアクセス。

3 言語のライブラリ版実装が同じ CSV を読むことで、章ごとの数値を突き合わせられる。
サイズの大きい 2 本だけはリポジトリに入れず、初回参照時に原著リポジトリから取得する。
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd

#: 共有データセットディレクトリ。環境変数で差し替えられる（CI やノートブック用）
_ENV_KEY = "GROKKING_ML_DATASETS"

#: リポジトリに含めず初回にダウンロードするファイルと、その取得元
_REMOTE_FILES: dict[str, str] = {
    "emails.csv": (
        "https://raw.githubusercontent.com/luisguiserrano/manning/master/"
        "Chapter_08_Naive_Bayes/emails.csv"
    ),
    "IMDB_Dataset.csv": (
        "https://raw.githubusercontent.com/luisguiserrano/manning/master/"
        "Chapter_06_Logistic_Regression/IMDB_Dataset.csv"
    ),
    # Keras が `keras.datasets.mnist.load_data()` で取りにいくのと同じファイル。
    # 3 言語で同じ画像を使うために共有する
    "mnist.npz": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
}


def datasets_dir() -> Path:
    """共有データセットディレクトリを返す。"""
    override = os.environ.get(_ENV_KEY)
    if override:
        return Path(override)
    # src/grokking_ml_lib/datasets.py -> apps/grokking-ml-python-lib -> apps
    return Path(__file__).resolve().parents[3] / "grokking-ml-datasets"


def dataset_path(name: str) -> Path:
    """データセットの絶対パスを返す。未取得の大きいファイルはダウンロードする。"""
    path = datasets_dir() / name
    if path.exists():
        return path
    if name not in _REMOTE_FILES:
        raise FileNotFoundError(f"データセットが見つかりません: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_REMOTE_FILES[name], path)
    return path


def load_csv(name: str, **kwargs) -> pd.DataFrame:
    """データセットを pandas の DataFrame として読み込む。"""
    return pd.read_csv(dataset_path(name), **kwargs)
