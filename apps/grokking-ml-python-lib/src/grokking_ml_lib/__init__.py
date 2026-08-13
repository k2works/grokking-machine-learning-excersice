"""Grokking Machine Learning の原著ノートブックを ML ライブラリで再現するパッケージ。

自前実装版（`apps/grokking-ml-python`）と対になる。こちらは NumPy・pandas・
scikit-learn・TensorFlow/Keras・XGBoost を使い、原著のコードに忠実な処理を書く。
"""

from grokking_ml_lib.datasets import dataset_path, load_csv

__all__ = ["dataset_path", "load_csv"]
