{ packages ? import <nixpkgs> {} }:
let
  baseShell = import ../../shells/shell.nix { inherit packages; };
in
packages.mkShell {
  inherit (baseShell) pure;
  buildInputs = baseShell.buildInputs ++ (with packages; [
    # ライブラリ版のサンプル実装は TensorFlow 2.16 系を使うため Python 3.12 に固定する。
    # 3.13 以降には対応する wheel が無い。
    python312
    uv
    # XGBoost の wheel が同梱する libxgboost.dylib は libomp.dylib を動的に要求する。
    # これが無いと `import xgboost` の時点で dlopen に失敗する。
    llvmPackages.openmp
  ]);
  shellHook = ''
    ${baseShell.shellHook}

    # XGBoost が libomp を見つけられるようにする（macOS / Linux の双方に効かせる）
    export DYLD_LIBRARY_PATH="${packages.llvmPackages.openmp}/lib''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${packages.llvmPackages.openmp}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    echo "Python ML development environment activated"
    echo "  - Python: $(python3.12 --version 2>&1)"
    echo "  - uv: $(uv --version 2>&1)"
    echo "  - libomp: ${packages.llvmPackages.openmp}/lib"
  '';
}
