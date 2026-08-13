{ packages ? import <nixpkgs> {} }:
let
  baseShell = import ../../shells/shell.nix { inherit packages; };
in
packages.mkShell {
  inherit (baseShell) pure;
  buildInputs = baseShell.buildInputs ++ (with packages; [
    # 記事が「JVM 21」を前提にしているため、無印の jdk ではなくバージョンを固定する。
    # nixpkgs の更新で既定 JDK が上がると、Kotlin コンパイラが新しい JDK の
    # バージョン文字列を解釈できずビルドが壊れることがある。
    jdk21
    kotlin
    gradle
  ]);
  shellHook = ''
    ${baseShell.shellHook}
    echo "Kotlin development environment activated"
    echo "  - JDK: $(javac -version 2>&1)"
    echo "  - Kotlin: $(kotlinc -version 2>&1)"
    echo "  - Gradle: $(gradle -version | grep Gradle)"
  '';
}
