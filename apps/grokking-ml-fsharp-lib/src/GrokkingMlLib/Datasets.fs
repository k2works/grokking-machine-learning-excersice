/// 共有データセット（`apps/grokking-ml-datasets`）へのアクセス。
///
/// Python 版・Kotlin 版と同じ CSV を読むことで、章ごとの数値を言語間で突き合わせられる。
/// サイズの大きい 2 本だけはリポジトリに入れず、初回参照時に原著リポジトリから取得する。
module GrokkingMlLib.Datasets

open System
open System.IO
open System.Net.Http

/// 共有データセットディレクトリを差し替える環境変数。CI やノートブックで使う
[<Literal>]
let private EnvKey = "GROKKING_ML_DATASETS"

/// リポジトリに含めず初回にダウンロードするファイルと、その取得元
let private remoteFiles =
    dict [
        "emails.csv",
        "https://raw.githubusercontent.com/luisguiserrano/manning/master/Chapter_08_Naive_Bayes/emails.csv"
        "IMDB_Dataset.csv",
        "https://raw.githubusercontent.com/luisguiserrano/manning/master/Chapter_06_Logistic_Regression/IMDB_Dataset.csv"
    ]

/// 実行ディレクトリから親へ辿って `grokking-ml-datasets` を探す。
///
/// テストは `tests/<プロジェクト>/bin/<構成>/<TFM>/`、ノートブックは別の深さから
/// 動くので、相対段数を決め打ちにせず探索する。
let rec private findUpwards (dir: DirectoryInfo) =
    if isNull (box dir) then
        None
    else
        let candidate = Path.Combine(dir.FullName, "grokking-ml-datasets")
        if Directory.Exists candidate then Some candidate else findUpwards dir.Parent

/// 共有データセットディレクトリを返す
let directory () =
    match Environment.GetEnvironmentVariable EnvKey with
    | null | "" ->
        match findUpwards (DirectoryInfo AppContext.BaseDirectory) with
        | Some found -> found
        | None -> failwith "grokking-ml-datasets ディレクトリが見つかりません"
    | value -> Path.GetFullPath value

/// データセットの絶対パスを返す。未取得の大きいファイルはダウンロードする
let path (name: string) =
    let target = Path.Combine(directory (), name)

    if File.Exists target then
        target
    else
        match remoteFiles.TryGetValue name with
        | false, _ -> raise (FileNotFoundException($"データセットが見つかりません: {target}", target))
        | true, url ->
            Directory.CreateDirectory(Path.GetDirectoryName target: string) |> ignore
            use client = new HttpClient()
            use stream = client.GetStreamAsync(url).GetAwaiter().GetResult()
            use file = File.Create target
            stream.CopyTo file
            target

/// データセットを Deedle のデータフレームとして読み込む
let loadFrame (name: string) =
    Deedle.Frame.ReadCsv(path name)
