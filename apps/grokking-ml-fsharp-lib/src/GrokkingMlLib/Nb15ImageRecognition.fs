/// 原著ノートブック #15 `Chapter_10_Neural_Networks/Image_recognition.ipynb`。
///
/// MNIST の手書き数字 7 万枚を、3 層のネットワークで 10 クラスに分類する。
/// 原著は 6 万件・10 エポックで正解率 0.942 を出す。
///
/// データは Keras が配布する `mnist.npz`（NumPy の `.npy` を ZIP でまとめたもの）を、
/// 3 言語で共有して読む。F# には NumPy の読み込みが無いので、
/// **必要な形式（`|u1`・C 順）だけを読む小さなパーサ** をここに書く。
module GrokkingMlLib.Nb15ImageRecognition

open System.IO
open System.IO.Compression
open System.Text.RegularExpressions
open Accord.Neuro
open Accord.Neuro.Learning

/// 画像の一辺
[<Literal>]
let ImageSize = 28

/// 入力の次元。原著の `reshape(-1, 28*28)` と同じ
[<Literal>]
let InputDim = 784

/// 原著の隠れ層
let hiddenUnits = [| 128; 64 |]

/// クラス数（数字 0〜9）
[<Literal>]
let Classes = 10

/// 原著の学習回数
[<Literal>]
let OriginalEpochs = 10

/// 読み込んだ MNIST
type Mnist =
    { XTrain: float[][]
      YTrain: int[]
      XTest: float[][]
      YTest: int[] }

/// `.npy` から取り出した符号なし 8 ビットの配列
type NpyArray = { Shape: int[]; Data: byte[] }

let private shapePattern = Regex(@"'shape':\s*\(([^)]*)\)")

/// `.npy` を読む。原著の MNIST は `|u1`（符号なし 1 バイト）だけなのでそこだけ扱う
let parseNpy (bytes: byte[]) =
    if bytes.[0] <> 0x93uy || System.Text.Encoding.ASCII.GetString(bytes, 1, 5) <> "NUMPY" then
        failwith "npy のマジックナンバーが違います"

    // 版が 1 ならヘッダ長は 2 バイト、2 以降は 4 バイト（どちらもリトルエンディアン）
    let major = int bytes.[6]
    let lengthSize = if major = 1 then 2 else 4

    let headerLength =
        if lengthSize = 2 then
            int (System.BitConverter.ToUInt16(bytes, 8))
        else
            int (System.BitConverter.ToUInt32(bytes, 8))

    let header = System.Text.Encoding.ASCII.GetString(bytes, 8 + lengthSize, headerLength)

    if not (header.Contains "'descr': '|u1'") then
        failwithf "対応していない dtype です: %s" header

    let shape =
        shapePattern.Match(header).Groups.[1].Value.Split(',')
        |> Array.map (fun part -> part.Trim())
        |> Array.filter (fun part -> part <> "")
        |> Array.map int

    let offset = 8 + lengthSize + headerLength
    { Shape = shape; Data = bytes.[offset..] }

/// `mnist.npz`（ZIP）から 1 本の `.npy` を取り出す
let readNpz (name: string) =
    use archive = ZipFile.OpenRead(Datasets.path "mnist.npz")

    let entry =
        match archive.GetEntry name with
        | null -> failwithf "%s が mnist.npz にありません" name
        | entry -> entry

    use stream = entry.Open()
    use memory = new MemoryStream()
    stream.CopyTo memory
    parseNpy (memory.ToArray())

/// MNIST を読み込む。画素値は 0〜255 のまま返す
let loadMnist () =
    let images name =
        let array = readNpz name

        Array.init array.Shape.[0] (fun row -> Array.init InputDim (fun column -> float array.Data.[row * InputDim + column]))

    let labels name =
        let array = readNpz name
        Array.init array.Shape.[0] (fun i -> int array.Data.[i])

    { XTrain = images "x_train.npy"
      YTrain = labels "y_train.npy"
      XTest = images "x_test.npy"
      YTest = labels "y_test.npy" }

/// 学習した重みの総数。原著の `model.summary()` が出す 109,386 と突き合わせる
let parameterCount =
    (InputDim * hiddenUnits.[0] + hiddenUnits.[0])
    + (hiddenUnits.[0] * hiddenUnits.[1] + hiddenUnits.[1])
    + (hiddenUnits.[1] * Classes + Classes)

/// ラベルを one-hot に直す。Accord の出力層はクラスごとに 1 つのシグモイドになる
let private oneHot (label: int) =
    Array.init Classes (fun i -> if i = label then 1.0 else 0.0)

/// 原著と同じ形のネットワークを学習する。
///
/// 原著と違うところが 2 つある。どちらも **実測して必要だと分かった** ものである。
///
/// 1. **画素値を 0〜1 に直してから渡す。** 0〜255 のままだとシグモイドが飽和して
///    学習が進まない。
/// 2. **`NguyenWidrow(network).Randomize()` が必須。**
///    [#13](Nb13NeuralNetworkBoundary.fs) と同じで、呼ばないと重みが初期化されず
///    何エポック回しても正解率が動かない。エラーも警告も出ない。
///
/// 出力層はソフトマックスではなくシグモイドが 10 個並ぶ形になる。
/// 予測はそのうち最大のものを選ぶ（`np.argmax` と同じ）。
let fit (mnist: Mnist) (epochs: int) (sampleSize: int) (seed: int) =
    Accord.Math.Random.Generator.Seed <- System.Nullable seed

    let inputs = Array.init sampleSize (fun row -> mnist.XTrain.[row] |> Array.map (fun v -> v / 255.0))
    let outputs = Array.init sampleSize (fun row -> oneHot mnist.YTrain.[row])

    let network =
        ActivationNetwork(SigmoidFunction(), InputDim, hiddenUnits.[0], hiddenUnits.[1], Classes)

    NguyenWidrow(network).Randomize()

    let teacher = BackPropagationLearning(network)

    for _ in 1..epochs do
        teacher.RunEpoch(inputs, outputs) |> ignore

    network

/// 1 枚を予測する。出力 10 個のうち最大のものを選ぶ
let predict (network: ActivationNetwork) (image: float[]) =
    let scaled = image |> Array.map (fun v -> v / 255.0)
    let output = network.Compute scaled
    output |> Array.mapi (fun index value -> index, value) |> Array.maxBy snd |> fst

/// テストセットに対する正解率
let testAccuracy (network: ActivationNetwork) (mnist: Mnist) (sampleSize: int) =
    let correct =
        seq { 0 .. sampleSize - 1 }
        |> Seq.filter (fun row -> predict network mnist.XTest.[row] = mnist.YTest.[row])
        |> Seq.length

    float correct / float sampleSize
