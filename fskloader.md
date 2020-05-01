# FSK Loader

FSK Loaderは、JR-100のSAVE/MSAVEコマンドで出力される音声データを解析し、格納されているプログラムをファイルとして取り出すライブラリです。

# できること

* データを格納したWAVファイルを解析し、プログラムデータとして取り出す。
* プログラムデータを以下の形式で出力する。
  * BASICテキストとして標準出力に出力
  * バイナリデータとして標準出力に出力
  * PROG形式としてファイルに出力

# 実行例

セーブデータを格納した`HELLO.wav`ファイルを読み込み、内容を標準出力に表示するには以下のコマンドを実行します。

```shell
$ fskloader HELLO.wav --dump_auto
10 PRINT"HELLO,WORLD\x82"
20 GOTO10
```

`--dump_auto`オプションは、セーブデータがBASICかマシン語かを自動判定して、適切なフォーマットで出力します。BASICのプログラムでも明示的に`--dump_binary`を指定することにより、バイナリデータとして16進ダンプを出力することもできます。

```shell
$ fskloader --dump_auto ../jr100emulib/testdata/HELLO.wav --dump_binary
0240:                   00 0a 50 52 49 4e 54 22 48 45
0250: 4c 4c 4f 2c 57 4f 52 4c 44 82 22 00 00 14 47 4f
0260: 54 4f 31 30 00 df df
```

JR-100ではBASICプログラムはアドレス`0x0246`から格納されていることが分かります。

またJR-100エミュレータver2で読み込めるPROG形式のフォーマットでファイル出力することもできます。`-o`オプションで、出力ファイル名を指定することができます。

```shell
$ fskloader --dump_auto ../jr100emulib/testdata/HELLO.wav -o HELLO.prg
10 PRINT"HELLO,WORLD\x82"
20 GOTO10
```


# コマンドの説明

コマンドライン引数は以下となります。

```shell
$ fskloader -h
usage: fskloader [-h] [-o file] [-s name] [--overwrite] [--fsk_thresholds min_1 max_1 min_0 max_0]
                 [--cp_threshold value] [--dump_auto] [--dump_basic] [--dump_binary] [-v]
                 input_file

positional arguments:
  input_file            specify an input wave file

optional arguments:
  -h, --help            show this help message and exit
  -o file, --output_file file
                        write to a PROG format file
  -s name, --config_section name
                        configuration section
  --overwrite           overwrite to the existing file
  --fsk_thresholds min_1 max_1 min_0 max_0
                        specify threshold values for FSK detector
  --cp_threshold value  specify a threshold value for comparator
  --dump_auto           Dump this program depending on the type of program
  --dump_basic          Dump this program as a BASIC text
  --dump_binary         Dump this program as a binary data
  -v, --verbose         enable verbose output
```

## `-h`

コマンドライン引数のヘルプを表示します。

## `-o file`または `--output_file file`

指定したファイル名`file`にPROG形式で出力します。指定したファイルが存在する場合はエラー終了し、ファイルは上書きされません。上書きする場合は`--overwrite`オプションを明示的に指定してください。

## `-s name`または`--config_section name`

FSK Loaderは、変換アルゴリズムをチューニングするための設定を`fskloader.ini`に格納しておくことができます。このファイルの形式は以下のようになります。

```
# デフォルト値を格納するためのセクション
[DEFAULT]
fsk_thresholds = 5 12 13 24
cp_threshold = 0.15

# 任意の名前のセクション
[demo1]
fsk_thresholds = 3 14 15 30
```

角括弧でセクション名を定義し、その中でパラメータ値を設定します。セクション名`DEFAULT`は個々のセクションで指定されなかったパラメータの既定値を定義する特別なセクションです。

`-s`オプションにより、どのセクションのパラメータを指定するかを指定できます。

以下のパラメータの設定が可能です。

* `fsk_thresholds`: FSKによるパルス幅を0と1に分離するためのしきい値

* `cp_threshold`: 音声データの立ち上がりエッジと立ち下がりエッジを検出するためのしきい値

## `--overwrite`

出力先のファイルが存在していても上書きします。

## `--fsk-thresholds min_1 max_1 min_0 max_0`

 FSKによるパルス幅を0と1に分離するためのしきい値を指定します。`fskloader.ini`の指定よりも優先されます。

## `--cp_threshold value`

音声データの立ち上がりエッジと立ち下がりエッジを検出するためのしきい値します。`fskloader.ini`の指定よりも優先されます。

## `--dump_auto`

読み込んだプログラムがBASICかマシン語かを自動判定して、プログラム内容を標準出力に出力します。

## `--dump_basic`

読み込んだBASICプログラムを標準出力に出力します。プログラムがマシン語の場合は何も出力しません。

## `--dump_binary`

 読み込んだプログラムを16進ダンプします。

## `-v`, `-vv`, `-vvv`, `-vvvv`

デバッグ 用の情報を出力します。`v`の数が多いほど、多くの情報が出力されます。

# 推奨する音声データのフォーマット

音声データを正しく読み込むために、以下のフォーマットで音声データを作成することを推奨します。

* サンプリング周波数: 22050Hz以上
* サンプリング データのビット数: 16bit

JR-100実機から直接オーディオキャプチャボードを介して音声データを取り込む場合は、サンプリング周波数が8000Hzでも正しく読み込めることが期待できます。逆にかなり過去に保存したカセットテープからデータをキャプチャする場合は、テープの劣化により0と1とをうまく分離できない可能性が高いため、できるだけ高精度、つまりサンプリング周波数を44100Hzなどで取り込むことが必要となります。
# JR-100 BASIC ROMの取り出し方

JR-100エミュレータ実行用にROMデータをWAVファイルとして取り出す必要があります。取り出し方の一例を以下に示します。

## 1. 取り込み環境の作成

JR-100とオーディオキャプチャデバイスを接続します。キャプチャ条件は以下とします。

* サンプリング周波数: 22050Hz
* サンプリング ビット数: 16bit
* チャンネル数: 1 (モノラル)

## 2. JR-100上でROM領域をセーブ

PC側でキャプチャ開始してから、JR-100上でBASIC ROM領域($E000〜$FFFF)全てをマシン語データとしてセーブします。

```basic
MSAVE "JR100ROM", $E000, $FFFF
```

## 3. WAVファイルから取り込み

```shell
$ fskloader jr100rom.wav -o jr100rom.prg
```



## JR-100のセーブデータのフォーマット

セーブデータ中では、1バイトの値をスタートビット1ビット(0b0)とストップビット2ビット(0b11)を合わせて、11ビット分で表現します。また、送信するビット列はビット単位でのリトルエンディアンであり、LSB(ビット0)からMSB(ビット8)の順に出力されます。

| オフセット      | バイト長 | 値                                             |
| --------------- | -------- | ---------------------------------------------- |
| (非データ部分1) | －       | 0b1                                            |
| (非データ部分2) | －       | 0b1111_1111_0 (9ビット) * 28 (*1)              |
| (非データ部分3) | －       | 0b1 * 3828                                     |
| 0               | 16       | ファイル名(*2)                                 |
| 16              | 2        | 先頭アドレス(*3)                               |
| 18              | 2        | プログラム長N                                  |
| 20              | 1        | フラグ (*4)                                    |
| 21              | 11       | パディング (0x00 * 11バイト)                   |
| 32              | 1        | オフセット0〜32の32バイトの合計値の下位1バイト |
| (非データ部分4) | －       | 0b1 * 255                                      |
| 0               | N        | プログラムデータ                               |
| N               | 1        | プログラムデータの合計値の下位1バイト          |

* (*1) JR-100に付属のデモカセットでは、非データ部分2の繰り返し回数が21回となっており、厳密に守らなくてもLOAD/MLOADで読み込めるようになっています。
* (*2) ファイル名の最大長は15バイト。16バイトに満たない部分は0x00で埋める。
* (*3) BASICプログラムの場合は常に0x0246である。
* (*4) BASICプログラムの場合は0x00、マシン後の場合は0x4D('M')となる。

