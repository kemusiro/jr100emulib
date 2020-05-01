# JR-100 Emulator Library

JR-100 Emulator Library (jr100emulib)は、松下電子工業が1982年に発売した8ビットマイコンJR-100のエミュレータをサポートするPythonパッケージです。

# 提供するモジュール

## FSK Loader

FSK Loaderは、JR-100のSAVE/MSAVEコマンドで出力された音声データを解析し、格納されているプログラムをファイルとして取り出すことができます。

詳しくは[fskloader.md](/fskloader.md)を参照してください。

# 前提条件

* Python 3.7以上
* NumPy
* SciPy

# インストール

pipでjr100emulibパッケージをインストールしてください。

```shell
$ pip install jr100emulib
```

不要になった場合はpipでアンインストールできます。

```shell
$ pip uninstall jr100emulib
```

# 使い方

モジュールとして使用する場合は、必要なモジュールをimportしてください。

```python
from jr100emulib import fskloader
```

またコマンドラインツールが用意されているものはそれぞれのモジュールのドキュメントを参照してください。

# 作者

Kenichi Miyata (kemusiro@asamomiji.jp)

# ライセンス

JR-100 Emulator Library is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.

JR-100 Emulator LibraryはMITライセンスの元にライセンスされています。詳細は[LICENSE](/LICENSE)を参照してください。