# -*- coding:utf-8 -*-

"""
JR-100 FSK Loader
"""

import wave
import numpy as np
import struct
import sys
import math
import re

_verbosity = 0

_config_file = 'fskloader.ini'
_mark_frequency = 2400.0
_space_frequency = 1200.0
_fsk_thresholds = None
_cp_threshold = 0.15
_wave_param = {}

def _read_wave(path):
    """
    waveファイルからサンプリングデータを読み込む。

    Parameters
    ----------
    path: str
        waveファイルへのパス名

    Returns
    -------
    data: numpy.ndarray
        読み込んだサンプリングデータ
    """
    global _fsk_thresholds

    if _verbosity >= 1:
        print('Reading WAV file: {}'.format(path))
    
    with wave.open(path) as f:
        param = f.getparams()
        _wave_param['nchannels'] = f.getnchannels()
        _wave_param['sample_width'] = f.getsampwidth()
        _wave_param['sampling_rate'] = f.getframerate()
        data = f.readframes(f.getnframes())

    # 不完全な音声ファイルの場合があるので余りを切り捨てる。
    r = len(data) % _wave_param['sample_width']
    data = data[:len(data)-r]

    # 各サンプルを-1.0から1.0に正規化する。
    if _wave_param['sample_width'] == 1:
        data = np.frombuffer(data, dtype = 'uint8')
        data = (data.astype('float') - 128) / 128
    elif _wave_param['sample_width'] == 2:
        data = np.frombuffer(data, dtype = 'int16')
        data = data.astype('float') / 32768
    else:
        raise NotImplementedError(
            "Sample width = {} is not supported)"
            .format(_wave_param['sample_width']))

    # ステレオの場合は左チャンネルを使う。
    if _wave_param['nchannels'] == 2:
        data = data[::2]
    elif _wave_param['nchannels'] != 1:
        raise NotImplementedError(
            "The number of channels = {} is not supported)".format(ch))

    if not _fsk_thresholds:
        _fsk_thresholds = _decide_thresholds(_wave_param['sampling_rate'],
                                             _mark_frequency,
                                             _space_frequency)

    if _verbosity >= 2:
        print('  nchannels = {}, sample_width = {}, frame_rate = {}, nframes = {}'
              .format(_wave_param['nchannels'],
                      _wave_param['sample_width'],
                      _wave_param['sampling_rate'],
                      len(data)))
        print('  FSK thresholds = {}'.format(_fsk_thresholds))
        
    return data

def _comparator(x0, x1, th_h, th_l):
    """
    ヒステリシス付きのコンパレータの動作を模擬する。

    Parameters
    ----------
    th_h: float
        立ち上がりエッジのしきい値
    th_l: float
        立ち下がりエッジのしきい値

    Returns
    -------
    value: float
        コンパレータの出力値
    """
    if x0 < th_h and x1 >= th_h:
        return 1.0
    elif x0 > th_l and x1 <= th_l:
        return -1.0
    else:
        return x0

def _write_wave(name, x, rate):
    '''
    サンプリング データをファイルに出力する。

    Parameters
    ----------
    name: str
        ファイル名
    x: numpy.ndarray
        サンプリング データ
    rate:
        サンプリング周波数
    '''
    import scipy.io.wavfile as siw
    siw.write(name, rate, x)
    
def _shape_wave(x):
    """
    サンプルデータを2値化する。

    Parameters
    ----------
    x: numpy.ndarray
        正規化されたサンプリングデータ

    Returns
    -------
    x: numpy.ndarray
        2値化されたサンプリングデータ
    """
    global _cp_threshold
    
    if _verbosity >= 1:
        print('Shaping')
        
    # 直流成分を除去する。
    offset = np.mean(x)
    if _verbosity >= 1:
        print('  Elminating DC component: offset = {}'.format(offset))
    x = x - offset

    # 波形の絶対値の最大値が1になるよう増幅する。
    a = 1 / max(np.max(x), abs(np.min(x)))
    if _verbosity >= 1:
        print('  Amplifying: a = {}'.format(a))
    x = a * x

    # 先頭と末尾の無音部分を除去する。
    pos_head = np.where(abs(x) >= 0.5)
    if _verbosity >= 1:
        print('  Removing silient part: {} - {}'
              .format(pos_head[0][0] / _wave_param['sampling_rate'],
                      pos_head[0][-1] / _wave_param['sampling_rate']))
    # x = x[pos_head[0][0]:pos_head[0][-1]+1]

    if _verbosity >= 4:
        _write_wave('shaped1.wav', x, _wave_param['sampling_rate'])

    # データを+1.0と-1.0に2値化する。
    if _verbosity >= 1:
        print('  Binarizing')
    x = np.append(-1, x) # 先頭にダミー初期値を追加
    for i in range(len(x)-1):
        x[i+1] = _comparator(x[i], x[i+1], _cp_threshold, -_cp_threshold)

    if _verbosity >= 4:
        _write_wave('shaped2.wav', x, _wave_param['sampling_rate'])

    return x[1:] # ダミーを削除して返す。

def _decide_thresholds(fs, f1, f0):
    """
    FSKで0と1の判定のためのしきい値を決定する。

    Parameters
    ----------
    fs: float
        音声データのサンプリング周波数
    f1: float
        FSKにおけるマーク(1)周波数
    f0: float
        FSKにおけるスペース(0)周波数
    """
    pos_f0 = math.floor(fs / f0)
    pos_f1 = math.floor(fs / f1)
    pos_center = (pos_f0 + pos_f1) / 2
    distance = math.floor(pos_center - pos_f1)
    return ((int(math.floor(pos_f1 - distance)),
             int(math.floor(pos_f1 + distance))),
            (int(math.floor(pos_f1 + distance) + 1),
             int(math.floor(pos_f0 + distance))))

def set_fsk_thresholds(min_1, max_1, min_0, max_0):
    """
    FSKで0と1の判定のためのしきい値を設定する。

    Parameters
    ----------
    min_1: int
        '1'と判定するための最小幅
    max_1:
        '1'と判定するための最大幅
    min_0: int
        '0'と判定するための最小幅
    max_0:
        '0'と判定するための最大幅
    """
    global _fsk_thresholds
    _fsk_thresholds = ((min_1, max_1), (min_0, max_0))

def set_cp_threshold(value):
    """
    コンパレータで立ち上がりと立ち下がりを判定するためのしきい値を設定する。

    Parameters
    ----------
    value: float
        しきい値
    """
    global _cp_thresholds
    _cp_thresholds = value
    
def _detector(v):
    """
    波形のフェーズを検出する。

    Parameters
    ----------
    v: float
        正規化されたサンプルデータ

    Returns
    -------
    phase: str
        フェーズ

    Notes
    -----
    フェーズは以下のいずれかである。
    * 0: 値'0'に対するフェーズ
    * 1: 値'1'に対するフェーズ
    * unknown: それ以外の未知のフェーズ
    """
    if _fsk_thresholds[1][0] * 2 <= v <= _fsk_thresholds[1][1] * 2:
        return 0
    elif  _fsk_thresholds[0][0] * 2 <= v <= _fsk_thresholds[0][1] * 2:
        return 1
    else:
        return -1

def _fsk_value(x, i):
    """
    FSKで符号化された音声フェーズ列から1つの値を求める。

    Parameters
    ----------
    x: bytearray
        フェーズ列
    i: int
        オフセット

    Returns
    -------
    value: int
        復号化された1つの値
    offset: int
        次のオフセット    

    Notes
    -----
    valueとoffsetはタプルで返す。
    """
    try:
        if _detector(x[i]+x[i+1]) == 0:
            return (0, i + 2)
        elif (_detector(x[i]+x[i+1]) == 1
              and _detector(x[i+2]+x[i+3]) == 1):
            return (1, i + 4)
        else:
            return (255, i + 1)
    except IndexError:
        return (255, i + 1)

def _dump_histogram(d):
    '''
    ヒストグラムを出力する。

    Parameters
    ----------
    d: numpy.ndarray
        立ち下がりエッジ間の距離のリスト
    '''
    hist, bins = np.histogram(
        abs(d),
        bins=np.arange(abs(d).min(), min(60, abs(d).max()+2), step=1))
    print('  --------------------')
    if _fsk_thresholds[0][0] < abs(d).min():
        print('  ---- start of H ----')
    for i, n in zip(bins, hist):
        if i == _fsk_thresholds[0][0]:
            print('  ---- start of H ----')
        elif i == _fsk_thresholds[1][0]:
            print('  ---- start of L ----')
            
        print('  {:3d}: {}'.format(i, n))
            
        if i == _fsk_thresholds[0][1]:
            print('  ---- end of H ----')
        elif i == _fsk_thresholds[1][1]:
            print('  ---- end of L ----')
    if _fsk_thresholds[1][1] > abs(d).max():
        print('  ---- end of L ----')
    print('  --------------------')
        
def _serialize(x):
    """
    ビット列を取り出す。

    Parameters
    ----------
    x: numpy.ndarray
        2値化されたサンプリングデータ

    Returns
    -------
    bits: bytearray
        ビット列
    """
    if _verbosity >= 1:
        print('Serializing')

    # 立ち下がりエッジの位置を検出する。
    falling_edge_pos = []
    previous = x[0]
    for i in range(1, len(x)):
        if previous > 0 and x[i] < 0:
            falling_edge_pos.append(i)
        previous = x[i]

    # 立ち下がりエッジ間の距離を求める。
    f = np.array(falling_edge_pos)
    d = np.diff(f)

    # FSK値1相当のパルス幅が10個連続で現れたらセーブデータの開始とみなす。
    start_of_signal_found = False
    i = 0
    count = 0
    while i < len(d) and not start_of_signal_found:
        if _fsk_thresholds[0][0] <= d[i] <= _fsk_thresholds[0][1]:
            count += 1
        else:
            count = 0
        if count == 10:
            start_of_signal_found = True
            break
        i += 1
    if not start_of_signal_found:
        print('  signal head not found')
    else:
        if _verbosity >= 2:
            print('  signal starts from sample #{}'.format(i))

    # 最初のゼロを探す。
    # このゼロの位置を起点に、2個('0'の場合)または4個('1'の場合)の単位で
    # FSKの復号を行う。
    first_zero_found = False
    while i < len(d) and not first_zero_found:
        if _fsk_thresholds[1][0] <= d[i] <= _fsk_thresholds[1][1]:
            first_zero_found = True
            break
        i += 1
    if not first_zero_found:
        print('  first zero not found')
    else:
        if _verbosity >= 2:
            print('  first zero found at sample #{}'.format(i))

    if _verbosity >= 2:
        print('  Histogram of the interval betwen falling edges')
        _dump_histogram(d)
        
    bits = bytearray()
    while i < len(d):
        value, new_i = _fsk_value(d, i)
        if value == 0 or value == 1:
            bits.append(value)
        i = new_i

    if _verbosity >= 3:
        print('  Bit sequence length = {}'.format(len(bits)))
        print('  Bit sequence contents:')
        print('------------------------')
        for k in range(0, len(bits), 1000):
            print('{} - {}'.format(k, k + 1000 - 1))
            box = bits[k:k+1000]
            print('\n'.join([''.join([str(box[i])
                                      if box[i] == 0 or box[i] == 1
                                      else '.'
                                      for i in range(len(box))][j:j+100])
                             for j in range(0, len(box), 100)]))
        print('----------------------')

    return bits

def _bits_to_byte(b):
    '''
    リトルエンディアンのビット列を整数値に変換する。

    Parameters
    ----------
    b: bytearray
        ビット列

    Returns
    -------
    value: int
        ビット列に対するバイト値
    '''
    a = [e if e == 0 or e == 1 else 0 for e in b]
    return sum([a[i] * 2**i for i in range(len(a))])
    
def _get_byte(b, i):
    """
    音声データのフェーズ列からビット列を取り出す。

    Parameters
    ----------
    b: bytearray
        ビット列
    i: int
        オフセット

    Returns
    -------
    byte: int
        バイト値
    offset: int
        次のオフセット
    skip: int
        スキップしたビット数

    Notes
    -----
    返値はbyteとoffsetのタプルである。
    """
    if i+10 < len(b):
        start = b[i]
        value = _bits_to_byte(b[i+1:i+9])
        stop1, stop2 = b[i+9], b[i+10]
        if (start, stop1, stop2) == (0, 1, 1):
            return (value, i+11, 0)
        else:
            if _verbosity >= 1:
                print('  invalid separator bits: {} at {}'.format(b[i:i+11], i))
            found = False
            current_i = i
            i += 1
            while not found and i+10 < len(b):
                if (b[i], b[i+9], b[i+10]) == (0, 1, 1):
                    found = True
                    break
                i += 1
            if found:
                value = _bits_to_byte(b[i+1:i+9])
                return (value, i+11, i - current_i)
            else:
                return (value, i+1, 0)
    else:
        return (255, i+1, 0)
    
def _decode(bits):
    """
    ビット列を復号化する。

    Parameters
    ----------
    bits: bytearray
        FSKデータから得られたビット列

    Returns
    -------
    byte_sequence: bytearray
        ビット列から取り出したバイト列    
    """
    if _verbosity >= 1:
        print('Decoding')

    bit_sequence = np.array(bits)
    header_index = -1

    # データ末尾のストップビットが欠落する場合があるため明示的に追加しておく。
    bit_sequence = np.append(bit_sequence, 1)

    header_index = -1
    count = 0
    pattern = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1])
    for i in range(len(bit_sequence)):
        if np.all(bit_sequence[i:i+9] == pattern):
            header_index = i
            break

    if _verbosity >= 2:
        print('  header start from {}'.format(header_index))
    
    if header_index >= 0:
        count = 0
        i = header_index
        while i < len(bit_sequence) and np.all(bit_sequence[i:i+9] == pattern):
            count += 1
            i += 9

        if _verbosity >= 2:
            print('  pattern count = {}'.format(count))

    # 0b011111111(9bit)のパターンが10個以上出現したらJR-100が生成するCMTデータとみなす。
    if header_index < 0 or count < 10:
        if _verbosity >= 1:
            print('  JR-100 Header not found')
        return None
    else:
        if _verbosity >= 2:
            print('  JR-100 Header found')

    # 最初の0(最初のバイトのスタートビット)の位置を求める。
    i += np.where(bit_sequence[i:] == 0)[0][0]
        
    byte_sequence = bytearray()

    # ヘッダ分の読み取り
    if _verbosity >= 1:
        print('  Reading Header from {}'.format(i))
    for _ in range(33):
        v, i, skip = _get_byte(bit_sequence, i)
        if skip > 0:
            print('  skip {} bits, tentative value = 0x{:02x}'.format(skip, v))
            print('  ...{}'.format(byte_sequence[-30:]))
        byte_sequence.append(v)

    # プログラムデータ分の読み取り
    if _verbosity >= 1:
        print('  Reading Body from {}'.format(i))
    i = np.where(bit_sequence[i:] == 0)[0][0] + i
    length = (byte_sequence[18] << 8) + byte_sequence[19]
    for _ in range(length + 1):
        v, i, skip = _get_byte(bit_sequence, i)
        if skip > 0:
            print('  skip {} bits, tentative value = 0x{:02x}'.format(skip, v))
            print('  ...{}'.format(byte_sequence[-30:]))
        byte_sequence.append(v)

    # チェックサムの検査
    if _verbosity >= 1:
        print('  Checking checksum')
    if np.sum(byte_sequence[0:31]) % 256 != byte_sequence[32]:
        print('Check sum error (header)')
    else:
        if _verbosity >= 2:
            print('  Chedk sum (header) OK')

    if np.sum(byte_sequence[33:33+length]) % 256 != byte_sequence[33+length]:
        print('Check sum error (body)')
    else:
        if _verbosity >= 2:
            print('  Chedk sum (body) OK')

    return byte_sequence

def get_param(program):
    """
    プログラムのパラメータを取り出す。

    Parameter
    ---------
    program: bytearray
        プログラムデータ

    Returns
    -------
    param: dict
        プログラムのパラメータ

    Notes
    -----
    以下のパラメータを返す。
    * name: プログラム名
    * start_addr: プログラムの開始アドレス
    * flag: BASICプログラムの場合は0、マシン語の場合は77(0x4d)
    """
    name, addr, len, flag = struct.unpack('>16sHHB11x', program[0:32])
    name = name.replace(b'\00', b'')
    return {'name': name, 'start_addr': addr, 'length': len, 'flag': flag}

def get_body(program):
    """
    プログラム本体を取り出す。

    Parameter
    ---------
    program: bytearray
        プログラムデータ

    Returns
    -------
    param: bytearray
        プログラム本体
    """
    length = get_param(program)['length']
    return program[33:33+length]

def write_file(program, path):
    """
    プログラムをPROG形式のファイルとして書き込む。

    Parameters
    ----------
    program: bytearray
        プログラムデータ
    path: str
        出力するファイルのパス名
    """
    param = get_param(program)
    name = param['name']
    addr = int(param['start_addr'])
    flag = 1 if param['flag'] == 0x4d else 0
    body = get_body(program)

    data = struct.pack('<4sII{}sIII{}s'.format(len(name), len(body)),
                       b'PROG',      # 'PROG' (4s)
                       0x0000_0001,  # フォーマットバージョン (I)
                       len(name),    # プログラム名長 (I)
                       name,         # プログラム名 ({}s)
                       addr,         # 先頭アドレス (I)
                       len(body),    # プログラム長 (I)
                       flag,         # フラグ (I)
                       body)         # プログラム ({}s)

    with open(path, 'wb') as f:
        f.write(data)

def decode_file(infile):
    """
    FSKで符号化されたJR-100プログラムの音声データを復号化する。

    Parameter
    ---------
    infile: str
        waveファイルへのパス名

    Returns
    -------
    program: bytearray
        復号化されたプログラム
    """
    return _decode(_serialize(_shape_wave(_read_wave(infile))))

def dump_basic(program):
    """
    読み込んだプログラムをBASICとしてダンプする。

    Parameter
    ---------
    program: bytes
        復号化されたプログラム
    """
    body = get_body(program)
    for m in re.finditer(
            b'(?P<num>[^\xdf][^\xdf])(?P<line>[^\x00]*?)\x00',
            body, flags=re.DOTALL):
        n = struct.unpack('>H', m.group('num'))[0]
        line = ''.join([r'\x{:02x}'.format(x) if x >= 128 else chr(x)
                        for x in m.group('line')])
        print('{} {}'.format(n, line))

def dump_binary(program):
    """
    読み込んだプログラムをバイナリデータとしてダンプする。

    Parameter
    ---------
    program: bytes
        復号化されたプログラム
    """
    param = get_param(program)
    body = get_body(program)
    
    if param['start_addr'] % 16 != 0:
        print('{:04x}: '.format(param['start_addr'] & 0xfff0), end='')
        print(' ' * ((param['start_addr'] & 0xf) * 3), end='')

    for offset in range(param['length']):
        addr = param['start_addr'] + offset
        if addr % 16 == 0:
            print('{:04x}: '.format(addr), end='')
        print('{:02x} '.format(body[offset]), end='')
        if addr % 16 == 15:
            print()
    print()

def dump_program(program):
    """
    読み込んだプログラムのフラグに応じてダンプする。

    Parameter
    ---------
    program: bytes
        復号化されたプログラム
    """
    if get_param(program)['flag'] == 0x4d:
        dump_binary(program)
    else:
        dump_basic(program)

def set_verbosity(level):
    global _verbosity
    _verbosity = level

def main():
    import argparse
    import os
    import traceback
    import configparser
    from collections import OrderedDict
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help="specify an input wave file")
    parser.add_argument('-o', '--output_file',
                        help="write to a PROG format file",
                        metavar="file")
    parser.add_argument('-s', '--config_section',
                        help="configuration section",
                        metavar="name")
    parser.add_argument('--overwrite',
                        help="overwrite to the existing file",
                        action='store_true')
    parser.add_argument('--fsk_thresholds',
                        help="specify threshold values for FSK detector",
                        metavar=('min_1', 'max_1', 'min_0', 'max_0'),
                        nargs=4, type=int)
    parser.add_argument('--cp_threshold',
                        help="specify a threshold value for comparator",
                        metavar=('value'),
                        type=float)
    parser.add_argument('--dump_auto',
                        help="Dump this program depending on the type of program",
                        action='store_true')
    parser.add_argument('--dump_basic',
                        help="Dump this program as a BASIC text",
                        action='store_true')
    parser.add_argument('--dump_binary',
                        help="Dump this program as a binary data",
                        action='store_true')
    parser.add_argument('-v', '--verbose',
                        help="enable verbose output",
                        action="count", default=0)
    
    args = parser.parse_args()
    if args.verbose:
        set_verbosity(args.verbose)

    config = configparser.ConfigParser()
    sec = OrderedDict()
    if os.path.exists(_config_file):
        config.read('fskloader.ini')
        sec.update(config.defaults())
        if config.has_section(args.config_section):
            sec.update(config[args.config_section])

    if args.fsk_thresholds:
        set_fsk_thresholds(
            args.fsk_thresholds[0],
            args.fsk_thresholds[1],
            args.fsk_thresholds[2],
            args.fsk_thresholds[3],
        )
    elif sec.get('fsk_thresholds'):
        s = sec.get('fsk_thresholds')
        set_fsk_thresholds(*[int(x) for x in s.split()])

    if args.cp_threshold:
        set_cp_threshold(args.cp_threshold)
    elif sec.get('cp_threshold'):
        set_cp_threshold(float(sec.get('cp_threshold')))

    _do_dump = (args.dump_auto
                or args.dump_basic
                or args.dump_binary)

    try:
        # プログラム読み込み前に上書きチェックを行う。
        if (args.output_file
            and os.path.isfile(args.output_file)
            and not args.overwrite):
                raise FileExistsError("File exists: {}".format(args.output_file))
        program = decode_file(args.input_file)
        if not program:
            raise Exception("Invalid JR-100 FSK format")
        if args.verbose >= 1:
            print('Decoding Successed: {}'.format(get_param(program)['name']))
                
        if args.output_file:
            write_file(program, args.output_file)
    
        if _do_dump:
            param = get_param(program)
            if args.dump_basic:
                dump_basic(program)
            elif args.dump_binary:
                dump_binary(program)
            else:
                dump_program(program)
    except Exception as e:
        if _verbosity >= 2:
            print(traceback.format_exc())
        else:
            print(e)

if __name__ == '__main__':
    main()
