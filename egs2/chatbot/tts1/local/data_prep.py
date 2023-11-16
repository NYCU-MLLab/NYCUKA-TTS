import argparse
import os

from espnet2.utils.types import str2bool
from itertools import count
import os
import numpy as np
import sys
from scipy.io import wavfile
from tqdm import tqdm
import csv
# from pypinyin import pinyin, Style。
import re
import os
from ch_normalizer import TextNorm
from opencc import OpenCC
from g2pw import G2PWConverter
from pypinyin.style._utils import get_finals, get_initials
SPK_LABEL_LEN = 7
sys.path.append(os.getcwd())

normalizer = TextNorm(remove_space = True, to_upper = True)
g2pw = G2PWConverter(style='pinyin', enable_non_tradional_chinese=True)
cc = OpenCC('t2s')
def g2pw_phone(text):
  phones = [
        p
        for phone in g2pw(text)[0]
        for p in [
            get_initials(phone, strict=True),
            get_finals(phone[:-1], strict=True) + phone[-1]
            if phone[-1].isdigit()
            else get_finals(phone[0], strict=True)
            if phone[-1].isalnum()
            else phone 
        ]
        # Remove the case of individual tones as a phoneme
        if len(p) != 0 and not p.isdigit()
    ]
  return phones
def full2half(text):
    text_ = [each for each in text]
    for i,each in enumerate(text_):
        if ord(each) >= 65281:
            text_[i] = chr(ord(each)-65248)
    text_ = "".join(text_)
    return text_
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--espnet_g2p", type=str2bool, default=True)

    args = parser.parse_args()
    src_all_folder = os.listdir(args.src)
    
    wavscp = open(os.path.join(args.dest, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(args.dest, "utt2spk"), "w", encoding="utf-8")
    text = open(os.path.join(args.dest, "text"), "w", encoding="utf-8")

    for audio_folder in src_all_folder:
        data_path = os.path.join(args.src,audio_folder)
        if "." in audio_folder:
            continue
        print(data_path)
        with open(os.path.join(data_path,"text.csv"), encoding="utf-8") as f:
            content = csv.reader(f, delimiter = ",")
            # print(content)
            for info in tqdm(content):
                # print(info)
                wav_name, text_seq = info[0], info[1]
                text_seq = full2half(text_seq)
                if re.search(r'[a-zA-Z]+',text_seq):
                    continue
                # print(text,end = " ")
                ori = text_seq
                if args.espnet_g2p:
                    text_seq = cc.convert(text_seq)
                    seq = normalizer(text_seq)
                else:
                    text_seq = normalizer(text_seq)
                    try:
                        phn_seq =  g2pw_phone(text_seq)
                    except:
                        continue
                    seq = " ".join(phn_seq)
                spk_id  = "chatbot"
                utt_id = audio_folder + "_" + wav_name[:-4]
                wav_path = os.path.join(data_path, wav_name)
                wavscp.write("{} {}\n".format(utt_id, wav_path))
                utt2spk.write("{} {}\n".format(utt_id, spk_id))
                text.write("{} {}\n".format(utt_id, seq))

    # transcript.close()
    wavscp.close()
    utt2spk.close()
    text.close()
