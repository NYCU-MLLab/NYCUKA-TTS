from espnet2.bin.tts_inference import Text2Speech
import glob
import kaldiio
import torch
import soundfile as sf
import time
from ch_normalizer import TextNorm
from g2pw import G2PWConverter
from pypinyin.style._utils import get_finals, get_initials

class Nycuka():
    def __init__(self) -> None:
        
        self.normalizer = TextNorm(remove_space = True, to_upper = True)
        self.g2p = G2PWConverter(style='pinyin', enable_non_tradional_chinese=True)
        self.text2speech = Text2Speech.from_pretrained(
            train_config="./exp/tts_finetune_fastpeech2_g2pw/config.yaml",
            model_file="./exp/tts_finetune_fastpeech2_g2pw/train.loss.ave_5best.pth",
            vocoder_tag="parallel_wavegan/ljspeech_parallel_wavegan.v1",
            device="cuda",
            )
        self.model_dir= "./dump"
        self.spembs = None
        if self.text2speech.use_spembs:
            xvector_ark = [p for p in glob.glob(f"{self.model_dir}/**/spk_xvector.ark", recursive=True) if "tr" in p][0]
            xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}
            spks = list(xvectors.keys())
            spk = spks[0]
            self.spembs = xvectors[spk].copy()
            print(f"selected spk: {spk}")
        # Reference speech selection for GST
        self.ref_speech = None
        if self.text2speech.use_speech:
            # speech, fs = sf.read("sleepiness_113-140_0121.wav")
            # self.ref_speech = torch.from_numpy(speech).float()
            speech = torch.randn(50000,) * 0.01
            self.ref_speech = torch.randn(50000,) * 0.01
    def g2pw_phone(self,text):
        text_ = [each for each in text]
        for i,each in enumerate(text_):
            if ord(each) >= 65281:
                text_[i] = chr(ord(each)-65248)
        text_ = "".join(text_)
        phones = [
                p
                for phone in self.g2p(text_)[0]
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





    def frontend(self,text) -> str:
        # punct = string.punctuation
        normalized_text = self.normalizer(text)
        phn_seq = self.g2pw_phone(normalized_text)
        # print(text)
        return " ".join(phn_seq) 

    def __call__(self,text):
        with torch.no_grad():
            start = time.time()
            phn_seq  = self.frontend(text) 
            wav = self.text2speech(phn_seq ,speech=self.ref_speech, spembs=self.spembs, decode_conf={"alpha":1.15})["wav"]
        duration = time.time() - start
        rtf = (duration) / (len(wav) / self.text2speech.fs)
        sf.write("out.wav", wav.cpu().numpy(), self.text2speech.fs, "PCM_16")
        print(f"RTF = {rtf:5f}, Duration: {duration:5f}, wav_len: {len(wav)}")
if __name__ == '__main__':
    text = "為鼓勵國內研究生從事計算語言學相關研究，發揮研究創新之能力，培養知識經濟時代優良研究人才。"
    nycuka = Nycuka()
    nycuka(text)