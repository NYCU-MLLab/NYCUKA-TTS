from espnet2.bin.tts_inference import Text2Speech
import glob
import kaldiio
import numpy as np
import torch
import soundfile as sf
import time
from opencc import OpenCC
# without vocoder
# tts = Text2Speech.from_pretrained(model_file="/path/to/model.pth")
# wav = tts("Hello, world")["wav"]

# with local vocoder
# tts = Text2Speech.from_pretrained(model_file="/path/to/model.pth", vocoder_file="/path/to/vocoder.pkl")
# wav = tts("Hello, world")["wav"]

# with pretrained vocoder (use ljseepch style melgan as an example)
text2speech = Text2Speech.from_pretrained(
train_config="./exp/tts_finetune_fastpeech2/config.yaml",
model_file="./exp/tts_finetune_fastpeech2/train.loss.ave_5best.pth",
vocoder_tag="parallel_wavegan/ljspeech_parallel_wavegan.v1",
device="cuda",
 )
model_dir= "./dump"

spembs = None
if text2speech.use_spembs:
    xvector_ark = [p for p in glob.glob(f"{model_dir}/**/spk_xvector.ark", recursive=True) if "tr" in p][0]
    xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}
    spks = list(xvectors.keys())

    # randomly select speaker
    random_spk_idx = np.random.randint(0, len(spks))
    spk = spks[random_spk_idx]
    spembs = xvectors[spk].copy()
    print(f"selected spk: {spk}")

# Speaker ID selection
sids = None
if text2speech.use_sids:
    spk2sid = glob.glob(f"{model_dir}/**/spk2sid", recursive=True)[0]
    with open(spk2sid) as f:
        lines = [line.strip() for line in f.readlines()]
    sid2spk = {int(line.split()[1]): line.split()[0] for line in lines}
    
    # randomly select speaker
    sids = np.array(np.random.randint(1, len(sid2spk)))
    spk = sid2spk[int(sids)]
    print(f"selected spk: {spk}")

# Reference speech selection for GST
speech = None
if text2speech.use_speech:
    # you can change here to load your own reference speech
    # e.g.
    # import soundfile as sf
    # speech, fs = sf.read("/home/TTS_data/chatbot_check/audio_file/split/101301/003.wav")
    # speech = torch.from_numpy(speech).float()
    speech = torch.randn(50000,) * 0.01
def frontend(text) -> str:
    # punct = string.punctuation
    full_punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' + '→↓△▿⋄•！？?〞＃＄％＆』。（）＊＋，－╱︰；＜＝＞＠〔╲〕 ＿ˋ｛∣｝∼、〃》「」『』【】﹝﹞【】〝〞–—『』「」…'
    text = text.translate(str.maketrans('', '', full_punctuation))
    text = cc.convert(text)
    # print(text)
    return text
cc = OpenCC('t2s')
def tts(text):
    text = frontend(text)
    with torch.no_grad():
        start = time.time()
        wav = text2speech(text,speech=speech, spembs=spembs)["wav"]
    duration = time.time() - start
    rtf = (duration) / (len(wav) / text2speech.fs)
    sf.write("out.wav", wav.cpu().numpy(), text2speech.fs, "PCM_16")
    print(f"RTF = {rtf:5f}, Duration: {duration:5f}, wav_len: {len(wav)}")


if __name__ == '__main__':
    text = "至評量下成績觀看個人期末考分數。"
    tts(text)