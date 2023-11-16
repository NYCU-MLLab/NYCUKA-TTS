# The handover on implementing nycuka TTS

## Installation
- This works depends on ESPnet2 with python 3.8

```sh
$ sudo apt-get install cmake
$ sudo apt-get install sox
$ sudo apt-get install zip
```

```sh
$ git clone -b nycuka https://github.com/Li-JEN/ConversationalAI-TTS.git
$ cd ConversationalAI-TTS/tools
$ ./setup_anaconda.sh miniconda espnet 3.8
$ . ./activate_python.sh
$ make TH_VERSION=1.10.1 CUDA_VERSION=11.3
$ pip install onedrivedownloader pyopenjtalk typeguard==2.13.3 Pillow==9.5.0 numpy==1.23.0 g2pw geomloss OpenCC POT==0.4.0
## need to install kaldi
$ cd ConversationalAI-TTS/tools/kaldi/tools
$ make -j <NUM-CPU> #There would has some error. Follow the hint and you can solve.
$ ./extras/install_openblas.sh
$ cd ConversationalAI-TTS/tools/kaldi/src
$ ./configure --openblas-root=../tools/OpenBLAS/install --use-cuda=no
$ make -j clean depend; make -j <NUM-CPU>
```
- For more detailed installation, please refer to [official tutorial](https://espnet.github.io/espnet/installation.html)

## Training
- Please check [the nycuka repo](egs2/nycuka/tts1/README.md)
