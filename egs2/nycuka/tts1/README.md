# The tutorial on re-run the experiments of Nycuka
## Data Preparing

### Checkpoint and inference file
#### For checkpoint and the inference file. Please access from here:
- [One Drive](https://nycu1-my.sharepoint.com/:u:/g/personal/lijen0918_ee10_m365_nycu_edu_tw/EYDFxirwOBxGgWWvA1dH1VkB9gw6YBP6Bo9EB8Rdc4hVLA?e=OepexQ)
#### This is consisent with the code provided for nycuGPT

### 1. Run the recipe until stage 
```sh
$ ./run.sh --stop-stage 5 
```
## AR model case (Tacotron2 / Transformer)
### This is for prepareing duration.
- Please down the pre-trained model below
```sh

### 2. Download pretrained model
## downloads aishell3_tacotron
pip install onedrivedownloader
mkdir -p downloads/aishell3_tacotron
python local/downloadFromOneDrive.py -u https://nycu1-my.sharepoint.com/:u:/g/personal/lijen0918_ee10_m365_nycu_edu_tw/EXOgGdhlD0VAjjIDUHwhUKIB6l_E0PwYC9gZIQGuwByiMA?e=CfCp5d -f "downloads/aishell3_tacotron"
## downloads aishell3_fastspeech2
mkdir -p downloads/aishell3_fastspeech2
python local/downloadFromOneDrive.py -u https://nycu1-my.sharepoint.com/:u:/g/personal/lijen0918_ee10_m365_nycu_edu_tw/Ea7QMCwl8npIrU17Y5A5HMEBUOw-H9Csfpq_RJl362yegw?e=XwZN7R -f "downloads/aishell3_fastspeech2"
cd -

```

### 3. Replace token list with pretrained model's one

Since we use the same language data for fine-tuning, we need to use the token list of the pretrained model instead of that of data for fine-tuning.
The downloaded pretrained model has `tokens_list` in the config, so first we create `tokens.txt` (`token_list`) from the config.

```sh
pyscripts/utils/make_token_list_from_config.py downloads/aishell3_tacotron/exp/tts_train_raw_phn_pypinyin_g2p_phone/config.yaml

# tokens.txt is created in model directory
$ ls downloads/aishell3_tacotron/exp/tts_train_raw_phn_pypinyin_g2p_phone/

```

Let us replace the `tokens.txt` with pretrained model's one.
```sh
# Make backup (Rename -> *.bak)
$ mv dump/token_list/phn_none/tokens.{txt,txt.bak}
# Make symlink to pretrained model's one (Just copy is also OK)
$ ln -sf $(pwd)/downloads/aishell3_tacotron/exp/tts_train_raw_phn_pypinyin_g2p_phone/tokens.txt dump/token_list/phn_none
```

### 4 (Optional). Replace statistics with pretrained model's one

Sometimes, using the feature statistics of the pretrained models is better than using that of adaptation data.
This is an optional step, so you can skip if you use the original statistics.

```sh
# Make backup (Rename -> *.bak)
$ mv exp/tts_stats_raw_phn_none/train/feats_stats.{npz,npz.bak}
# Make symlink to pretrained model's one (Just copy is also OK)
$ ln -s $(pwd)/downloads/aishell3_tacotron/exp/tts_stats_raw_phn_pypinyin_g2p_phone/train/feats_stats.npz exp/tts_stats_raw_phn_none/train
```
### 5. Run fine-tuning

Run the recipe from stage 6.

You need to specify `--init_param` for `--train_args` to load pretrained parameters (Or you can write them in `*.yaml` config).
Here `--init_param /path/to/model.pth:a:b` represents loading "a" parameters in model.pth into "b", and `:tts:tts` means load parameters except for the feature normalizer.

```sh
# Recommend using --tag to name the experiment directory
$ ./run.sh \
    --stage 6 \
    --train_config conf/tuning/finetune_tacotron2.yaml \
    --train_args "--init_param downloads/aishell3_tacotron/exp/tts_train_raw_phn_pypinyin_g2p_phone/123epoch.pth:tts:tts" \
    --tag finetune_tacotron2
```

For more complex loading of pretrained parameters, please check [`How to load pretrained model?`](../../TEMPLATE/tts1/README.md#how-to-load-the-pretrained-model) For example, if you want to perform fine-tuning of English model with Japanese data, you may want to load the network except for the token embedding layer.


## Non-AR model case (FastSpeech / FastSpeech2)

To finetune non-AR models, we need to preapre `durations` file.
Therefore, at first, please finish the finetuning of AR models by the above steps.

Here, we show the procedure of FastSpeech2 fine-tuning with the above fine-tuened tacotron2 as the teacher.

### 1. Prepare durations file using the adapted AR model

First, prepare the `durations` for all sets by running AR model inference with teacher forcing.

```sh
$ ./run.sh \
    --stage 7 \
    --tts_exp exp/tts_finetune_tacotron2 \
    --inference_model train.loss.ave_5best.pth \
    --inference_args "--use_teacher_forcing true" \
    --test_sets "train_no_dev_test_phn dev_phn test_phn"
```

You can find `durations` files in `exp/tts_finetune_tacotron2/decode_use_teacher_forcingtrue_train.loss.ave_5best/*`.


Please make sure this model used the same `token_list` as the teacher AR model.

### 3. Run fine-tuning

Here we skip the replacement of the statistics (Of course you can do it).
And we assume that `tokens.txt` is already replaced in AR model fine-tuning.

Since fastspeech2 requires extra feature calculation, run from stage 5.

```sh
# Recommend using --tag to name the experiment directory
$ ./run.sh \
    --stage 5 \
    --lang zh \
    --feats_type raw \
    --token_type phn \
    --cleaner none \
    --use_xvector true \
    --gpu_inference true \
    --write_collected_feats true \
    --teacher_dumpdir exp/tts_finetune_tacotron2/decode_use_teacher_forcingtrue_train.loss.ave_5best \
    --tts_stats_dir exp/tts_finetune_tacotron2/decode_use_teacher_forcingtrue_train.loss.ave_5best/stats \
    --train_config conf/tuning/finetune_fastspeech2.yaml  \
    --train_args "--init_param downloads/aishell3_fastspeech2/exp/tts_train_gst+xvector_conformer_fastspeech2_raw_phn_pypinyin_g2p_phone/train.loss.ave_5best.pth:tts:tts" \
    --tag finetune_fastpeech2_g2pw_handover \
```

