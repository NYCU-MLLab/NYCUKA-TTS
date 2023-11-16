#!/usr/bin/env bash

# set -e
# set -u
# set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=3
threshold=35
nj=40

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# if [ -z "${AISHELL3}" ]; then
#    log "Fill the value of 'AISHELL3' of db.sh"
#    exit 1
# fi
db_root="downloads/chatbot recordings/audio_file/split"
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ ! -e "${db_root}" ]; then
    mkdir -p downloads
    log "stage -1: download data from google drive"
    cd downloads
    # We do not provide the dataset
    log "Error occured because no dataset link found"
    unzip *.zip
    cd -
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: prepare chatbot data"
    # mkdir -p data
    # for x in all; do
    #     mkdir -p data/${x}
    #     python local/data_prep.py --src "${db_root}" --dest data/${x} --espnet_g2p true
    #     sort data/${x}/utt2spk -o data/${x}/utt2spk
    #     sort data/${x}/wav.scp -o data/${x}/wav.scp
    #     sort data/${x}/text -o data/${x}/text
    #     utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
    #     utils/validate_data_dir.sh --no-feats data/${x}
    # done

    for x in all_phn; do
        mkdir -p data/${x}
        python local/data_prep.py --src "${db_root}" --dest data/${x} --espnet_g2p false
        sort data/${x}/utt2spk -o data/${x}/utt2spk
        sort data/${x}/wav.scp -o data/${x}/wav.scp
        sort data/${x}/text -o data/${x}/text
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
        utils/validate_data_dir.sh --no-feats data/${x}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: scripts/audio/trim_silence.sh"
    for x in all_phn; do
        # shellcheck disable=SC2154
        scripts/audio/trim_silence.sh \
             --cmd "${train_cmd}" \
             --nj "${nj}" \
             --fs 44100 \
             --win_length 2048 \
             --shift_length 512 \
             --threshold "${threshold}" \
             data/${x} data/${x}/log

        utils/fix_data_dir.sh data/${x}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: split for development set"
    # utils/subset_data_dir.sh data/all 50 data/dev
    # utils/copy_data_dir.sh data/all data/train_no_dev
    # utils/filter_scp.pl --exclude data/dev/wav.scp \
    #     data/all/wav.scp > data/train_no_dev/wav.scp
    # utils/fix_data_dir.sh data/train_no_dev

    # utils/subset_data_dir.sh data/train_no_dev 50 data/test
    # utils/copy_data_dir.sh data/train_no_dev data/train_no_dev_test
    # utils/filter_scp.pl --exclude data/test/wav.scp \
    #     data/train_no_dev/wav.scp > data/train_no_dev_test/wav.scp
    # utils/fix_data_dir.sh data/train_no_dev_test
    #for phn
    utils/subset_data_dir.sh data/all_phn 50 data/dev_phn
    utils/copy_data_dir.sh data/all_phn data/train_no_dev_phn
    utils/filter_scp.pl --exclude data/dev_phn/wav.scp \
        data/all_phn/wav.scp > data/train_no_dev_phn/wav.scp
    utils/fix_data_dir.sh data/train_no_dev_phn

    utils/subset_data_dir.sh data/train_no_dev_phn 50 data/test_phn
    utils/copy_data_dir.sh data/train_no_dev_phn data/train_no_dev_test_phn
    utils/filter_scp.pl --exclude data/test_phn/wav.scp \
        data/train_no_dev_phn/wav.scp > data/train_no_dev_test_phn/wav.scp
    utils/fix_data_dir.sh data/train_no_dev_test_phn
    
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

