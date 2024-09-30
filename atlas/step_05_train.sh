#!/bin/bash -xe

DO_TRAIN='true'

source header.sh

MODEL_NAME='multi_branch_model'

reformat_labels_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_REFORMAT_LABELS_DIR}
output_split_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_SPLIT_DATASETS_DIR}
output_norm_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_NORM_DIR}
models_dir=${STEP_05_MODELS}

if [ ${DO_TRAIN} = 'true' ]; then
  cd ${WORK_DIR}

#  [ -d ${models_dir} ] && cp -r ${models_dir} ${models_dir}_`date +"%Y-%m-%d_%H_%M"`_backup
#  [ -d ${STEP_04_OUTPUT_DIR}  ] && cp -r ${STEP_04_OUTPUT_DIR} ${models_dir}_`date +"%Y-%m-%d_%H_%M"`_backup/${STEP_04_OUTPUT_DIR}

  rm -rf ${models_dir}
  mkdir -p ${models_dir}
  python3 ${CODEBASE_DIR}/step_05_train.py --config-fname ${CONFIG_FNAME} \
    --model-name ${MODEL_NAME} \
    --profiles ${output_norm_dir}/${STEP_04_NORM_PROFILES} \
    --labels ${reformat_labels_dir}/${STEP_04_LABELS_DATASET} \
    --profiles-csv ${output_split_dir}/${STEP_04_SPLIT_DATASET} \
    --model-info ${STEP_05_MODEL_INFO_CSV} \
    --output-dir ${models_dir}
fi
