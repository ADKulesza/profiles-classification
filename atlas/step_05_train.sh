#!/bin/bash -xe

DO_TRAIN='true'

source header.sh

MODEL_NAME='multi_branch_model'


if [ ${DO_TRAIN} = 'true' ]; then
  cd ${WORK_DIR}

#  [ -d ${STEP_05_MODELS} ] && cp -r ${STEP_05_MODELS} ${STEP_05_MODELS}_`date +"%Y-%m-%d_%H_%M"`_backup
#  [ -d ${STEP_04_OUTPUT_DIR}  ] && cp -r ${STEP_04_OUTPUT_DIR} ${STEP_05_MODELS}_`date +"%Y-%m-%d_%H_%M"`_backup/${STEP_04_OUTPUT_DIR}

  rm -rf ${STEP_05_MODELS}
  mkdir -p ${STEP_05_MODELS}
  python3 ${CODEBASE_DIR}/step_05_train.py --config-fname ${CONFIG_FNAME} \
    --model-name ${MODEL_NAME} \
    --profiles ${STEP_04_OUTPUT_DIR}/${STEP_04_NORM_PROFILES} \
    --labels ${STEP_04_OUTPUT_DIR}/${STEP_04_LABELS_DATASET} \
    --profiles-csv ${STEP_04_OUTPUT_DIR}/${STEP_04_SPLIT_DATASET} \
    --model-info ${STEP_05_MODEL_INFO_CSV} \
    --output-dir ${STEP_05_MODELS}
fi
