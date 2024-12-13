#!/bin/bash -xe

DO_TRAIN='true'

export ROCM_PATH=/opt/rocm

source header.sh

MODEL_NAME='simple_model'
#----------------------------------------------------------
# Train model
#----------------------------------------------------------
if [ ${DO_TRAIN} = true ]; then

  for label_dir in ${STEP_04_ONE_VS_ALL}/*/
  do
    label_id=${label_dir#*/}
    label_id=${label_id%%/}

#    if [[ $label_id -ge 10 ]]; then
#      continue
#    fi
#
#    if [[ $label_id -le 4 ]]; then
#      continue
#    fi

    mkdir -p ${STEP_05_MODELS_ONE_VS_ALL}
    models_dir=${STEP_05_MODELS_ONE_VS_ALL}/${label_id}
    mkdir -p ${models_dir}
    # ---

    python3 ${CODEBASE_DIR}/step_05_train.py --config-fname ${label_dir}/${CONFIG_FNAME}\
            --model-name ${MODEL_NAME} \
            --profiles ${STEP_04_ONE_VS_ALL}/${STEP_04_NORM_PROFILES} \
            --labels ${label_dir}/${STEP_04_LABELS_DATASET} \
            --profiles-csv ${label_dir}${STEP_04_SPLIT_DATASET} \
            --model-info ${STEP_05_MODEL_INFO_CSV} \
            --output-dir ${models_dir} \
            --one-vs-all
  sleep 10;
  done
fi
