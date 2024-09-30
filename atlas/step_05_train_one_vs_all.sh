#!/bin/bash -xe

DO_TRAIN='true'

MODEL_NAME='multi_branch_binary_model'

source header.sh

#----------------------------------------------------------
# Train model
#----------------------------------------------------------
if [ ${DO_TRAIN} = true ]; then

  for label_dir in ${STEP_04_ONE_VS_ALL}/*/
  do
    # ---
    reformat_labels_dir=${label_dir}${STEP_04_REFORMAT_LABELS_DIR}
    output_norm_dir=${label_dir}${STEP_04_NORM_DIR}

    label_id=${label_dir#*/}

#    if [[ $label_id = 1* ]]; then
#      continue
#    fi

    mkdir -p ${STEP_05_MODELS_ONE_VS_ALL}
    models_dir=${STEP_05_MODELS_ONE_VS_ALL}/${label_id}
    mkdir -p ${models_dir}
    # ---

    python3 ${CODEBASE_DIR}/step_05_train.py --config-fname ${label_dir}/${CONFIG_FNAME}\
            --model-name ${MODEL_NAME} \
            --profiles ${output_norm_dir}/${STEP_04_NORM_PROFILES} \
            --labels ${reformat_labels_dir}/${STEP_04_LABELS_DATASET} \
            --profiles-csv ${label_dir}${STEP_04_LABELS_INFO} \
            --model-info ${STEP_05_MODEL_INFO_CSV} \
            --output-dir ${models_dir} \
            --one-vs-all
  sleep 10;
  done

fi
