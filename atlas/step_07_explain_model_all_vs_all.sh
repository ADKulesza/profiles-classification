#!/bin/bash -xe

source header.sh

CAM_EXP='false'
DO_BOOTSTRAP='true'

#output_picked_dir=${OUTPUT_DIR}/${OUTPUT_PICKED_PROFILES_DIR}
#
#output_norm_dir=${OUTPUT_DIR}/${OUTPUT_NORM_DATASETS_DIR}
#output_hold_out=${OUTPUT_DIR}/${OUTPUT_EVALUATION}/${OUTPUT_EVALUATION_NO_DATE}/${OUTPUT_EVALUATION_HOLD_OUT}

output_picked_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_PICK_PROFILES_DIR}

output_norm_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_NORM_DIR}

if [ ${CAM_EXP} = true ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/; do

    holdout_id=${holdout_dir#*/}
    holdout_id="${holdout_id%%/}"

    python ${CODEBASE_DIR}/step_07_get_heatmaps.py --config-fname ${CONFIG_FNAME}\
        --profiles ${holdout_dir}"x_norm.npy" \
        --holdout-id ${holdout_id} \
        --profiles-csv ${holdout_dir}/"holdout_info.csv" \
        --models-info ${STEP_05_MODELS}/${STEP_05_MODEL_INFO_CSV} \
        --models-order ${STEP_06_EVALUATION_ALL_VS_ALL}/${STEP_06_MODELS_ORDER} \
        --output-dir ${STEP_06_EVALUATION_ALL_VS_ALL}

  done
fi

if [ ${DO_BOOTSTRAP} = true ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/; do

    holdout_id=${holdout_dir#*/}
    holdout_id="${holdout_id%%/}"
      python ${CODEBASE_DIR}/step_07_heatmaps_bootstrap.py --config-fname ${CONFIG_FNAME}\
        --label-names ${LABEL_NAMES}\
        --holdout-id ${holdout_id} \
        --profiles-csv ${holdout_dir}/"holdout_info.csv" \
        --output-dir ${holdout_dir}
    done
fi
