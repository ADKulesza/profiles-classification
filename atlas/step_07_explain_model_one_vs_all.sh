#!/bin/bash -xe

source header.sh

CAM_EXP='true'
DO_BOOTSTRAP='false'

if [ ${ALL_VS_ALL} = true ]; then

  output_picked_dir=${OUTPUT_DIR}/${OUTPUT_PICKED_PROFILES_DIR}

  output_norm_dir=${OUTPUT_DIR}/${OUTPUT_NORM_DATASETS_DIR}
  output_hold_out=${OUTPUT_DIR}/${OUTPUT_EVALUATION}/${OUTPUT_EVALUATION_NO_DATE}/${OUTPUT_EVALUATION_HOLD_OUT}

  if [ ${CAM_EXP} = true ]; then
    #    python3 ${CODEBASE_DIR}/step_07_cam_explain_model.py --config-fname ${CONFIG_FNAME}\
    #                --model-path ${OUTPUT_DIR}/${MODEL_NAME}\
    #                --input-norm-valid ${output_norm_dir}/${OUTPUT_VALID_NORM_PROFILES}\
    #                --output-heatmaps ${output_hold_out}/${OUTPUT_HEATMAPS}

    python3 ${CODEBASE_DIR}/step_07_CAM_plus.py --config-fname ${CONFIG_FNAME}\
    --model-path ${OUTPUT_DIR}/${MODEL_NAME}\
    --input-norm-valid ${output_norm_dir}/${OUTPUT_VALID_NORM_PROFILES}\
    --output-heatmaps ${output_hold_out}/${OUTPUT_HEATMAPS}
  fi

  if [ ${DO_BOOTSTRAP} = true ]; then
    python3 ${CODEBASE_DIR}/step_07_CAM_bootstrap.py --config-fname ${CONFIG_FNAME}\
    --output-heatmaps ${output_hold_out}/${OUTPUT_HEATMAPS}\
    --label-names ${LABEL_NAMES}\
    --output-csv ${output_hold_out}/${OUTPUT_EVALUATION_HOLD_OUT_CSV}
  fi

fi

if [ ${ONE_VS_ALL} = true ]; then
  output_dir=${OUTPUT_ONE_VS_ALL}/${OUTPUT_EVALUATION}/${OUTPUT_EVALUATION_NO_DATE}/${OUTPUT_EVALUATION_HOLD_OUT}/"heatmaps"
  mkdir -p ${output_dir}
  for f in ${OUTPUT_ONE_VS_ALL}/output_label_*; do
    model_path=${f}/${MODEL_NAME}

    model_num="${f//[!0-9]/}"

    python3 ${CODEBASE_DIR}/step_07_cam_explain_model.py --config-fname ${CONFIG_FNAME}\
    --model-path ${model_path}\
    --input-norm-valid ${OUTPUT_ONE_VS_ALL}/${OUTPUT_VALID_NORM_PROFILES}\
    --output-heatmaps ${output_dir}/heatmaps_${model_num}

    sleep 30s

  done
fi
