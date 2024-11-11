#!/bin/bash -xe

source header.sh

export ROCM_PATH=/opt/rocm

CAM_EXP='true'
DO_BOOTSTRAP='false'



if [ ${CAM_EXP} = true ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/; do
    for model_set_dir in ${holdout_dir}/*_set_*/
        do
      holdout_id=${holdout_dir#*/}
      holdout_id="${holdout_id%%/}"


      model_id="${model_set_dir#*/}"
      output="${STEP_06_EVALUATION_ALL_VS_ALL}/${model_id}/${STEP_07_HEATMAPS}"

      model_id="${model_id#*/}"
      model_path="${STEP_05_MODELS}/${model_id}"

      mkdir -p ${output}

      python ${CODEBASE_DIR}/step_07_get_heatmaps.py --config-fname ${CONFIG_FNAME}\
          --profiles ${holdout_dir}"x_norm.npy" \
          --model ${model_path} \
          --output ${output}
    done
  done
fi

if [ ${DO_BOOTSTRAP} = true ]; then
#  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/; do
#    for model_set_dir in ${holdout_dir}/*_set_*/
#        do
#            heatmap_dir="${model_set_dir}/${STEP_07_HEATMAPS}"
            model_set_dir="step_06_evaluation_all_vs_all/holdout_2/multi_branch_model_set_20"
            heatmap_dir="step_06_evaluation_all_vs_all/holdout_2/multi_branch_model_set_20/${STEP_07_HEATMAPS}"

            python ${CODEBASE_DIR}/step_07_heatmaps_bootstrap.py --config-fname "${CONFIG_FNAME}"\
              --profiles-csv ${model_set_dir}/"results.csv" \
              --heatmap "${heatmap_dir}/heatmap.npy" \
              --output-dir "${heatmap_dir}"
#        done
#    done
fi
