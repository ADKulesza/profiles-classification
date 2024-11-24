#!/bin/bash -xe

DO_HOLDOUT_PREDICT='false'
DO_HOLDOUT_CONFMAT='false'
DO_HOLDOUT_TYPES_CONFMAT='false'
DO_METRICS='false'
DO_ERRORS_ACROSS_TYPES='false'

DO_SECTION_EVALUATION='true'

export ROCM_PATH=/opt/rocm

source header.sh


#----------------------------------------------------------
# Evaluate profiles; ALL vs all approach
#----------------------------------------------------------

if [ ${DO_HOLDOUT_PREDICT} = true ]; then

#   [ -d ${STEP_06_EVALUATION_DIR} ] \\
#   && cp -r ${STEP_06_EVALUATION_DIR} ${STEP_06_EVALUATION_DIR}_`date +"%Y-%m-%d_%H_%M"`_backup

#  rm -rf ${STEP_06_EVALUATION_ALL_VS_ALL}
  mkdir -p "${STEP_06_EVALUATION_ALL_VS_ALL}"

#   --split-profiles-csv step_04_.../output_04_split_datasets/split_datasets_processed.csv
  python "${CODEBASE_DIR}"/step_06_01_get_holdout_datasets.py \
    --profiles ${STEP_04_OUTPUT_DIR}/${STEP_04_NORM_PROFILES} \
    --split-profiles-csv ${STEP_04_OUTPUT_DIR}/${STEP_04_SPLIT_DATASET} \
    --output-dir ${STEP_06_EVALUATION_ALL_VS_ALL} \
    --output-models-order ${STEP_06_EVALUATION_ALL_VS_ALL}/${STEP_06_MODELS_ORDER}


  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/*/
    do
      holdout_id=${holdout_dir#*/}
      holdout_id="${holdout_id%%/}"

      python ${CODEBASE_DIR}/step_06_02_predict.py \
        --profiles ${holdout_dir}"x_norm.npy" \
        --holdout-id ${holdout_id} \
        --profiles-csv ${holdout_dir}/"holdout_info.csv" \
        --models-info ${STEP_05_MODELS}/${STEP_05_MODEL_INFO_CSV} \
        --models-order ${STEP_06_EVALUATION_ALL_VS_ALL}/${STEP_06_MODELS_ORDER} \
        --output-dir ${STEP_06_EVALUATION_ALL_VS_ALL}

      sleep 10;
    done

fi


if [ ${DO_HOLDOUT_CONFMAT} = true ]; then

  reformat_labels_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_REFORMAT_LABELS_DIR}

  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/
    do
      for model_set_dir in ${holdout_dir}/*_set_*/
        do
          python ${CODEBASE_DIR}/step_06_03_get_cmat.py \
          --validation-csv ${model_set_dir}/"results.csv"\
          --output-dir ${model_set_dir}

      done
    done

  python ${CODEBASE_DIR}/step_06_03_get_mean_cmat.py \
    --evaluation-dir ${STEP_06_EVALUATION_ALL_VS_ALL}\
    --output ${STEP_06_EVALUATION_ALL_VS_ALL}/"cmat_mean.npy"


fi

if [ ${DO_HOLDOUT_TYPES_CONFMAT} = true ]; then

  reformat_labels_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_REFORMAT_LABELS_DIR}

  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/
    do
      for model_set_dir in ${holdout_dir}/*_set_*/
        do
          output_dir="${model_set_dir}/${STEP_06_TYPE_CMAT_DIR}"
          mkdir -p ${output_dir}

          python ${CODEBASE_DIR}/step_06_04_get_type_cmat.py \
          --validation-csv ${model_set_dir}/"results.csv"\
          --output-dir ${output_dir}

      done
    done


fi


if [ ${DO_METRICS} = true ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/
    do
      for model_set_dir in ${holdout_dir}/*_set_*/
        do
          mkdir -p ${model_set_dir}/${STEP_06_METRICS_DIR}

          python ${CODEBASE_DIR}/step_06_05_get_metrics.py \
          --validation-csv ${model_set_dir}/"results.csv"\
          --output-dir ${model_set_dir}/${STEP_06_METRICS_DIR}
      done

  done


  python ${CODEBASE_DIR}/step_06_06_aggregate_metrics.py \
    --evaluation-dir ${STEP_06_EVALUATION_ALL_VS_ALL}\
    --output ${STEP_06_EVALUATION_ALL_VS_ALL}/"all_macro_metrics.csv" \
    --area-output ${STEP_06_EVALUATION_ALL_VS_ALL}/"all_area_metrics.csv"


fi


if [ ${DO_SECTION_EVALUATION} = true ]; then


  sections_dir=${STEP_06_EVALUATION_ALL_VS_ALL}/${STEP_06_SECTION_EVALUATION}

#  mkdir -p "${sections_dir}"
#
#  python ${CODEBASE_DIR}/step_06_07_get_holdout_sections_evaluation_sets.py \
#            --config-fname "${CONFIG_FNAME}" \
#            --areas-def "${LABEL_NAMES}" \
#            --profiles "${STEP_03_HOLDOUT_DIR}/${STEP_03_SECTIONS_HO_PROFILES}" \
#            --profiles-csv "${STEP_03_HOLDOUT_DIR}/${STEP_03_SECTIONS_HO_PROFILES_CSV}" \
#            --output-profiles "${sections_dir}/${STEP_06_NORM_PROFILES}" \
#            --output-y "${sections_dir}/y_true.npy" \
#            --output-df "${sections_dir}/${STEP_06_SECTION_CSV}"
  for model_path in ${STEP_05_MODELS}/*/
    do

      model_dir=${model_path#*/}
      model_dir="${model_dir%%/}"

      output_dir="${sections_dir}/${model_dir}"
      mkdir -p "${output_dir}"

      python ${CODEBASE_DIR}/step_06_08_predict_sections_all_vs_all.py \
        --profiles "${sections_dir}/${STEP_06_NORM_PROFILES}" \
        --profiles-csv "${sections_dir}/${STEP_06_SECTION_CSV}" \
        --model-path "${model_path}" \
        --output-dir "${output_dir}"



    done


#  for model_set_dir in ${STEP_05_MODELS}/*/
#    do
#
#      _set_dir=${model_set_dir%*/}
#      _set_dir=${_set_dir##*/}
#
#      section_model_dir=${sections_dir}/${_set_dir}
#    done

fi


if [ ${DO_ERRORS_ACROSS_TYPES} = true ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/
    do
      for model_set_dir in ${holdout_dir}/*_set_*/
        do
          output_dir="${model_set_dir}/${STEP_06_ERRORS_ACROSS_TYPE}"
          mkdir -p ${output_dir}

#          python ${CODEBASE_DIR}/step_06_10_errors_across_type.py \
#          --validation-csv ${model_set_dir}/"results.csv"\
#          --output-dir ${output_dir}

          python ${CODEBASE_DIR}/step_06_09_errors_across_type.py \
          --areas-def ${LABEL_NAMES} \
          --validation-csv ${model_set_dir}/"results.csv"\
          --cmat ${model_set_dir}/"cmat.npy"\
          --output-dir ${output_dir}
#
      done
  break
  done

fi