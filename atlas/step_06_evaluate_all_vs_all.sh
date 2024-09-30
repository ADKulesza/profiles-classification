#!/bin/bash -xe

DO_HOLDOUT_PREDICT='false'
DO_HOLDOUT_CONFMAT='false'
DO_METRICS='true'

DO_SECTION_EVALUATION='false'

source header.sh

output_norm_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_NORM_DIR}
output_split_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_SPLIT_DATASETS_DIR}


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
    --profiles ${output_norm_dir}/${STEP_04_NORM_PROFILES} \
    --split-profiles-csv ${output_split_dir}/${STEP_04_SPLIT_DATASET} \
    --label-names ${LABEL_NAMES} \
    --area-order "areas_order.json"\
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

  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/*/
    do
      for model_set_dir in ${holdout_dir}/*/
        do
          python ${CODEBASE_DIR}/step_06_03_get_cmat.py \
          --validation-csv ${model_set_dir}/"results.csv"\
          --labels-processed ${reformat_labels_dir}/${STEP_04_LABELS_PROCESSED} \
          --output-dir ${model_set_dir}

          mkdir -p ${model_set_dir}/${STEP_06_BINARY_CMAT_DIR}

#          python ${CODEBASE_DIR}/step_06_get_binary_cmat.py \
#            --confmat ${holdout_dir}/"cmat.npy" \
#            --label-names ${LABEL_NAMES} \
#            --labels-processed ${reformat_labels_dir}/${STEP_04_LABELS_PROCESSED} \
#            --output ${holdout_dir}/${STEP_06_BINARY_CMAT_DIR}
      done
    done

  python ${CODEBASE_DIR}/step_06_03_get_mean_cmat.py \
    --evaluation-dir ${STEP_06_EVALUATION_ALL_VS_ALL}\
    --output ${STEP_06_EVALUATION_ALL_VS_ALL}/"cmat_mean.npy"


fi

if [ ${DO_METRICS} = true ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/
    do
      for model_set_dir in ${holdout_dir}/*_set_*/
        do
          mkdir -p ${model_set_dir}/${STEP_06_METRICS_DIR}

          python ${CODEBASE_DIR}/step_06_04_get_metrics.py \
          --validation-csv ${model_set_dir}/"results.csv"\
          --output-dir ${model_set_dir}/${STEP_06_METRICS_DIR}
      done

  done


  python ${CODEBASE_DIR}/step_06_05_aggregate_metrics.py \
    --evaluation-dir ${STEP_06_EVALUATION_ALL_VS_ALL}\
    --output ${STEP_06_EVALUATION_ALL_VS_ALL}/"all_macro_metrics.csv" \
    --area-output ${STEP_06_EVALUATION_ALL_VS_ALL}/"all_area_metrics.csv"


fi


if [ ${DO_SECTION_EVALUATION} = true ]; then

  reformat_labels_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_REFORMAT_LABELS_DIR}
  sections_dir=${STEP_06_EVALUATION_ALL_VS_ALL}/${STEP_06_SECTION_EVALUATION}

  mkdir -p "${sections_dir}"

  python ${CODEBASE_DIR}/step_06_06_get_holdout_sections_evaluation_sets.py \
            --config-fname ${CONFIG_FNAME} \
            --profiles ${STEP_03_HOLDOUT_DIR}/${STEP_03_SECTIONS_HO_PROFILES} \
            --profiles-csv ${STEP_03_HOLDOUT_DIR}/${STEP_03_SECTIONS_HO_PROFILES_CSV} \
            --label-names ${LABEL_NAMES} \
            --area-order "areas_order.json"\
            --labels-processed ${reformat_labels_dir}/${STEP_04_LABELS_PROCESSED} \
            --output-profiles ${sections_dir}/${STEP_06_NORM_PROFILES} \
            --output-y ${sections_dir}/y_true.npy \
            --output-df ${sections_dir}/${STEP_06_SECTION_CSV}

  for model_set_dir in ${STEP_05_MODELS}/*/
    do

      _set_dir=${model_set_dir%*/}
      _set_dir=${_set_dir##*/}

      section_model_dir=${sections_dir}/${_set_dir}
#      mkdir -p ${section_model_dir}

#      python ${CODEBASE_DIR}/step_06_07_predict_sections_all_vs_all.py \
#        --profiles ${sections_dir}/${STEP_06_NORM_PROFILES} \
#        --profiles-csv ${sections_dir}/${STEP_06_SECTION_CSV} \
#        --labels-processed ${reformat_labels_dir}/${STEP_04_LABELS_PROCESSED} \
#        --label-names ${LABEL_NAMES} \
#        --area-order "areas_order.json" \
#        --model-path ${model_set_dir} \
#        --output-dir ${section_model_dir}
#
#        python ${CODEBASE_DIR}/step_06_08_get_vtk_sections.py --config-fname ${CONFIG_FNAME}\
#        --profiles-csv ${section_model_dir}/"results.csv" \
#        --streamlines-dir ${STEP_01_STREAMLINES} \
#        --output ${section_model_dir}
    done

fi
