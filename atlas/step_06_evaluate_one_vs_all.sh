#!/bin/bash -xe

ONE_VS_ALL_HOLDOUT='false'
DO_GLOBAL_HOLDOUT='true'

DO_HOLDOUT_CONFMAT='false'
DO_METRICS='false'

export ROCM_PATH=/opt/rocm

source header.sh


#----------------------------------------------------------
# HOLDOUT DATASET
#----------------------------------------------------------
if [ ${ONE_VS_ALL_HOLDOUT} = true ]; then

  #  [ -d ${STEP_06_EVALUATION_ONE_VS_ALL} ] && cp -r ${STEP_06_EVALUATION_ONE_VS_ALL} ${STEP_06_EVALUATION_ONE_VS_ALL}_`date +"%Y-%m-%d_%H_%M"`_backup

  #  rm -rf ${STEP_06_EVALUATION_ONE_VS_ALL}
  mkdir -p ${STEP_06_EVALUATION_ONE_VS_ALL}

  for label_dir in ${STEP_04_ONE_VS_ALL}/*/; do

    label_id=${label_dir#*/}
    label_id=${label_id%%/}

    output_dir="${STEP_06_EVALUATION_ONE_VS_ALL}/${label_id}"
    mkdir -p "${output_dir}"

    python "${CODEBASE_DIR}"/step_06_01_get_holdout_datasets.py \
      --profiles "${STEP_04_ONE_VS_ALL}/${STEP_04_NORM_PROFILES}" \
      --split-profiles-csv "${label_dir}/${STEP_04_SPLIT_DATASET}" \
      --output-dir "${output_dir}"\
      --output-models-order "${output_dir}/${STEP_06_MODELS_ORDER}"

    for holdout_dir in ${output_dir}/*/
      do
        holdout_id=${holdout_dir#*/}
        holdout_id=${holdout_id#*/}
        holdout_id="${holdout_id%%/}"

        python "${CODEBASE_DIR}/step_06_02_predict.py" \
          --profiles "${holdout_dir}x_norm.npy" \
          --holdout-id "${holdout_id}" \
          --profiles-csv "${holdout_dir}/holdout_info.csv" \
          --models-info "${label_dir}/${STEP_05_MODEL_INFO_CSV}" \
          --models-order "${output_dir}/${STEP_06_MODELS_ORDER}" \
          --output-dir "${output_dir}"

        sleep 3;
      done
  done

fi


if [ ${DO_GLOBAL_HOLDOUT} = true ]; then

  HOLDOUT_LIST=(
    "holdout_0"
)

  summ_output=${STEP_06_EVALUATION_ONE_VS_ALL}/${STEP_06_SUMMARY_ONE_VS_ALL}
  mkdir -p ${summ_output}

  for holdout_id in "${HOLDOUT_LIST[@]}"; do
    summ_holdout_dir=${summ_output}/${holdout_id}
    mkdir -p ${summ_holdout_dir}
    echo $summ_holdout_dir
    python ${CODEBASE_DIR}/step_06_one_vs_all_winner_takes_all.py \
      --evaluation-dir ${STEP_06_EVALUATION_ONE_VS_ALL} \
      --profiles-csv "${STEP_03_HOLDOUT_DIR}/${STEP_03_TO_TRAIN_CSV}" \
      --holdout-id ${holdout_id} \
      --models-order ${STEP_06_EVALUATION_ONE_VS_ALL}/"model_order.json"\
      --output ${summ_holdout_dir}

  done

fi


if [ ${DO_HOLDOUT_CONFMAT} = true ]; then

  for label_dir in ${STEP_06_EVALUATION_ONE_VS_ALL}/[0-9]*/; do

    label_id=${label_dir#*/}
    label_id=${label_id%%/}


    reformat_labels_dir=${STEP_04_ONE_VS_ALL}/${label_id}/${STEP_04_REFORMAT_LABELS_DIR}

    for holdout_dir in ${label_dir}*/
    do

      holdout_id=${holdout_dir#*/}
      holdout_id=${holdout_id#*/}

      for model_set_dir in ${holdout_dir}/*_set_*/
        do

        python ${CODEBASE_DIR}/step_06_03_get_cmat.py \
            --validation-csv ${model_set_dir}/"results.csv"\
            --labels-processed ${reformat_labels_dir}/${STEP_04_LABELS_PROCESSED} \
            --output-dir ${model_set_dir} \
            --one-vs-all

        done
    done

    python ${CODEBASE_DIR}/step_06_03_get_mean_cmat.py \
    --evaluation-dir ${label_dir}\
    --output ${label_dir}"cmat_mean.npy"

  done
fi


if [ ${DO_METRICS} = true ]; then

  for label_dir in ${STEP_06_EVALUATION_ONE_VS_ALL}/[0-9]*/; do
    for holdout_dir in ${label_dir}*/
      do
        for model_set_dir in ${holdout_dir}/*_set_*/
          do
            mkdir -p ${model_set_dir}/${STEP_06_METRICS_DIR}

            python ${CODEBASE_DIR}/step_06_04_get_metrics.py \
            --validation-csv ${model_set_dir}/"results.csv" \
            --output-dir ${model_set_dir}/${STEP_06_METRICS_DIR} \
            --one-vs-all
        done
    done

    python ${CODEBASE_DIR}/step_06_05_aggregate_metrics.py \
    --evaluation-dir ${label_dir}\
    --output ${label_dir}/"all_macro_metrics.csv" \
    --area-output ${label_dir}/"all_area_metrics.csv" \
    --one-vs-all
  done
fi

