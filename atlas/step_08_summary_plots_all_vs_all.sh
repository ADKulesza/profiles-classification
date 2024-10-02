#!/bin/bash -xe

source header.sh

CONFUSION_MATRIX='false'
CONFUSION_MATRIX_REGION='false'
CONFIDENCE_PER_LABEL='false'
METRICS='false'
METRICS_VS_TYPE='true'

CAM_VISUALIZATION='false'
CAM_VISUALIZATION_PROFILE='false'
CAM_VISUALIZATION_SIG='false'

reformat_labels_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_REFORMAT_LABELS_DIR}

if [ ${CONFUSION_MATRIX} = 'true' ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/; do
      for model_set_dir in ${holdout_dir}/*/
      do
      cmat="${model_set_dir}cmat.npy"
      python ${CODEBASE_DIR}/step_08_plot_confusion_matrix.py \
      --area-order "areas_order.json"\
      --confmat ${cmat} \
      --figsize 15 \
      --output-confmat-plot ${cmat}
    done

#    for binary_cmat_npy in ${holdout_dir}/${STEP_06_BINARY_CMAT_DIR}/*".npy"; do
#      cmat="${binary_cmat_npy%%".npy"}"
#
#      python ${CODEBASE_DIR}/step_08_plot_binary_confusion_matrix.py \
#      --confmat ${binary_cmat_npy} \
#      --figsize 4 \
#      --output-confmat-plot ${cmat} \
#      --show-values
#    done

  done

  python ${CODEBASE_DIR}/step_08_plot_confusion_matrix.py \
  --area-order "areas_order.json" \
  --confmat ${STEP_06_EVALUATION_ALL_VS_ALL}/"cmat_mean.npy" \
  --figsize 15\
  --output-confmat-plot ${STEP_06_EVALUATION_ALL_VS_ALL}/"cmat_mean"

fi

if [ ${CONFUSION_MATRIX_REGION} = 'true' ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/; do
      for model_set_dir in ${holdout_dir}/*/
      do
        cmat_dir="${model_set_dir}region_cmat"
        python ${CODEBASE_DIR}/step_08_plot_region_cmat.py \
        --area-order "areas_order.json"\
        --input-directory ${cmat_dir} \
        --output-confmat-plot "${cmat_dir}"
    done
  done


fi

if [ ${METRICS} = 'true' ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/; do
    for model_set_dir in ${holdout_dir}/*_set_*/
      do
      python ${CODEBASE_DIR}/step_08_plot_metrics_zoom.py \
      --area-metrics-csv ${model_set_dir}/${STEP_06_METRICS_DIR}/"area_metrics.csv" \
      --output-dir ${model_set_dir}/${STEP_06_METRICS_DIR}
    done
  done

fi

if [ ${METRICS_VS_TYPE} = 'true' ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/holdout_*/; do
    for model_set_dir in ${holdout_dir}/*_set_*/
      do
        output_dir="${model_set_dir}/${STEP_08_METRICS_VS_DIR}"

        mkdir -p ${output_dir}

        python ${CODEBASE_DIR}/step_08_plot_metrics_vs_type.py \
        --area-metrics-csv ${model_set_dir}/${STEP_06_METRICS_DIR}/"area_metrics.csv" \
        --label-names "${LABEL_NAMES}" \
        --output-dir "${output_dir}"
    done
  done

fi

if [ ${CONFIDENCE_PER_LABEL} = 'true' ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/*/; do
    output_conf=${holdout_dir}/"confidence_per_label"
    mkdir -p ${output_conf}

    df_path="${holdout_dir}results.csv"

    python ${CODEBASE_DIR}/step_08_confidence_of_label_plot.py \
    --validation-csv ${df_path} \
    --areas-info ${STEP_06_EVALUATION_ALL_VS_ALL}/areas_info.csv \
    --output-dir ${output_conf}
  done
fi


if [ ${CAM_VISUALIZATION} = 'true' ]; then
  for holdout_dir in ${STEP_06_EVALUATION_ALL_VS_ALL}/*/; do
    for heatmap in ${holdout_dir}"heatmap*.npy"; do
      plots_dir=${holdout_dir}/'cam_heatmap_stat_plots'
      mkdir -p ${plots_dir}

      df_path="${holdout_dir}results.csv"
      python ${CODEBASE_DIR}/step_08_CAM_medians.py --config-fname ${CONFIG_FNAME} \
      --label-names ${LABEL_NAMES}\
      --heatmaps ${heatmap} \
      --output-csv ${df_path} \
      --output-dir ${plots_dir}
    done
  done
fi

#if [ ${CAM_VISUALIZATION_PROFILE} = 'true' ]
#  then
#    plots_dir=${output_hold_out}/'cam_heatmap_plots'
#    mkdir -p ${plots_dir}
#    python3 ${CODEBASE_DIR}/step_08_CAM_visualization3.py --config-fname ${CONFIG_FNAME} \
#            --label-names ${LABEL_NAMES}\
#            --input-norm-valid ${output_norm_dir}/${OUTPUT_VALID_NORM_PROFILES}\
#            --heatmaps ${output_hold_out}/heatmaps_plus.npy \
#            --output-csv ${output_hold_out}/${OUTPUT_EVALUATION_HOLD_OUT_CSV}\
#            --output-dir ${plots_dir}
#fi
#
#if [ ${CAM_VISUALIZATION_SIG} = 'true' ]
#  then
#    plots_dir=${output_hold_out}/'cam_heatmap_plots'
#    mkdir -p ${plots_dir}
#    python3 ${CODEBASE_DIR}/step_08_CAM_significance.py --config-fname ${CONFIG_FNAME} \
#            --label-names ${LABEL_NAMES}\
#            --input-norm-valid ${output_norm_dir}/${OUTPUT_VALID_NORM_PROFILES}\
#            --heatmaps ${output_hold_out}/heatmaps_plus.npy \
#            --output-csv ${output_hold_out}/${OUTPUT_EVALUATION_HOLD_OUT_CSV}\
#            --output-dir ${plots_dir}
#fi
#
#if [ ${PLOT_METRICS} = 'true' ]
#  then
#  plots_dir=${output_hold_out}/"metrics"
#  mkdir -p ${plots_dir}
#  python3 ${CODEBASE_DIR}/step_08_plot_metrics.py \
#          --area-metrics-csv ${output_hold_out}/"area_metrics.csv" \
#          --label-names ${LABEL_NAMES} \
#          --output-dir ${plots_dir}
#fi
