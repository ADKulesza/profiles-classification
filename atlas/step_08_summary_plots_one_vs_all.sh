#!/bin/bash -xe

source header.sh

CONFIDENCE_PER_LABEL='false'
CONFUSION_MATRIX='true'
PLOT_METRICS='false'
CAM_VISUALIZATION='false'
CAM_VISUALIZATION_PROFILE='true'
CAM_VISUALIZATION_SIG='false'

#evaluation_dir=${OUTPUT_ONE_VS_ALL}/${OUTPUT_EVALUATION}
#
#current_result_dir=${evaluation_dir}/${OUTPUT_EVALUATION_NO_DATE}
#
#
#output_dir=${current_result_dir}/${OUTPUT_EVALUATION_HOLD_OUT}

if [ ${CONFUSION_MATRIX} = 'true' ]; then
  for label_dir in ${STEP_06_EVALUATION_ONE_VS_ALL}/*; do
    label_id=${label_dir#*/}
    label_id=${label_id%%/*}
    for holdout_dir in ${label_dir}/holdout_*/
      do
        binary_cmat_npy=${holdout_dir}"cmat.npy"
        python ${CODEBASE_DIR}/step_08_plot_binary_confusion_matrix.py \
          --confmat ${binary_cmat_npy} \
          --figsize 4 \
          --output-confmat-plot ${holdout_dir}"cmat" \
          --show-values
    done
  done
fi