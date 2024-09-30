#!/bin/bash -xe

DO_CREATE_DATASETS='true'
DO_VISUALIZATION='false'
DO_SPLIT_DATASETS='true'
DO_NORMALIZATION='true'

source header.sh

mkdir -p ${STEP_04_OUTPUT_DIR}

output_picked_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_PICK_PROFILES_DIR}
reformat_labels_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_REFORMAT_LABELS_DIR}
output_split_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_SPLIT_DATASETS_DIR}
output_norm_dir=${STEP_04_OUTPUT_DIR}/${STEP_04_NORM_DIR}

#----------------------------------------------------------
# Create datasets
#----------------------------------------------------------

if [ ${DO_CREATE_DATASETS} = 'true' ]
then


  mkdir -p ${output_picked_dir}

#  profile_arrays_dir=${PROFILES_OUTPUT}/${PROFILES_ARRAYS_OUTPUT}

  python ${CODEBASE_DIR}/step_04_areas_picker.py --config-fname ${CONFIG_FNAME} \
          --labels ${STEP_02_PROFILE_ARRAYS}/${PRE_LABELS_TO_IDX} \
          --input-profiles-npy ${STEP_03_HOLDOUT_DIR}/${STEP_03_TO_TRAIN} \
          --acc-profiles-csv ${STEP_03_HOLDOUT_DIR}/${STEP_03_TO_TRAIN_CSV} \
          --graph-path ${STEP_01_STREAMLINES}/${OUTPUT_GRAPH} \
          --labels-processed ${output_picked_dir}/${STEP_04_LABELS_PROCESSED} \
          --output-profiles ${output_picked_dir}/${STEP_04_PICK_PROFILES} \
          --output-csv ${output_picked_dir}/${STEP_04_PICK_PROFILES_CSV} \
          --output-label-weights ${output_picked_dir}/${STEP_04_LABELS_WEIGHTS}


  mkdir -p ${reformat_labels_dir}

  python ${CODEBASE_DIR}/step_04_labels_array_extractor.py \
          --labels ${output_picked_dir}/${STEP_04_LABELS_PROCESSED} \
          --profiles-info-csv ${output_picked_dir}/${STEP_04_PICK_PROFILES_CSV} \
          --output-profiles-info-csv ${reformat_labels_dir}/${STEP_04_PICK_PROFILES_CSV}\
          --output-raw-labels ${reformat_labels_dir}/${STEP_04_RAW_LABELS}\
          --output-one-hot ${reformat_labels_dir}/${STEP_04_LABELS_DATASET} \
          --output-labels-processed ${reformat_labels_dir}/${STEP_04_LABELS_PROCESSED}

  python ${CODEBASE_DIR}/step_04_labels_preprocessing_report.py --config-fname ${CONFIG_FNAME} \
          --profiles-info-csv ${output_picked_dir}/${STEP_04_PICK_PROFILES_CSV} \
          --output-report ${STEP_04_OUTPUT_DIR}/${STEP_04_REPORT_FNAME}

fi

#----------------------------------------------------------
# Visualization (histograms)
#----------------------------------------------------------

if [ ${DO_VISUALIZATION} = 'true' ]
then

  output_vis_dir=${OUTPUT_DIR}/${OUTPUT_VISUALIZATION_DIR}
  rm -r -f ${output_vis_dir}
  mkdir -p ${output_vis_dir}

  python ${CODEBASE_DIR}/step_03_visualization.py --config-fname ${CONFIG_FNAME} \
          --labels-to-idx ${PROFILES_OUTPUT}/${OUTPUT_LABELS_TO_IDX} \
          --input-visualization-csv ${output_picked_dir}/${OUTPUT_VISUALIZATION_CSV} \
          --output-visualizastion ${output_vis_dir}

fi

#----------------------------------------------------------
# Split dataset into train and valid
#----------------------------------------------------------

if [ ${DO_SPLIT_DATASETS} = 'true' ]
then
  cd ${WORK_DIR}

  rm -r -f ${output_split_dir}
  mkdir -p ${output_split_dir}

  python ${CODEBASE_DIR}/step_04_split_stratified.py --config-fname ${CONFIG_FNAME} \
          --profiles ${output_picked_dir}/${STEP_04_PICK_PROFILES} \
          --labels ${reformat_labels_dir}/${STEP_04_RAW_LABELS} \
          --profiles-csv ${reformat_labels_dir}/${STEP_04_PICK_PROFILES_CSV} \
          --output-split-profiles-csv ${output_split_dir}/${STEP_04_SPLIT_DATASET}

fi

#----------------------------------------------------------
# Normalization
#----------------------------------------------------------

if [ ${DO_NORMALIZATION} = 'true' ]
then
  cd ${WORK_DIR}

  rm -r -f ${output_norm_dir}
  mkdir -p ${output_norm_dir}

  python ${CODEBASE_DIR}/step_04_norm.py --config-fname ${CONFIG_FNAME} \
          --input-profiles ${output_picked_dir}/${STEP_04_PICK_PROFILES} \
          --output-norm-profiles ${output_norm_dir}/${STEP_04_NORM_PROFILES}
fi
