#!/bin/bash -xe

DO_CREATE_DATASETS='true'
DO_VISUALIZATION='false'
DO_SPLIT_DATASETS='true'
DO_NORMALIZATION='true'

source header.sh

mkdir -p ${STEP_04_OUTPUT_DIR}

#----------------------------------------------------------
# Create datasets
#----------------------------------------------------------

if [ ${DO_CREATE_DATASETS} = 'true' ]
then
  python ${CODEBASE_DIR}/step_04_01_areas_picker.py --config-fname ${CONFIG_FNAME} \
          --areas-def ${LABEL_NAMES} \
          --input-profiles-npy ${STEP_03_HOLDOUT_DIR}/${STEP_03_TO_TRAIN} \
          --profiles-csv ${STEP_03_HOLDOUT_DIR}/${STEP_03_TO_TRAIN_CSV} \
          --output-profiles ${STEP_04_OUTPUT_DIR}/${STEP_04_PICK_PROFILES} \
          --output-csv ${STEP_04_OUTPUT_DIR}/${STEP_04_PICK_PROFILES_CSV}

  python ${CODEBASE_DIR}/step_04_labels_array_extractor.py \
          --profiles-info-csv ${STEP_04_OUTPUT_DIR}/${STEP_04_PICK_PROFILES_CSV} \
          --output-raw-labels ${STEP_04_OUTPUT_DIR}/${STEP_04_RAW_LABELS}\
          --output-one-hot ${STEP_04_OUTPUT_DIR}/${STEP_04_LABELS_DATASET}
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
          --input-visualization-csv ${STEP_04_OUTPUT_DIR}/${OUTPUT_VISUALIZATION_CSV} \
          --output-visualizastion ${output_vis_dir}

fi

#----------------------------------------------------------
# Split dataset into train and valid
#----------------------------------------------------------
if [ ${DO_SPLIT_DATASETS} = 'true' ]
then

  python ${CODEBASE_DIR}/step_04_02_split_stratified.py --config-fname ${CONFIG_FNAME} \
          --profiles ${STEP_04_OUTPUT_DIR}/${STEP_04_PICK_PROFILES} \
          --labels ${STEP_04_OUTPUT_DIR}/${STEP_04_RAW_LABELS} \
          --profiles-csv ${STEP_04_OUTPUT_DIR}/${STEP_04_PICK_PROFILES_CSV} \
          --output-split-profiles-csv ${STEP_04_OUTPUT_DIR}/${STEP_04_SPLIT_DATASET}
fi

#----------------------------------------------------------
# Normalization
#----------------------------------------------------------
if [ ${DO_NORMALIZATION} = 'true' ]
then
  python ${CODEBASE_DIR}/step_04_norm.py --config-fname ${CONFIG_FNAME} \
          --input-profiles ${STEP_04_OUTPUT_DIR}/${STEP_04_PICK_PROFILES} \
          --output-norm-profiles ${STEP_04_OUTPUT_DIR}/${STEP_04_NORM_PROFILES}
fi
