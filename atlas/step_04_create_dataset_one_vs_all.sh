#!/bin/bash -xe

DO_CLEAN='false'
DO_CREATE_DATASETS='true'
DO_VISUALIZATION='false'
DO_SPLIT_DATASETS='true'
DO_NORMALIZATION='true'

source header.sh

STEP_02_DIR=${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}


#if [ ${DO_CREATE_DATASETS} = 'true' ]; then
#  [ -d ${STEP_04_ONE_VS_ALL} ] && cp -r ${STEP_04_ONE_VS_ALL} ${STEP_04_ONE_VS_ALL}_$(date +"%Y-%m-%d_%H_%M")_backup
#  rm -rf ${STEP_04_ONE_VS_ALL}
#fi

mkdir -p ${STEP_04_ONE_VS_ALL}

labels_fname=${STEP_02_PROFILE_ARRAYS}/${PRE_LABELS_TO_IDX}

while IFS="," read -r area_id _l _idx; do
  if [[ ${area_id} == 0 ]]; then
    continue
  fi

  label_dir=${STEP_04_ONE_VS_ALL}/${area_id}
  mkdir -p ${label_dir}

  cp ${WORK_DIR}/${ONE_VS_ALL_CONFIG} ${label_dir}/${CONFIG_FNAME}
  sed -i 's/"%%area_id%%"/'${area_id}'/' ${label_dir}/${CONFIG_FNAME}

  # ---
  output_picked_dir=${label_dir}/${STEP_04_PICK_PROFILES_DIR}
  reformat_labels_dir=${label_dir}/${STEP_04_REFORMAT_LABELS_DIR}
  output_split_dir=${label_dir}/${STEP_04_SPLIT_DATASETS_DIR}
  output_norm_dir=${label_dir}/${STEP_04_NORM_DIR}
  # ---

  #----------------------------------------------------------
  # Create datasets
  #----------------------------------------------------------
  if [ ${DO_CREATE_DATASETS} = 'true' ]; then

    mkdir -p ${output_picked_dir}

    python ${CODEBASE_DIR}/step_04_areas_picker.py --config-fname ${label_dir}/${CONFIG_FNAME} \
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
      --output-profiles-info-csv ${reformat_labels_dir}/${STEP_04_PICK_PROFILES_CSV} \
      --output-raw-labels ${reformat_labels_dir}/${STEP_04_RAW_LABELS} \
      --output-one-hot ${reformat_labels_dir}/${STEP_04_LABELS_DATASET} \
      --output-labels-processed ${reformat_labels_dir}/${STEP_04_LABELS_PROCESSED}

    python ${CODEBASE_DIR}/step_04_labels_preprocessing_report.py --config-fname ${label_dir}/${CONFIG_FNAME} \
      --profiles-info-csv ${output_picked_dir}/${STEP_04_PICK_PROFILES_CSV} \
      --output-report ${label_dir}/${STEP_04_REPORT_FNAME}
  fi

  #----------------------------------------------------------
  # Split dataset into train and valid
  #----------------------------------------------------------
  if [ ${DO_SPLIT_DATASETS} = 'true' ]; then
    mkdir -p ${output_split_dir}

    python ${CODEBASE_DIR}/step_04_split_stratified.py --config-fname ${label_dir}/${CONFIG_FNAME} \
          --profiles ${output_picked_dir}/${STEP_04_PICK_PROFILES} \
          --labels ${reformat_labels_dir}/${STEP_04_RAW_LABELS} \
          --profiles-csv ${reformat_labels_dir}/${STEP_04_PICK_PROFILES_CSV} \
          --output-split-profiles-csv ${output_split_dir}/${STEP_04_SPLIT_DATASET}

    python ${CODEBASE_DIR}/step_04_get_label_probabilities.py --config-fname ${label_dir}/${CONFIG_FNAME} \
      --labels ${labels_fname} \
      --graph-path ${STEP_01_STREAMLINES}/${OUTPUT_GRAPH} \
      --split-profiles-csv ${output_split_dir}/${STEP_04_SPLIT_DATASET} \
      --output-prob-split-profiles-csv ${label_dir}/${STEP_04_LABELS_INFO}
  fi


  #----------------------------------------------------------
  # Normalization
  #----------------------------------------------------------
  if [ ${DO_NORMALIZATION} = 'true' ]; then

      mkdir -p ${output_norm_dir}

    python ${CODEBASE_DIR}/step_04_norm.py --config-fname ${label_dir}/${CONFIG_FNAME} \
            --input-profiles ${output_picked_dir}/${STEP_04_PICK_PROFILES} \
            --output-norm-profiles ${output_norm_dir}/${STEP_04_NORM_PROFILES}
  fi
  done \
    < <(tail -n +2 ${labels_fname})




