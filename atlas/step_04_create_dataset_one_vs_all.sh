#!/bin/bash -xe

DO_CLEAN='false'
DO_CREATE_DIRS='true'
DO_CREATE_DATASETS='true'
DO_VISUALIZATION='false'
DO_SPLIT_DATASETS='true'
DO_NORMALIZATION='true'

source header.sh


#if [ ${DO_CREATE_DATASETS} = 'true' ]; then
#  [ -d ${STEP_04_ONE_VS_ALL} ] && cp -r ${STEP_04_ONE_VS_ALL} ${STEP_04_ONE_VS_ALL}_$(date +"%Y-%m-%d_%H_%M")_backup
#  rm -rf ${STEP_04_ONE_VS_ALL}
#fi

mkdir -p ${STEP_04_ONE_VS_ALL}

  #----------------------------------------------------------
  # Create datasets
  #----------------------------------------------------------
  if [ ${DO_CREATE_DATASETS} = 'true' ]; then

    python ${CODEBASE_DIR}/step_04_01_areas_picker.py --config-fname ${CONFIG_FNAME} \
      --areas-def ${LABEL_NAMES} \
      --input-profiles-npy ${STEP_03_HOLDOUT_DIR}/${STEP_03_TO_TRAIN} \
      --profiles-csv ${STEP_03_HOLDOUT_DIR}/${STEP_03_TO_TRAIN_CSV} \
      --output-profiles ${STEP_04_ONE_VS_ALL}/${STEP_04_PICK_PROFILES} \
      --output-csv ${STEP_04_ONE_VS_ALL}/${STEP_04_PICK_PROFILES_CSV}

  fi

labels_fname=${STEP_02_PROFILE_ARRAYS}/${PRE_LABELS_TO_IDX}

while IFS="," read -r area_id _l _idx; do
  if [[ ${area_id} == 0 ]]; then
    continue
  fi

  label_dir=${STEP_04_ONE_VS_ALL}/${area_id}
  if [ ${DO_CREATE_DIRS} = 'true' ]; then
    mkdir -p ${label_dir}

    cp ${WORK_DIR}/${ONE_VS_ALL_CONFIG} ${label_dir}/${CONFIG_FNAME}
    sed -i 's/"%%area_id%%"/'${area_id}'/' ${label_dir}/${CONFIG_FNAME}
  fi

  #----------------------------------------------------------
  # Create datasets
  #----------------------------------------------------------
  if [ ${DO_CREATE_DATASETS} = 'true' ]; then

    python ${CODEBASE_DIR}/step_04_01_areas_picker.py --config-fname ${label_dir}/${CONFIG_FNAME} \
      --areas-def ${LABEL_NAMES} \
      --input-profiles-npy ${STEP_04_ONE_VS_ALL}/${STEP_04_PICK_PROFILES} \
      --profiles-csv ${STEP_04_ONE_VS_ALL}/${STEP_04_PICK_PROFILES_CSV} \
      --output-profiles ${label_dir}/${STEP_04_PICK_PROFILES} \
      --output-csv ${label_dir}/${STEP_04_PICK_PROFILES_CSV} \
      --one-vs-all


    python ${CODEBASE_DIR}/step_04_labels_array_extractor.py \
      --profiles-info-csv ${label_dir}/${STEP_04_PICK_PROFILES_CSV} \
      --output-raw-labels ${label_dir}/${STEP_04_RAW_LABELS} \
      --output-one-hot ${label_dir}/${STEP_04_LABELS_DATASET}

      rm ${label_dir}/${STEP_04_PICK_PROFILES_CSV}
      [ -e ${label_dir}/${STEP_04_PICK_PROFILES} ] && rm ${label_dir}/${STEP_04_PICK_PROFILES}



  fi


  #----------------------------------------------------------
  # Split dataset into train and valid
  #----------------------------------------------------------
  if [ ${DO_SPLIT_DATASETS} = 'true' ]; then

    python ${CODEBASE_DIR}/step_04_02_split_stratified.py --config-fname ${label_dir}/${CONFIG_FNAME} \
          --profiles ${STEP_04_ONE_VS_ALL}/${STEP_04_PICK_PROFILES} \
          --labels ${label_dir}/${STEP_04_RAW_LABELS} \
          --profiles-csv ${STEP_04_ONE_VS_ALL}/${STEP_04_PICK_PROFILES_CSV} \
          --output-split-profiles-csv ${label_dir}/${STEP_04_SPLIT_DATASET}

    rm ${label_dir}/${STEP_04_RAW_LABELS}

  fi

done \
    < <(tail -n +2 ${labels_fname})


  #----------------------------------------------------------
  # Normalization
  #----------------------------------------------------------
  if [ ${DO_NORMALIZATION} = 'true' ]; then

    python ${CODEBASE_DIR}/step_04_norm.py --config-fname ${CONFIG_FNAME} \
          --input-profiles ${STEP_04_OUTPUT_DIR}/${STEP_04_PICK_PROFILES} \
          --output-norm-profiles ${STEP_04_ONE_VS_ALL}/${STEP_04_NORM_PROFILES}
  fi
