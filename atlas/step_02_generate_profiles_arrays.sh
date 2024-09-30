#!/bin/bash -xe

DO_GENERATE_LABELS_FILE='true'
DO_GENERATE_PROFILES_ARRAYS='true'
DO_GENERATE_CONFIDENCE_DATASET='true'
DO_VISUALIZATION='true'
DO_HIST='true'

source header.sh

mkdir -p ${STEP_02_PROFILE_ARRAYS}

#=${WORK_DIR}/${PROFILES_ARRAYS_OUTPUT}/${INSIDE_PROFILES_ARRAYS_OUTPUT}
#----------------------------------------------------------
# Generate csv label file
#----------------------------------------------------------

if [ ${DO_GENERATE_LABELS_FILE} = 'true' ]; then

  # Delete existing file
  # shellcheck disable=SC2115
  rm -rf ${STEP_02_PROFILE_ARRAYS}/${PRE_LABELS_TO_IDX}

  while IFS="," read -r _ case_id stack_name st_section end_section; do
    input_stack=${NIFTI_INPUT}/${case_id}/${stack_name}${VOL_SEGMENTATION_SUFFIX}

    python ${CODEBASE_DIR}/step_02_get_labels.py --config-fname ${CONFIG_FNAME}\
    --input-dir ${NIFTI_INPUT}/average_segmentation.nii.gz\
    --output-path ${STEP_02_PROFILE_ARRAYS}/${PRE_LABELS_TO_IDX}
  done < <(tail -n +2 ${STEP_01_STREAMLINES}/${DATA_INFO})
fi

#----------------------------------------------------------
# Generate profiles and labels arrays
#----------------------------------------------------------

if [ ${DO_GENERATE_PROFILES_ARRAYS} = 'true' ]; then

  # shellcheck disable=SC2115
  rm -rfv ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}
  mkdir -p ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}

  while IFS="," read -r _ case_id stack_name st_section end_section; do
    python ${CODEBASE_DIR}/step_02_generate_profiles_array.py --config-fname ${CONFIG_FNAME} \
      --profile-storage ${STEP_01_STREAMLINES} \
      --case ${case_id} \
      --output-profiles-npy ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES} \
      --output-profiles-csv ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES_CSV}
  done < <(tail -n +2 ${STEP_01_STREAMLINES}/${DATA_INFO})
fi


#----------------------------------------------------------
# Generate confidence condition dataset
#----------------------------------------------------------

if [ ${DO_GENERATE_CONFIDENCE_DATASET} = 'true' ]; then
  python ${CODEBASE_DIR}/step_02_confidence_trial.py --config-fname ${CONFIG_FNAME} \
          --profiles-npy ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES} \
          --profiles-csv ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES_CSV} \
          --split-profiles-csv  ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_SPLIT_PROFILES_CSV} \
          --accept-profiles-csv  ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ACC_PROFILES_CSV} \
          --output-accept-profile-npy  ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ACC_PROFILES}

fi

if [ ${DO_VISUALIZATION} = 'true' ]; then
  mkdir -p ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_VISUALIZATION}

  python ${CODEBASE_DIR}/step_02_visualization.py --config-fname ${CONFIG_FNAME} \
          --input-profiles-csv ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_SPLIT_PROFILES_CSV} \
          --output-dir ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_VISUALIZATION}

fi

if [ ${DO_HIST} = 'true' ]; then
  mkdir -p ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_VISUALIZATION}

  python ${CODEBASE_DIR}/step_02_plot_confidence_hist.py --config-fname ${CONFIG_FNAME} \
          --input-profiles-csv ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES_CSV} \
          --input-profiles-npy ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES}\
          --output-dir ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_VISUALIZATION}

fi
