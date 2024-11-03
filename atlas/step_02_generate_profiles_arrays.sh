#!/bin/bash -xe


DO_GENERATE_PROFILES_ARRAYS='false'
DO_VISUALIZATION='false'
DO_HIST='false'

source header.sh

mkdir -p ${STEP_02_PROFILE_ARRAYS}


#----------------------------------------------------------
# Generate profiles and labels arrays
#----------------------------------------------------------

if [ ${DO_GENERATE_PROFILES_ARRAYS} = 'true' ]; then

  # shellcheck disable=SC2115
  rm -rfv ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}
  mkdir -p ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}

  while IFS="," read -r _ case_id stack_name st_section end_section; do
    python ${CODEBASE_DIR}/step_02_01_generate_profiles_array.py \
      --profile-storage ${STEP_01_STREAMLINES} \
      --case ${case_id} \
      --output-profiles-npy ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES} \
      --output-profiles-csv ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES_CSV}
  done < <(tail -n +2 ${STEP_01_STREAMLINES}/${DATA_INFO})

  python ${CODEBASE_DIR}/step_02_02_confidence_trial.py --config-fname ${CONFIG_FNAME} \
          --profiles-csv ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES_CSV}

fi


if [ ${DO_VISUALIZATION} = 'true' ]; then

  rm -rfv ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_VISUALIZATION}
  mkdir -p ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_VISUALIZATION}

  python ${CODEBASE_DIR}/step_02_visualization.py --config-fname ${CONFIG_FNAME} \
          --input-profiles-csv ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES_CSV} \
          --output-dir ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_VISUALIZATION}

fi

if [ ${DO_HIST} = 'true' ]; then
  mkdir -p ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_VISUALIZATION}

  python ${CODEBASE_DIR}/step_02_plot_confidence_hist.py --config-fname ${CONFIG_FNAME} \
          --input-profiles-csv ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES_CSV} \
          --input-profiles-npy ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_ALL_PROFILES}\
          --output-dir ${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}/${STEP_02_VISUALIZATION}

fi
