#!/bin/bash -xe

DO_SETS='true'

source header.sh

STEP_02_DIR=${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}

#----------------------------------------------------------
# Create datasets
#----------------------------------------------------------

if [ ${DO_SETS} = 'true' ]; then

#  [ -d ${STEP_03_HOLDOUT_DIR} ] && cp -r ${STEP_03_HOLDOUT_DIR} ${STEP_03_HOLDOUT_DIR}_`date +"%Y-%m-%d_%H_%M"`_backup

  rm -rf "${STEP_03_HOLDOUT_DIR}"
  mkdir -p "${STEP_03_HOLDOUT_DIR}"

  python ${CODEBASE_DIR}/step_03_create_holdout.py --config-fname "${CONFIG_FNAME}" \
  --all-profiles "${STEP_02_DIR}/${STEP_02_ALL_PROFILES}" \
  --all-profiles-csv "${STEP_02_DIR}/${STEP_02_ALL_PROFILES_CSV}" \
  --output-to-train-profiles "${STEP_03_HOLDOUT_DIR}/${STEP_03_TO_TRAIN}" \
  --output-to-train-profiles-csv "${STEP_03_HOLDOUT_DIR}/${STEP_03_TO_TRAIN_CSV}" \
  --output-holdout-sections-profiles "${STEP_03_HOLDOUT_DIR}/${STEP_03_SECTIONS_HO_PROFILES}" \
  --output-holdout-sections-profiles-csv "${STEP_03_HOLDOUT_DIR}/${STEP_03_SECTIONS_HO_PROFILES_CSV}"
fi
