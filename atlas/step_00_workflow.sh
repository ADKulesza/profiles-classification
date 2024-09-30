#!/bin/bash -xe

# Sprawdzenie przed processingiem

source header.sh

STEP_02_DIR=${STEP_02_PROFILE_ARRAYS}/${STEP_02_CONFIDENCE}

clean_output () {
  read -r -p "Are you sure to remove ${OUTPUT_DIR}? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
  then
      rm -r -f ${OUTPUT_DIR}
  else
      exit 1;
  fi
}

clean_profiles () {
  read -r -p "Are you sure to remove ${PROFILES_OUTPUT}? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
  then
      rm -r -f ${PROFILES_OUTPUT}
  else
      exit 1;
  fi

}

generate_labels_graph () {
  #----------------------------------------------------------
  # Generate_label_graph
  #----------------------------------------------------------
  mkdir -p ${STEP_01_STREAMLINES}
  python ${CODEBASE_DIR}/step_00_graph_of_labels.py \
          --seg-path ${NM_TEMPLATE_INPUT}/*${VOL_SEGMENTATION_SUFFIX} \
          --output-path ${STEP_01_STREAMLINES}/${OUTPUT_GRAPH}
}

get_profile_samples () {
  #----------------------------------------------------------
  # Generate_label_graph
  #----------------------------------------------------------

  plot_dir="profiles_plot"
  mkdir -p ${plot_dir}

  cd ${WORK_DIR}

  python3 ${CODEBASE_DIR}/step_00_plot_profile_samples.py --config-fname ${CONFIG_FNAME} \
          --input-profiles-npy ${STEP_02_DIR}/${STEP_02_ALL_PROFILES} \
          --input-labels-npy ${STEP_02_DIR}/${OUTPUT_LABELS_ARRAY} \
          --all-profiles-csv ${STEP_02_DIR}/${STEP_02_ALL_PROFILES_CSV} \
          --output-dir ${plot_dir}\
          --section 225 \
          --area-id 37 \
          --plots-number 100
}

get_profiles_mean_sample () {
  #----------------------------------------------------------
  # Generate_label_graph
  #----------------------------------------------------------

  plot_dir="profiles_mean_plots"
  mkdir -p ${plot_dir}

  cd ${WORK_DIR}

  python3 ${CODEBASE_DIR}/step_00_plot_profiles_mean_sample.py \
          --input-profiles-npy ${STEP_02_DIR}/${STEP_02_ALL_PROFILES} \
          --all-profiles-csv ${STEP_02_DIR}/${STEP_02_ALL_PROFILES_CSV} \
          --output-dir ${plot_dir}\
          --section 523 \
          --start-profile-id 980 \
          --end-profile-id 1000
}


case $1 in
  clean_output) "$@";;
  clean_profiles) "$@";;
  generate_labels_graph) "$@";;
  get_profile_samples) "$@";;
  get_profiles_mean_sample) "$@";;
esac