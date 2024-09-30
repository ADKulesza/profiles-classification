#!/bin/bash -xe

DO_GET_DATA_INFO='true'
DO_PREPROCESS_SLICES='false'
DO_GENERATE_STREAMLINES='false'
DO_GENERATE_CONFIDENCE='false'
DO_EXTRACT_PROFILES='false'

#----------------------------------------------------------
# Load data and create the directories
#----------------------------------------------------------
source header.sh
mkdir -p ${STEP_01_STREAMLINES}

cp ${TEMPLATE_JSON} ${STEP_01_STREAMLINES}/"template.json"

if [ ${DO_GET_DATA_INFO} = 'true' ]; then
  python3 ${CODEBASE_DIR}/step_01_collect_data_info.py \
    --input-dir ${NIFTI_INPUT} \
    --ctx-suffix ${VOL_CORTICAL_THICKNESS_SUFFIX} \
    --output-csv ${STEP_01_STREAMLINES}/${DATA_INFO}

fi

#----------------------------------------------------------
# Loop through the cases
#----------------------------------------------------------

while IFS="," read -r _ case_id stack_name st_section end_section; do
  # Create a directory for a specific case

  mkdir -p ${STEP_01_STREAMLINES}/${case_id}

  dir_input_stacks=${NIFTI_INPUT}/${case_id}

  vol_images=${stack_name}${VOL_IMG_SUFFIX}
  vol_segmentation=${stack_name}${VOL_SEGMENTATION_SUFFIX}
  vol_cortical_thickness=${stack_name}${VOL_CORTICAL_THICKNESS_SUFFIX}

  cd ${STEP_01_STREAMLINES}/${case_id}

  #----------------------------------------------------------
  # Preprocess slices
  #----------------------------------------------------------
  if [ ${DO_PREPROCESS_SLICES} = 'true' ]; then

    for i in $(seq -w "${st_section}" $SECTION_SPACING "${end_section}"); do

      echo ${i}
      rm -r -f ${i}
      mkdir ${i}
      cp ${WORK_DIR}/${STEP_01_STREAMLINES}/"template.json" ${i}/${i}.json

      cd ./${i}
      pwd
      sed -i 's/%%input_prefix%%/'${i}'/' ${i}.json
      sed -i 's/%%output_prefix%%/'${i}'/' ${i}.json

      c3d ${dir_input_stacks}/${vol_images} -slice y $i -o ./${i}_image.vtk
      c3d ${dir_input_stacks}/${vol_segmentation} -slice y $i -type uchar -o ./${i}_segmentation.vtk
      c3d ${dir_input_stacks}/${vol_cortical_thickness} -slice y $i -replace 2 1 3 2 -type uchar -o ./${i}_borders.vtk
      c2d ./${i}_borders.vtk -replace 2 1 -type uchar -o ./${i}_mask.vtk
      c3d ${dir_input_stacks}/${vol_segmentation} -slice y ${i} -replace 2 0 3 0 -type uchar -o ./${i}_cortex_mask.vtk

      cd ../
    done
  fi

  #----------------------------------------------------------
  # Generate streamlines
  #----------------------------------------------------------
  if [ ${DO_GENERATE_STREAMLINES} = 'true' ]; then
    for dir in [0-9]*/; do
      file_dir=${dir%*/}

      cd $file_dir
      file_num=${file_dir##*/}

      #----------------------------------------------------------
      # Remove old files
      #----------------------------------------------------------
      rm -rfv ${file_num}_divergence.vtk ${file_num}_inner_contour.vtk ${file_num}_laplacian.vtk \
        ${file_num}_laplacian_grad.vtk ${file_num}_midthickness.vtk ${file_num}_outer_contour.vtk \
        ${file_num}_resliced.vtk ${file_num}_streamlines.vtk \
        ${file_num}_stream_seeds.vtk ${file_num}_streamlines_confidence.vtk
      #----------------------------------------------------------

      fname=${file_num}.json
      singularity exec ${STREAMLINES_SIF} python /opt/LaplacianCorticalThickness/pythickener/thickness.py ${WORK_DIR}/${STEP_01_STREAMLINES}/${case_id}/${file_dir}/${fname}
      sed -i.bak s/inf/0/g ${file_num}_streamlines.vtk
      cd ..

    done
  fi

  #----------------------------------------------------------
  # Generate confidence
  #----------------------------------------------------------
  if [ ${DO_GENERATE_CONFIDENCE} = 'true' ]; then
    cd ${WORK_DIR}
    for dir in ${STEP_01_STREAMLINES}/${case_id}/[0-9]*/; do

      file_dir=${dir%*/}
      file_num=${file_dir##*/}
      fname=${dir}/${file_num}_streamlines

      python3 ${CODEBASE_DIR}/step_01_add_segmentation_confidence.py --streamlines-input ${fname}.vtk --output ${fname}_confidence.vtk
    done

    cd ../
  fi

  #----------------------------------------------------------
  # Extract profiles
  #----------------------------------------------------------
  if [ ${DO_EXTRACT_PROFILES} = 'true' ]; then
    cd ${WORK_DIR}
    for dir in ${STEP_01_STREAMLINES}/${case_id}/[0-9]*/; do

      file_dir=${dir%*/}
      file_num=${file_dir##*/}
      fname=${dir}/${file_num}_streamlines_confidence.vtk



      python3 ${CODEBASE_DIR}/step_01_extract_profiles.py --confidence-input ${fname} --config-fname ${CONFIG_FNAME} --output-dir ${dir}
    done
  fi

  # End of case loop
done < <(tail -n +2 ${STEP_01_STREAMLINES}/${DATA_INFO})
