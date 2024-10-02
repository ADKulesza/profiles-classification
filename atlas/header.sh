#!/bin/bash
 set -xe

CODEBASE_DIR="/f/Studia/Magisterka/profiles-pipeline/marmoset_profiles_codebase"
WORK_DIR="/f/Studia/Magisterka/profiles-pipeline/atlas"

NM_TEMPLATE_INPUT="/f/Studia/Magisterka/profiles-pipeline/NM_"

TEMPLATE_JSON=${WORK_DIR}/template.json
STEP_01_STREAMLINES="step_01_streamlines"
NIFTI_INPUT="/f/Studia/Magisterka/profiles-pipeline/NM_"

STREAMLINES_SIF="/home/akulesza/repos/LaplacianCorticalThickness/LaplacianCorticalThickness.sif"
STREAMLINES_SCRIPT="/opt/LaplacianCorticalThickness/pythickener/thickness.py"

SECTION_SPACING=1

CONFIG_FNAME='dataset_settings.json'
LABEL_NAMES='areas_definitions.csv'
AREAS_ORDER='areas_order.json'
OUTPUT_GRAPH='labels_graph.json'

VOL_IMG_SUFFIX=".nii.gz"
VOL_SEGMENTATION_SUFFIX="_segmentation.nii.gz"
VOL_CORTICAL_THICKNESS_SUFFIX="_cortical_thickness_mask.nii.gz"

DATA_INFO="data_info.csv"

STEP_02_PROFILE_ARRAYS="step_02_profile_arrays"
STEP_02_CONFIDENCE="067_profiles_arrays"
PRE_LABELS_TO_IDX="labels.csv"
STEP_02_ALL_PROFILES='all_profiles_array.npy'
STEP_02_ALL_PROFILES_CSV='all_profiles.csv'
STEP_02_SPLIT_PROFILES_CSV='confidence_split_profiles.csv'
STEP_02_ACC_PROFILES_CSV='accepted_profiles.csv'
STEP_02_ACC_PROFILES='accepted_profiles_array.npy'
STEP_02_VISUALIZATION='visualization'

STEP_03_HOLDOUT_DIR='step_03_holdout_split'
STEP_03_TO_TRAIN='split_datasets.npy'
STEP_03_TO_TRAIN_CSV='split_datasets.csv'

STEP_03_SECTIONS_HO_PROFILES='sections_holdout_profiles.npy'
STEP_03_SECTIONS_HO_PROFILES_CSV='sections_holdout_profiles.csv'

# ALL VS ALL
STEP_04_OUTPUT_DIR="step_04_processed_profiles_all_vs_all"

STEP_04_PICK_PROFILES_DIR='output_01_areas_pick'
STEP_04_LABELS_PROCESSED='labels_processed.csv'
STEP_04_PICK_PROFILES='profiles.npy'
STEP_04_PICK_PROFILES_CSV='profiles_info.csv'
STEP_04_LABELS_WEIGHTS='label_weights.csv'

STEP_04_REFORMAT_LABELS_DIR='output_02_reformat_labels'
STEP_04_RAW_LABELS='raw_segmentation.npy'
STEP_04_LABELS_DATASET='one_hot_segmentation.npy'
STEP_04_REPORT_FNAME='report.json'

STEP_04_VISUALIZATION='output_03_visualization'

STEP_04_SPLIT_DATASETS_DIR='output_04_split_datasets'
STEP_04_SPLIT_DATASET="split_datasets_processed.csv"

STEP_04_NORM_DIR='output_05_norm_profiles'
STEP_04_NORM_PROFILES='norm_profiles.npy'

# ONE VS ALL
STEP_04_ONE_VS_ALL='step_04_processed_profiles_one_vs_all'
ONE_VS_ALL_CONFIG='template_dataset_settings.json'
STEP_04_LABELS_INFO='profile_labels_info_processed.csv'

MODEL_NAME='multi_branch_model'
STEP_05_MODELS='step_05_trained_models'
STEP_05_MODEL_INFO_CSV='model_info.csv'

STEP_05_MODELS_ONE_VS_ALL='step_05_trained_models_one_vs_all'


STEP_06_EVALUATION_ALL_VS_ALL='step_06_evaluation_all_vs_all'
STEP_06_MODELS_ORDER='model_order.json'
STEP_06_BINARY_CMAT_DIR='binary_cmat'
STEP_06_METRICS_DIR='metrics'

STEP_06_NORM_PROFILES="x_norm.npy"

STEP_06_SECTION_EVALUATION="sections"
STEP_06_SECTION_CSV="section_holdout.csv"

STEP_06_EVALUATION_ONE_VS_ALL='step_06_evaluation_one_vs_all'
STEP_06_SUMMARY_ONE_VS_ALL='summary_one_vs_all'

STEP_08_METRICS_VS_DIR='metrics_vs'


#
#
#OUTPUT_ONE_VS_ALL='step_5_one_vs_all'

#STEP_05_HOLDOUT_CSV='holdout_datasets.csv'
#STEP_05_PROFILES_INFO='profiles_info_processed.csv'
#
#STEP_06_EVALUATION_DIR='step_06_evaluation'

#

#STEP_05_NORM_PROFILES='norm_profiles.npy'

#STEP_05_ACC_PROFILES_CSV='accepted_profiles.csv'
#
#STEP_05_LABELS_PROCESSED='labels_processed.csv'
##STEP_05_SPLIT_PROB_PROFILES_CSV="split_datasets_prob_processed.csv"
#
#STEP_05_OUTPUT_DIR="step_03_processed_profiles"

#
#STEP_05_REFORMAT_LABELS_DIR='output_02_reformat_labels'
#STEP_05_RAW_LABELS='raw_segmentation.npy'
#STEP_05_LABELS_DATASET='one_hot_segmentation.npy'
#STEP_05_REPORT_FNAME='report.json'
#
#STEP_05_VISUALIZATION='output_03_visualization'
#
##STEP_05_SPLIT_DATASETS_DIR='output_04_split_datasets'
##STEP_05_SPLIT_DATASET="split_datasets_processed.csv"
#
#STEP_05_NORM_DIR='output_05_norm_profiles'
#STEP_05_NORM_PROFILES='norm_profiles.npy'
#
#STEP_05_MODELS_DIR='binary_trained_models'
#
#OUTPUT_ONE_VS_ALL_PREDICTIONS='predictions'


#OUTPUT_MODELS_DIR='output_06_trained_models'
#OUTPUT_EVALUATION='output_07_evaluation'
#
#
#OUTPUT_PROFILES_ARRAY='all_profiles_array.npy'
#OUTPUT_LABELS_ARRAY='all_labels_array.npy'
#OUTPUT_PROFILES_ARRAY_CSV='all_profiles.csv'
#
#OUTPUT_ACC_PROFILES_ARRAY_CSV='accepted_profiles.csv'
#OUTPUT_SPLIT_PROFILES_ARRAY_CSV='confidence_split_profiles.csv'
#OUTPUT_ACC_PROFILES_ARRAY='accepted_profiles_array.npy'
#OUTPUT_ACC_LABELS_ARRAY='accepted_labels_array.npy'
#
#OUTPUT_ONE_VS_ALL_VALID_PROFILES_CSV='valid_profiles.csv'
#OUTPUT_ONE_VS_ALL_VALID_PROFILE_ARRAY='valid_profiles.npy'
#OUTPUT_ONE_VS_ALL_VALID_LABELS_ARRAY='valid_segmentation.npy'
#

#
#OUTPUT_LABELS_IDX="labels_processed.csv"
#OUTPUT_PROFILES_DATASET='profiles.npy'
#OUTPUT_RAW_LABELS='raw_segmentation.npy'
#OUTPUT_LABELS_DATASET='one_hot_segmentation.npy'
#OUTPUT_DATASET_PROCESSED='profiles_info.csv'
##OUTPUT_REJECT_PROFILES_DATASET='rejected_profiles_dataset.csv'
#OUTPUT_VISUALIZATION_CSV='visualization_data.csv'
#OUTPUT_REPORT_FNAME='report.json'
#OUTPUT_LABEL_WEIGHTS='label_weights.csv'
#
#OUTPUT_SPLIT_DATASET="split_datasets_processed.csv"
#OUTPUT_HOLD_OUT_DATASET="hold_out_split_profiles.csv"
#
#
##OUTPUT_TRAIN_NORM_PROFILES='train_norm.npy'
#OUTPUT_NORM_PROFILES='norm_profiles.npy'
#OUTPUT_LABELS="labels.npy"
##OUTPUT_TEST_NORM_PROFILES='test_norm.npy'
##OUTPUT_VALID_NORM_PROFILES='valid_norm.npy'
#
##OUTPUT_HOLD_OUT_NORM_PROFILES='hold_out_norm.npy'
##OUTPUT_HOLD_OUT_LABELS='labels_hold_out.npy'
#
#OUTPUT_STREAMLINES_DIR='streamlines'
#
#
#OUTPUT_EVALUATION_NO_DATE='01_evaluation'
#OUTPUT_PICK_PROFILES_TO_EVALUATE='__profiles_to_evaluate.csv'
#OUTPUT_NORM_PROFILES_TO_EVALUATE='norm_profiles.npy'
#OUTPUT_RAW_LABELS_TO_EVALUATE='raw_segmentation.npy'
#OUTPUT_LABEL_PREDICTION='label_prediction.npy'
#OUTPUT_FINAL_PREDICTION='final_prediction.npy'
#OUTPUT_CONFMAT='confusion_matrix.npy'
#OUTPUT_CONFMAT_PLOT_NAME='confmat'
#
#OUTPUT_EVALUATION_HOLD_OUT='hold_out'
#OUTPUT_EVALUATION_HOLD_OUT_CSV='profile_validation.csv'
#OUTPUT_HEATMAPS='heatmaps.npy'






