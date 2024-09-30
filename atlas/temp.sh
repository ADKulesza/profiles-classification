#path="/home/akulesza/Public/2023-03-07-profiles-piepeline/atlas/_output_one_vs_all"
#
#for f in ${path}/label_*/
#do
#  mPath=${f}"output_06_trained_models/flatten_conv_model_kfold_0/"
#  newPath="${mPath/_output_one_vs_all/output_one_vs_all}"
#  rm -rfv $newPath
#  newPath="${newPath/flatten_conv_model_kfold_0/flatten_conv_model_kfold_0_0}"
#  cp -R $mPath $newPath
#done