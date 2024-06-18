
export CUDA_VISIBLE_DEVICES=2

data_name=ace04
archi_name=nner_slg

home_dir=/mnt/LJH/XFM/NERProject
code_dir=$home_dir/span-level/
data_dir=$home_dir/span-level/data/datasets/$data_name
old_data_dir=$home_dir/span-level/data/datasets/$data_name/old_data

model_dir=$home_dir/span-level/__models/$data_name/$archi_name
result_dir=$home_dir/span-level/__results/$data_name/$archi_name

mkdir -p $model_dir
mkdir -p $result_dir

shell_id=41_18

ulimit -n 40000
export LOGLEVEL=INFO

export LOGFILENAME=$result_dir/"$shell_id".log

export LOGFILENAME=$result_dir/"$shell_id".log
echo "hell owrol"