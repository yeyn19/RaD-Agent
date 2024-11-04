DATA_DIR="./data"
current_time=$(pwd +"%Y-%m-%d_%H:%M:%S")
process_num=1

input_dir=$current_time
method="ETS_annealing_sqrt_woInitElo_s6_f1_t173.72_p0.5_c4_m1_rn3_rg4" # BFS_w2_e2, CoT@1, CoT@3, DFS_woFilter_w2, DFS_w2

DATA_DIR_FINAL="$DATA_DIR/output/$input_dir"
echo $DATA_DIR_FINAL
mkdir $DATA_DIR_FINAL

python answer_generation.py --process_num $process_num --output_answer_file $DATA_DIR_FINAL --method $method