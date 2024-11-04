source eval_variables.sh
DIR=../data/output/$input_dir # the output dir set in `run_tash.sh`
python eval_scores.py --output_dir1 $DIR --start 0 --end 500
