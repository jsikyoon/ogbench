SESSION_NAME=2
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
tmux send-keys -t $SESSION_NAME "
export CUDA_VISIBLE_DEVICES=2
python run.py --kld_weight 1e-10 
" C-m

SESSION_NAME=3
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
tmux send-keys -t $SESSION_NAME "
export CUDA_VISIBLE_DEVICES=3
python run.py --kld_weight 1e-10 --lr=1e-3
" C-m