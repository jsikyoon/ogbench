# SESSION_NAME=2
# tmux new-session -d -s $SESSION_NAME
# tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
# tmux send-keys -t $SESSION_NAME "
# export CUDA_VISIBLE_DEVICES=2
# python run.py --kld_weight 1e-6
# python run.py --kld_weight 1e-8
# python run.py --kld_weight 1e-10
# python run.py --kld_weight 0
# " C-m
# sleep 60

# SESSION_NAME=3
# tmux new-session -d -s $SESSION_NAME
# tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
# tmux send-keys -t $SESSION_NAME "
# export CUDA_VISIBLE_DEVICES=3
# python run.py --kld_weight 1e-6 --lr=1e-3
# python run.py --kld_weight 1e-8 --lr=1e-3
# " C-m

SESSION_NAME=4
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
tmux send-keys -t $SESSION_NAME "
export CUDA_VISIBLE_DEVICES=4
python run.py --kld_weight 1e-8 --lr=5e-4
" C-m

# SESSION_NAME=5
# tmux new-session -d -s $SESSION_NAME
# tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
# tmux send-keys -t $SESSION_NAME "
# export CUDA_VISIBLE_DEVICES=5
# python run.py --kld_weight 1e-6 --lr=5e-3
# " C-m

