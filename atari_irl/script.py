import argparse
from atari_irl.irl import main

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env_name',  type=str, help='environment name', default='PLECatcher-v0')
parser.add_argument('--total_timesteps_expert', type=float, default=40e6)
parser.add_argument('--num_trajectories', type=int, default=10)
parser.add_argument('--use_trajectories_file', type=str, default='')
parser.add_argument('--use_expert_file', type=str, default='')
parser.add_argument('--score_discrim', type=bool, default=True)
parser.add_argument('--update_ratio', type=int, default=32)
parser.add_argument('--buffer_size', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--do_irl', type=bool, default=True)

args = parser.parse_args()

print(args)

main(
    env_name=args.env_name,
    total_timesteps=args.total_timesteps_expert,
    num_trajectories=args.num_trajectories,
    use_trajectories_file=args.use_trajectories_file,
    use_expert_file=args.use_expert_file,
    score_discrim=args.score_discrim,
    update_ratio=args.update_ratio,
    buffer_size=args.buffer_size,
    seed=args.seed,
    do_irl=args.do_irl
)
