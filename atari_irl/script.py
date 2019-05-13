import argparse
from atari_irl.irl import main

def handle_bool(var):
    var = var.lower()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env_name',  type=str, help='environment name', default='PLECatcher-v0')
parser.add_argument('--expert_total_timesteps', type=float, default=40e6)
parser.add_argument('--imitator_total_timesteps', type=float, default=100e6)
parser.add_argument('--num_trajectories', type=int, default=10)
parser.add_argument('--use_trajectories_file', type=str, default='')
parser.add_argument('--use_expert_file', type=str, default='')
parser.add_argument('--score_discrim', type=bool, default=True)
parser.add_argument('--update_ratio', type=int, default=32)
parser.add_argument('--buffer_size', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--expert_type', type=str, default='PPO')
parser.add_argument('--imitator_policy_type', type=str, default='PPO')
parser.add_argument('--state_only', dest='state_only', action='store_true')
parser.set_defaults(state_only=False)
parser.add_argument('--irl', dest='do_irl', action='store_true')
parser.add_argument('--no-irl', dest='do_irl', action='store_false')
parser.set_defaults(do_irl=True)
parser.add_argument('--num_envs', type=int, default=8)
parser.add_argument('--load_policy_initialization', type=str, default=None)
parser.add_argument('--information_bottleneck_bits', type=float, default=None)
parser.add_argument('--reward_change_penalty', type=float, default=None)

args = parser.parse_args()

print(args)

main(
    env_name=args.env_name,
    expert_total_timesteps=args.expert_total_timesteps,
    imitator_total_timesteps=args.imitator_total_timesteps,
    num_trajectories=args.num_trajectories,
    use_trajectories_file=args.use_trajectories_file,
    use_expert_file=args.use_expert_file,
    score_discrim=args.score_discrim,
    update_ratio=args.update_ratio,
    buffer_size=args.buffer_size,
    seed=args.seed,
    expert_type=args.expert_type,
    imitator_policy_type=args.imitator_policy_type,
    do_irl=args.do_irl,
    state_only=args.state_only,
    num_envs=args.num_envs,
    information_bottleneck_bits=args.information_bottleneck_bits,
    reward_change_penalty=args.reward_change_penalty,
)
