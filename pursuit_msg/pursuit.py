from termcolor import colored

from .envs.my_pursuit import my_parallel_env
from .envs.my_pursuit_msg import my_parallel_env_message
from .envs.my_pursuit_grid_loc import my_parallel_env_grid_loc
from .envs.my_pursuit_full import my_parallel_env_full


print(colored(f"Successfully import all pursuit envs", 'green'))

confirm_msg = "Are you sure that you already called 'pip install' after u changed the env? (y/n) "
reply = input(colored(confirm_msg, 'yellow'))
while reply.lower() != 'y':
    if reply.lower() == 'n':
        exit(0)
    reply = input(colored(confirm_msg, 'yellow'))

print("-" * 20)
