import importlib
from termcolor import colored

from .envs.my_pursuit import my_parallel_env
from .envs.my_pursuit_msg import my_parallel_env_message
from .envs.my_pursuit_grid_loc import my_parallel_env_grid_loc
from .envs.my_pursuit_full import my_parallel_env_full
from .envs.my_pursuit_ic3 import my_parallel_env_ic3
from .envs.my_pursuit_noise import my_parallel_env_noise
from .envs.my_pursuit_toggle import my_parallel_env_toggle

__version__ = importlib.metadata.version(__package__ or __name__)
print(f"Currently using {__package__} {colored(f'v{__version__}', 'blue')}")
print(colored(f"Successfully import all pursuit envs", 'green'))

confirm_msg = "Are you sure that you already called 'pip install' after u changed the env? (Y/n) "
reply = input(colored(confirm_msg, 'yellow')) or 'y'
while reply.lower() != 'y':
    if reply.lower() == 'n':
        exit(0)
    reply = input(colored(confirm_msg, 'yellow'))

print("-" * 20)
