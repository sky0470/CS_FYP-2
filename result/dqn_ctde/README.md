# Load Policy
## Usage
- `tensorboard --logdir .` to open tensorboard
- `python load_and_test.py --path XXX.pth`

## Note
- seed not working currently, need change of library
- load_and_test.py requires the shape of loaded policy be same as that in the code. it should be serialized and loaded in the future
- find training parameter in *.para

