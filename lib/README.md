Customizing our game environment
===
## Edit game field
- file to edit: `two_d_maps.py`
  - location: `/path/to/site-packages/pettingzoo/sisl/pursuit/utils/two_d_maps.py`

### Changes to be made
- to remove obstacles (the building in the middle of the map)
  - comment all lines in `retangle_map()` except `rmap = np.zero(...)` in `two_d_maps.py`
    

## Edit game rule
- file to edit: `pursuit_base.py`
  - location: `/path/to/site-packages/pettingzoo/sisl/pursuit/pursuit_base.py`

### How to use
- repace pursuit_base.py in library with pursuit_base_new.py
  
- pursuit_base_original.py is the original file in the library

### Changed
I change the following behaviors: (see line with ###)
- do not remove prey after it is caught
- originally purs_sur is a bool array that indicate which predator captured prey, now it is a int array that shows how much reward it should get
(I only changed stuff assuming surround=False)

### Code explanation
- petting zoo assume env is ACE (agent environment cycle) game, hence step() is called for 1 agent at atime. Game is changed to parallel with wrapper
- in step(), 1 agent moved, then is_last check if that agent is the last among all agents to move, then prey will be removed and move
- reward() calculate the tagging reward for each predator. An predator get tagging reward if there is a prey in its up/down/left/right
- remove_agents() calculate which prey to be removed, and return array purs_sur which indicate which predator removed an agent

