from pysc2.lib import features

PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
PLAYER_ID = features.SCREEN_FEATURES.player_id.index

PLAYER_SELF = 1

NOT_QUEUED = [0]
QUEUED = [1]

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5
FOOD_REWARD = 0.1

PLAYER_HOSTILE = 4

DATA_FILE = 'agent_data'