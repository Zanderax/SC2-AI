from pysc2.lib import actions

import action_constants as act
import ids.unit_typeid as uid

from constants import *
import random

def isBaseTopLeft( obs ):
    player_y, player_x = (obs.observation["feature_minimap"][PLAYER_RELATIVE] == PLAYER_SELF).nonzero()
    return player_y.mean() <= 31

def transformLocation(x, y, base_top_left):
    if not base_top_left:
        return [64 - x, 64 - y]
    return [x, y]

def transformDistance(x, x_distance, y, y_distance, base_top_left):
    if not base_top_left:
        return [x - x_distance, y - y_distance]
    return [x + x_distance, y + y_distance]

def select_building( obs, building_id ):
    unit_type = obs.observation['feature_screen'][UNIT_TYPE]
    unit_y, unit_x = (unit_type == building_id).nonzero()
    if unit_y.any():
        # Grab a random co-ordinate to select only one barracks
        i = random.randint(0, len(unit_y) - 1)
        target = [unit_x[i], unit_y[i]]
        return actions.FunctionCall(act.SELECT_POINT, [NOT_QUEUED, target])

def attack( obs, x, y, base_top_left ):
    if obs.observation['single_select'][0][0] != uid.SCV:
        x = x * 16 + 8
        y = y * 16 + 8
        return action_if_avalible( obs, act.ATTACK_MINIMAP, [NOT_QUEUED, transformLocation(int(x), int(y), base_top_left)])


def action_if_avalible( obs, action, queue ):
    if action in obs.observation['available_actions']:
        return actions.FunctionCall(action, queue)


def DO_NOTHING( obs, base_top_left ):
        return actions.FunctionCall(act.NO_OP, [])   

def SELECT_SCV( obs, base_top_left ):
    unit_type = obs.observation['feature_screen'][UNIT_TYPE]
    unit_y, unit_x = (unit_type == uid.SCV).nonzero()
        
    if unit_y.any():
        i = random.randint(0, len(unit_y) - 1)
        target = [unit_x[i], unit_y[i]]
        
        return actions.FunctionCall(act.SELECT_POINT, [NOT_QUEUED, target])
        
def BUILD_SUPPLY_DEPOT( obs, base_top_left ):
    if act.BUILD_SUPPLYDEPOT_SCREEN in obs.observation['available_actions']:
        unit_type = obs.observation['feature_screen'][UNIT_TYPE]
        unit_y, unit_x = (unit_type == uid.COMMANDCENTER).nonzero()
        
        if unit_y.any():
            target = transformDistance(int(unit_x.mean()), random.randint(0,10), int(unit_y.mean()), random.randint(20,30), base_top_left)
        
            return actions.FunctionCall(act.BUILD_SUPPLYDEPOT_SCREEN, [NOT_QUEUED, target])
        
def BUILD_BARRACKS( obs, base_top_left ):
    if act.BUILD_BARRACKS_SCREEN in obs.observation['available_actions']:
        unit_type = obs.observation['feature_screen'][UNIT_TYPE]
        unit_y, unit_x = (unit_type == uid.COMMANDCENTER).nonzero()
        
        if unit_y.any():
            target = transformDistance(int(unit_x.mean()), random.randint(20,30), int(unit_y.mean()), random.randint(0,10), base_top_left)

            return actions.FunctionCall(act.BUILD_BARRACKS_SCREEN, [NOT_QUEUED, target])
    
def SELECT_BARRACKS( obs, base_top_left ):
    return select_building( obs, uid.BARRACKS )

def BUILD_MARINE( obs, base_top_left ):
    return action_if_avalible( obs, act.TRAIN_MARINE_QUICK, [QUEUED])

def SELECT_COMMANDCENTRE( obs, base_top_left ):
    return select_building( obs, uid.COMMANDCENTER )
        
def BUILD_SVC( obs, base_top_left ):
    return action_if_avalible( obs, act.TRAIN_SCV_QUICK, [QUEUED])
        
def SELECT_ARMY( obs, base_top_left ):
    return action_if_avalible( obs, act.SELECT_ARMY, [NOT_QUEUED])

# Attack 0
def ATTACK_0_0( obs, base_top_left ):
    return attack( obs, 0, 0 , base_top_left)

def ATTACK_0_1( obs, base_top_left ):
    return attack( obs, 0, 1 , base_top_left)

def ATTACK_0_2( obs, base_top_left ):
    return attack( obs, 0, 2 , base_top_left)

def ATTACK_0_3( obs, base_top_left ):
    return attack( obs, 0, 3 , base_top_left)


# Attack 1
def ATTACK_1_0( obs, base_top_left ):
    return attack( obs, 0, 0 , base_top_left)

def ATTACK_1_1( obs, base_top_left ):
    return attack( obs, 1, 1 , base_top_left)

def ATTACK_1_2( obs, base_top_left ):
    return attack( obs, 1, 2 , base_top_left)

def ATTACK_1_3( obs, base_top_left ):
    return attack( obs, 1, 3 , base_top_left)


# Attack 2
def ATTACK_2_0( obs, base_top_left ):
    return attack( obs, 2, 0 , base_top_left)

def ATTACK_2_1( obs, base_top_left ):
    return attack( obs, 2, 1 , base_top_left)

def ATTACK_2_2( obs, base_top_left ):
    return attack( obs, 2, 2 , base_top_left)

def ATTACK_2_3( obs, base_top_left ):
    return attack( obs, 2, 3 , base_top_left)


# Attack 3
def ATTACK_3_0( obs, base_top_left ):
    return attack( obs, 3, 0 , base_top_left)

def ATTACK_3_1( obs, base_top_left ):
    return attack( obs, 3, 1 , base_top_left)

def ATTACK_3_2( obs, base_top_left ):
    return attack( obs, 3, 2 , base_top_left)

def ATTACK_3_3( obs, base_top_left ):
    return attack( obs, 3, 3 , base_top_left)