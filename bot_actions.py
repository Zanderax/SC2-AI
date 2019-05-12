from pysc2.lib import actions

import action_constants as act
import ids.unit_typeid as uid

from constants import *
import random

def isBaseTopLeft( obs ):
    player_y, player_x = (obs.observation["feature_minimap"][PLAYER_RELATIVE] == PLAYER_SELF).nonzero()
    return player_y.mean() <= 31

def transformLocation( obs, x, x_distance, y, y_distance ):
    if not isBaseTopLeft( obs ):
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

def action_if_avalible( obs, action, queue ):
    if action in obs.observation['available_actions']:
        return actions.FunctionCall(action, [queue])     


def DO_NOTHING( obs ):
        return actions.FunctionCall(act.NO_OP, [])   

def SELECT_SCV( obs ):
    unit_type = obs.observation['feature_screen'][UNIT_TYPE]
    unit_y, unit_x = (unit_type == uid.SCV).nonzero()
        
    if unit_y.any():
        i = random.randint(0, len(unit_y) - 1)
        target = [unit_x[i], unit_y[i]]
        
        return actions.FunctionCall(act.SELECT_POINT, [NOT_QUEUED, target])
        
def BUILD_SUPPLY_DEPOT( obs ):
    if act.BUILD_SUPPLYDEPOT_SCREEN in obs.observation['available_actions']:
        unit_type = obs.observation['feature_screen'][UNIT_TYPE]
        unit_y, unit_x = (unit_type == uid.COMMANDCENTER).nonzero()
        
        if unit_y.any():
            target = transformLocation(obs, int(unit_x.mean()), random.randint(0,20), int(unit_y.mean()), random.randint(20,30))
        
            return actions.FunctionCall(act.BUILD_SUPPLYDEPOT_SCREEN, [NOT_QUEUED, target])
        
def BUILD_BARRACKS( obs ):
    if act.BUILD_BARRACKS_SCREEN in obs.observation['available_actions']:
        unit_type = obs.observation['feature_screen'][UNIT_TYPE]
        unit_y, unit_x = (unit_type == uid.COMMANDCENTER).nonzero()
        
        if unit_y.any():
            target = transformLocation(obs, int(unit_x.mean()), random.randint(20,30), int(unit_y.mean()), random.randint(0,20))

            return actions.FunctionCall(act.BUILD_BARRACKS_SCREEN, [NOT_QUEUED, target])
    
def SELECT_BARRACKS( obs ):
    return select_building( obs, uid.BARRACKS )

def BUILD_MARINE( obs ):
    return action_if_avalible( obs, act.TRAIN_MARINE_QUICK, QUEUED)

def SELECT_COMMANDCENTRE( obs ):
    return select_building( obs, uid.COMMANDCENTER )
        
def BUILD_SVC( obs ):
    return action_if_avalible( obs, act.TRAIN_SCV_QUICK, QUEUED)
        
def SELECT_ARMY( obs ):
    return action_if_avalible( obs, act.SELECT_ARMY, NOT_QUEUED)
        
def ATTACK( obs ):
    if obs.observation['single_select'][0][0] != uid.SCV and act.ATTACK_MINIMAP in obs.observation["available_actions"]:
        if isBaseTopLeft( obs ):
            return actions.FunctionCall(act.ATTACK_MINIMAP, [NOT_QUEUED, [39, 45]])
        else:
            return actions.FunctionCall(act.ATTACK_MINIMAP, [NOT_QUEUED, [21, 24]])