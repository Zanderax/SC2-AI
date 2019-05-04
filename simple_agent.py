# Following this tutorial - https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c

import random
import time

from pysc2.agents import base_agent
from pysc2.lib import actions, features

import action_constants as act
import ids.unit_typeid as uid


# Features
PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Parameters
PLAYER_SELF = 1
NOT_QUEUED = [0]
QUEUED = [1]
SUPPLY_USED = 3
SUPPLY_MAX = 4

class SimpleAgent(base_agent.BaseAgent):
    base_top_left = None
    supply_depot_built = False
    scv_selected = False
    barracks_built = False
    barracks_selected = False
    barracks_rallied = False
    obs = None
    selected = 0

    def step(self, obs):
        super(SimpleAgent, self).step(obs)
        self.obs = obs

        # Figure out starting location by taking the mean position of nearby units
        if self.base_top_left is None:
            # for key, value in obs.observation.items():
            #     print(key,value)
            player_y, player_x = (obs.observation["feature_minimap"][PLAYER_RELATIVE] == PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31

        if obs.observation["player"][SUPPLY_USED] == obs.observation["player"][SUPPLY_MAX]:
            self.supply_depot_built = False

        # Build Supply Depot
        if not self.supply_depot_built:
            if self.selected != uid.SCV:
                action_res, target = self.select_unit_action( uid.SCV ) 
                if action_res != act.NO_OP:
                    return actions.FunctionCall(action_res, [NOT_QUEUED, target])
            elif self.canDoAction( act.BUILD_SUPPLYDEPOT_SCREEN ):
                unit_type = obs.observation["feature_screen"][UNIT_TYPE]
                unit_y, unit_x =(unit_type == uid.COMMANDCENTER).nonzero()
                target = self.transformLocation(int(unit_x.mean()), random.randint(0,20), int(unit_y.mean()),random.randint(0,20))
                self.supply_depot_built = True
                return self.getActionFunction(act.BUILD_SUPPLYDEPOT_SCREEN, target)

        # Build Barracks
        if not self.barracks_built and self.canDoAction( act.BUILD_BARRACKS_SCREEN ):
            unit_type = obs.observation["feature_screen"][UNIT_TYPE]
            unit_y, unit_x =(unit_type == uid.COMMANDCENTER).nonzero()
            target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
            self.barracks_built = True
            return self.getActionFunction(act.BUILD_BARRACKS_SCREEN, target)

        # Build Soldiers
        if not self.barracks_rallied:
            if self.selected != uid.BARRACKS:
                action_res, target = self.select_unit_action( uid.BARRACKS ) 
                if action_res != act.NO_OP:
                    return actions.FunctionCall(action_res, [NOT_QUEUED, target])
            #Rally Barracks
            elif self.canDoAction(act.RALLY_UNITS_MINIMAP):
                self.barracks_rallied = True
                if self.base_top_left:
                    return actions.FunctionCall(act.RALLY_UNITS_MINIMAP, [NOT_QUEUED, [29, 21]])
                return actions.FunctionCall(act.RALLY_UNITS_MINIMAP, [NOT_QUEUED, [29, 46]])

        if self.barracks_rallied:
            if self.selected != uid.BARRACKS:
                action_res, target = self.select_unit_action( uid.BARRACKS ) 
                if action_res != act.NO_OP:
                    return actions.FunctionCall(action_res, [NOT_QUEUED, target])
            if self.canDoAction(act.TRAIN_MARINE_QUICK):
                return actions.FunctionCall(act.TRAIN_MARINE_QUICK, [QUEUED])

        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    def transformLocation( self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def canDoAction( self, action ):
        return action in self.obs.observation["available_actions"]

    def printActions( self ):
        print("actions avalible" )
        print( self.obs.observation["available_actions"] )


    def getActionFunction( self, action, target ):
        return actions.FunctionCall(action, [NOT_QUEUED, target])

    def select_unit_action( self, unit_to_select ):
        unit_type = self.obs.observation["feature_screen"][UNIT_TYPE]
        unit_y, unit_x = (unit_type == unit_to_select).nonzero()
        if unit_y.any():
            target = [unit_x[0], unit_y[0]]
            self.selected = unit_to_select
            return act.SELECT_POINT, target
        return act.NO_OP, None

