# Following this tutorial - https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c

import time

from pysc2.agents import base_agent
from pysc2.lib import actions, features

import action_constants as act
import ids.unit_typeid


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

    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        # Figure out starting location by taking the mean position of nearby units
        if self.base_top_left is None:
            # for key, value in obs.observation.items():
            #     print(key,value)
            player_y, player_x = (obs.observation["feature_minimap"][PLAYER_RELATIVE] == PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31

        if not self.supply_depot_built:
            if not self.scv_selected:
                unit_type = obs.observation["feature_screen"][UNIT_TYPE]
                unit_y, unit_x = (unit_type == SCV).nonzero()

                target = [unit_x[0], unit_y[0]]
                self.scv_selected = True
                return actions.FunctionCall(act.SELECT_POINT, [NOT_QUEUED, target])
            elif self.canDoAction( act.BUILD_SUPPLYDEPOT_SCREEN, obs ):
                unit_type = obs.observation["feature_screen"][UNIT_TYPE]
                unit_y, unit_x =(unit_type == COMMANDCENTER).nonzero()
                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()),20)
                self.supply_depot_built = True
                return self.getActionFunction(act.BUILD_SUPPLYDEPOT_SCREEN, target)
        elif not self.barracks_built and self.canDoAction( act.BUILD_BARRACKS_SCREEN, obs ):
            unit_type = obs.observation["feature_screen"][UNIT_TYPE]
            unit_y, unit_x =(unit_type == COMMANDCENTER).nonzero()
            target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
            self.barracks_built = True
            return self.getActionFunction(act.BUILD_BARRACKS_SCREEN, target)
        
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    def transformLocation( self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def canDoAction( self, action, obs ):
        return action in obs.observation["available_actions"]

    def getActionFunction( self, action, target ):
        return actions.FunctionCall(action, [NOT_QUEUED, target])
