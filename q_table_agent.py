# Following Tutorial from here https://chatbotslife.com/building-a-smart-pysc2-agent-cdc269cb095d

import random
import math

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import bot_actions as botact
import action_constants as act
import ids.unit_typeid as uid

from constants import *

bot_actions = [
    botact.DO_NOTHING,
    botact.SELECT_SCV,
    botact.BUILD_SUPPLY_DEPOT,
    botact.BUILD_BARRACKS,
    botact.SELECT_BARRACKS,
    botact.BUILD_MARINE,
    botact.SELECT_ARMY,
    #Attack 0
    botact.ATTACK_0_0,
    botact.ATTACK_0_1,
    botact.ATTACK_0_2,
    botact.ATTACK_0_3,
    #Attack 1
    botact.ATTACK_1_0,
    botact.ATTACK_1_1,
    botact.ATTACK_1_2,
    botact.ATTACK_1_3,
    #Attack 2
    botact.ATTACK_2_0,
    botact.ATTACK_2_1,
    botact.ATTACK_2_2,
    botact.ATTACK_2_3,
    #Attack 3
    botact.ATTACK_3_0,
    botact.ATTACK_3_1,
    botact.ATTACK_3_2,
    botact.ATTACK_3_3,
    # botact.SELECT_COMMANDCENTRE,
    # botact.BUILD_SVC,
]



# Taken from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class QTableAgent(base_agent.BaseAgent):
    def __init__(self):
        super(QTableAgent, self).__init__()
        
        self.qlearn = QLearningTable(actions=list(range(len(bot_actions))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None
        self.previous_minerals = 0
        self.previous_food_used = 0
    
    def step(self, obs):
        super(QTableAgent, self).step(obs)

        player_y, player_x = (obs.observation['feature_minimap'][PLAYER_RELATIVE] == PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
        unit_type = obs.observation['feature_screen'][UNIT_TYPE]

        depot_y, depot_x = (unit_type == uid.SUPPLYDEPOT).nonzero()
        supply_depot_count = 1 if depot_y.any() else 0

        barracks_y, barracks_x = (unit_type == uid.BARRACKS).nonzero()
        barracks_count = 1 if barracks_y.any() else 0
            
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        minerals = obs.observation['player'][1]

        # Food used [0] by army [1]
        food_used = obs.observation['score_by_category'][0][1]
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]

        
        current_state = [
            supply_depot_count,
            barracks_count,
            supply_limit,
            army_supply,
        ]

        if self.previous_action is not None:
            reward = 0
                
            if killed_unit_score > self.previous_killed_unit_score:
                print( "Killed unit = " + str(killed_unit_score))
                reward += KILL_UNIT_REWARD
                    
            if killed_building_score > self.previous_killed_building_score:
                print( "Killed building  = " + str(killed_building_score))
                reward += KILL_BUILDING_REWARD

            # Reward and punish when gaining/losing army units
            if food_used != self.previous_food_used:
                print( "Food used by army = " + str(food_used))
                reward += FOOD_REWARD * ( food_used - self.previous_food_used )
            
            # if minerals > self.previous_minerals:
            #     reward += ( minerals - self.previous_minerals ) / 10
                
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

            if obs.last():
                self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
                
        # Update qtable and decide on action
        rl_action = self.qlearn.choose_action(str(current_state))
        action = bot_actions[rl_action]

        # Update scores
        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action
        self.previous_minerals = minerals
        self.previous_food_used = food_used

        # Create action object to return to pysc2
        res = action( obs, self.base_top_left )
        if res:
            return res
        return actions.FunctionCall(act.NO_OP, [])  




