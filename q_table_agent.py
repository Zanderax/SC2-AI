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

ACTION_DO_NOTHING = botact.DO_NOTHING
ACTION_SELECT_SCV = botact.SELECT_SCV
ACTION_BUILD_SUPPLY_DEPOT = botact.BUILD_SUPPLY_DEPOT
ACTION_BUILD_BARRACKS = botact.BUILD_BARRACKS
ACTION_SELECT_BARRACKS = botact.SELECT_BARRACKS
ACTION_BUILD_MARINE = botact.BUILD_MARINE
ACTION_SELECT_ARMY = botact.SELECT_ARMY
ACTION_ATTACK = botact.ATTACK

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
    # ACTION_ATTACK,
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
        
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        
        return [x + x_distance, y + y_distance]
    
    def step(self, obs):
        super(QTableAgent, self).step(obs)
        # print("obs.observations.keys() - " + str(obs.observation.keys()) )
        player_y, player_x = (obs.observation['feature_minimap'][PLAYER_RELATIVE] == PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
        unit_type = obs.observation['feature_screen'][UNIT_TYPE]

        depot_y, depot_x = (unit_type == uid.SUPPLYDEPOT).nonzero()
        supply_depot_count = 1 if depot_y.any() else 0

        barracks_y, barracks_x = (unit_type == uid.BARRACKS).nonzero()
        barracks_count = 1 if barracks_y.any() else 0
            
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]

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
                reward += KILL_UNIT_REWARD
                    
            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD
                
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
                
        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action
        action = smart_action( obs )
        if action:
            return action
        return actions.FunctionCall(act.NO_OP, [])  




