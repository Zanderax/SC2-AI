## SC2 AI 
### A UTS research project to use a reinforcement learning model (bot) to play Starcraft 2
### UTS Subject 31243
### By David Belcher - UTS Student Number 11979782

The project structure is as follows:

ids/ - Contains lists of PySC2 IDs for Units and Buildings
action_constant.py - PySC2 IDs for actions
bot_actions.py - Python implementations of the possible actions the bot can make
constants.py - A list of constants used by the bot
q_table_agent.py - The main block of code for the bot. Implements the SC2 Learning Environment step() function. Also contains an copied implementation of Tensorflow's QLearningTable.
q_table_bot.bat - A windows bat file to easily run the bot
requirements.txt - Containing the list of python requirements to run the bot

This project depends on a local installation of Starcraft 2 to run.
