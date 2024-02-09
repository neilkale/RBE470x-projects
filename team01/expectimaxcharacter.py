# This is necessary to find the main code
import sys

sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '../../bomberman/monsters')

# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

from world import World
from sensed_world import SensedWorld

from priority_queue import PriorityQueue

from selfpreserving_monster import SelfPreservingMonster
from stupid_monster import StupidMonster

from enum import Enum
import numpy as np
import math

MAX_DEPTH = 2

# possible action set that the character can take
class ActionSet(Enum):
    N = (0,-1)
    NE = (1,-1)
    E = (1,0)
    SE = (1,1)
    S = (0,1)
    SW = (-1,1)
    W = (-1,0)
    WN = (-1,-1)
    BOMB = (0,0)


class ExpectimaxCharacter(CharacterEntity):

    def __init__(self):
        pass
    
    def do(self, wrld):
        """
        Main function called every turn. This is where the expectimax computation starts.
        - wrld: The current state of the world.
        Output: The best action determined by expectimax (e.g., move direction).
        """
        # Start the expectimax search from the current state
        action = self.expectimax(wrld, depth=0)

        # Execute the action
        self.takeAction(action)

    def takeAction(self, wrld, action):
        """
        Executes the move based on the action
        - wrld: The current state of the world.
        - action: The action that the character will take
        """
        if action == ActionSet.BOMB:
            self.place_bomb()
        else:
            self.move(action[0],action[1])
    

    def expectimax(self, wrld, depth):
        """
        The recursive expectimax function that evaluates the possible outcomes and chooses the best action.
        - wrld: The current state of the world.
        - depth: The current depth in the search tree.
        Output: The best score and corresponding action at this level of the tree.
        """
        bestAction, bestScore = None, np.Inf

        actions = self.findPossibleActions(wrld)
        for act in actions:
            score = self.evaluateChanceNode(wrld, act, depth)
            if (score > bestScore):
                bestAction, bestScore = act, score
        return bestAction, bestScore

    def evaluateChanceNode(self, wrld, action, depth):
        possibleWorlds = self.simulateAction(wrld, action)
        # possible_worlds is [(World, probability), (World, probability), ...]
        score = 0

        for (possibleWorld, probability) in possibleWorlds:
            heuristic = self.heuristic(possibleWorld)
            if (depth >= MAX_DEPTH or abs(heuristic) >= 10):
                score += heuristic*probability
            else:
                _, possibleScore = self.expectimax(possibleWorld, depth+1)
                score += possibleScore * probability

        return score
    
    def simulateAction(self, wrld, action):
        possibleWorlds = []

        # Assume there is only one monster and every future world is equally likely (stupid monster)
        if (len(wrld.monsters.values()) == 1):
            monster = wrld.monsters.values()[0]
            actions = self.findPossibleMonsterActions(monster, wrld)
            numWorlds = len(actions)
            for act in actions:
                newWorld = SensedWorld.fromWorld(wrld)
                newMonster = newWorld.monsters.values()[0]
                newMonster.move(act[0], act[1])
                newWorld = newWorld.next()
                possibleWorlds.append((newWorld, 1/numWorlds))
            
            return possibleWorlds
            
    def findPossibleActions(self, wrld): 
        """ finds all the possible actions that can be taken
        Output: the list of the possible action that the charater can take
        """
        # is this action possible? is there a wall next to me
        possibleActions = [] # stores the list of possible action
        for action in ActionSet:
            dx,dy = action
            if action == ActionSet.BOMB:
                pass # implement this later
            else: 
                if (0 <= self.x + dx < wrld.width() and 0 <= self.y + dy < wrld.height()):
                    if (wrld.exit_at(self.x + dx, self.y + dy) or
                            wrld.empty_at(self.x + dx, self.y + dy)):
                        possibleActions.append(action)
        return possibleActions
                    
    def findPossibleMonsterActions(self, monster, wrld): 
        """ finds all the possible actions that can be taken
        Output: the list of the possible action that the charater can take
        """
        # is this action possible? is there a wall next to me
        possibleActions = [] # stores the list of possible action
        for action in ActionSet:
            dx,dy = action
            if action == ActionSet.BOMB:
                pass
            else: 
                if (0 <= monster.x + dx < wrld.width() and 0 <= monster.y + dy < wrld.height()):
                    if not wrld.wall_at(monster.x + dx, monster.y + dy):
                        possibleActions.append(action)
        return possibleActions
                    

