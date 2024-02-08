# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '../../bomberman/monsters')

# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

from world import World
import math
from priority_queue import PriorityQueue

from selfpreserving_monster import SelfPreservingMonster
from stupid_monster import StupidMonster

from enum import Enum

dumbMonsterExists = False
smartMonsterExist = {}

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
        depth = 0
        action = self.expectimax(wrld, depth)

        # Decide the best action based on expectimax result
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
        bestAction = None
        # bestScore = numpy  

        # Base case: if terminal state or maximum depth reached, return the evaluated score

        # If the current agent is the player, call the max_value function

        # If the current agent is a monster/environment, call the exp_value function

    def findPossibleAction(self, wrld): 
        """ finds all the possible actions that can be taken
        Output: the list of the possible action that the charater can take
        """
        # is this action possible? is there a wall next to me
        possibleActions = [] # stores the list of possible action
        x,y = wrld.characters_at() # stores the current location of the character
        for action in ActionSet:
            dx,dy = action
            if action == ActionSet.BOMB:
                # implement this later
                pass
            else: 
                if (wrld.exit_at(x + dx, y + dy) or
                        wrld.empty_at(x + dx, y + dy)):
                    possibleActions.append(action)

        return possibleActions
                    
        

