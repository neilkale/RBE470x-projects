# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

from enum import Enum

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

class minimaxCharacter(CharacterEntity):

    def __init__(self):
        pass

    def do(self, wrld):
        # Your code here
        """
        Main function called every turn. This is where the minimax computation starts.
        - wrld: The current state of the world.
        Output: The best action determined by minimax (e.g., move direction).
        """
        # Start the expectimax search from the current state
        action = self.minimax(wrld, depth=0)

        # Execute the action
        self.takeAction(wrld, action)

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

    def minimax(self, wrld, depth):
        pass

    def maxValue(self,wrld,depth):
        pass

    def minValue(self, wrld, depth):
        pass


    

    

