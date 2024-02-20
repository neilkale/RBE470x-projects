# This is necessary to find the main code
from enum import Enum
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import numpy as np
from events import Event

s, a = None, None
GAMMA = 0.8 #tune value
ALPHA = 0.7 #tune value
WEIGHTS_FILE_NAME = "weights.txt"

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
    NOTHING = (0,0)

class TrainCharacter(CharacterEntity):

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.w = readFile(WEIGHTS_FILE_NAME) # (w1, w2, w3)
        
    def do(self, s_prime):
        global s,a 
        
        # s and a are from the previous iteration
        if (s == None): # this was the first step
            pass
        else: # this was not the first step
            # Get (s, a, s', r)
            r = getReward(s_prime)
            # Compute delta
            delta = r + GAMMA * max([self.getQValue(s_prime,a_prime) for a_prime in self.getActions(s_prime)]) - self.getQValue(s,a)
            # Update the weights
            w += ALPHA * delta * self.getFValues(s,a)

        # Set s = s'
        s = s_prime
        # Find all possible actions and their Q-values
        actions = self.getActions(s)
        qValues = [self.getQValue(s,a) for a in actions]
        a = self.selectAction(actions, qValues)
        # Take the action
        self.takeAction(a)

        if isTerminalState(s):
            writeFile(self.w)
    
    # Compute the Q-value for the inputted state-action pair.
    def getQValue(self, s, a):
        f = self.getFValues(s, a)
        return np.dot(self.w, f)

    # Compute the f-values required to compute the Q-value for a state-action pair
    def getFValues(self, s, a):
        MAXDIST = s.width() + s.height() # world height and wrld grid
        # fs should be normalized from 0 to 1
        f_e = manhattanDist((self.x,self.y), s.exitcell) #manhattan dist to the exit 

        f_m = MAXDIST
        for i in range(len(list(s.monsters.values()))):
            monster = list(s.monsters.values())[i][0]
            f_m = manhattanDist((self.x,self.y), (monster.x,monster.y)) #manhattan dist to the monster 
        
        f_x = MAXDIST
        for i in range(len(list(s.monsters.values()))):
            explosion = list(s.explosion.values())[i][0]
            f_x = manhattanDist((self.x,self.y), (explosion.x,explosion.y)) #manhattan dist to the explosion
        
        # normalization
        f_e /= MAXDIST
        f_m /= MAXDIST
        f_x /= MAXDIST

        return (f_e, f_m, f_x)
        
    def takeAction(self, action):
        """
        Executes the move based on the action
        - wrld: The current state of the world.
        - action: The action that the character will take
        """
        if action == ActionSet.BOMB:
            self.place_bomb()
        else:
            self.move(action.value[0],action.value[1])

    def getActions(self, s):
        """ finds all the possible actions that can be taken
        Output: the list of the possible action that the charater can take
        """
        # is this action possible? is there a wall next to me
        possibleActions = [] # stores the list of possible action
        for action in ActionSet:
            dx,dy = action.value[0], action.value[1]
            if action == ActionSet.BOMB:
                possibleActions.append(action)
            else: 
                if (0 <= self.x + dx < s.width() and 0 <= self.y + dy < s.height()):
                    if not s.wall_at(self.x + dx, self.y + dy):
                        possibleActions.append(action)
        return possibleActions
    
    def selectAction(self, action, qValues):
        pass
        
# Returns the reward of state s based on the events in s
def getReward(s):
    for event in s.events:
        if (event.tpe == 2 or event.tpe == 3): # Character killed
            return -1000
        elif (event.tpe == 4): # Character found exit
            return 1000
    return -1 # Default reward

# Returns true if the agent is dead or has reached the exit, otherwise false
def isTerminalState(s):
    events = [event.tpe for event in s.events]
    if (Event.CHARACTER_FOUND_EXIT in events or Event.BOMB_HIT_CHARACTER in events or Event.CHARACTER_KILLED_BY_MONSTER in events):
        return True
    return False

# Finds the manhattan distance between two entities
def manhattanDist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return abs(x2 - x1) + abs(y2 - y1)