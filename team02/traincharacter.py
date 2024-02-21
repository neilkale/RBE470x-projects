# This is necessary to find the main code
# System imports
import sys
sys.path.insert(0, '../bomberman')
sys.path.insert(1, '..')

# AStar Imports
from priority_queue import PriorityQueue
from world import World

# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from events import Event

# Math imports
from enum import Enum
import numpy as np
from random import random
import math

# Unit testing and debugging imports
from icecream import ic

s, a = None, None
GAMMA = 0.8 #tune value
ALPHA = 0.2 #tune value
EPSILON = 0.8 #tune value
WEIGHTS_FILE_NAME = "weights.npy"
VERBOSE = True

CREATE_NEW_WEIGHTS = False
NUM_WEIGHTS = 7

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
        if (CREATE_NEW_WEIGHTS):
            self.w = np.ones(NUM_WEIGHTS)
        else:
            self.w =  np.load(WEIGHTS_FILE_NAME) # np.array([1.,1.,1.,1.]) # (w1, w2, w3, w4)
        self.w = (self.w - np.mean(self.w))
        
    def do(self, s_prime):
        global s,a 
        
        # s and a are from the previous iteration
        if (s == None): # this was the first step
            pass
        else: # this was not the first step
            # Get (s, a, s', r)
            r = self.getReward(s_prime)
            # Compute delta
            delta = r + GAMMA * max([self.getQValue(s_prime,a_prime) for a_prime in self.getActions(s_prime)]) - self.getQValue(s,a)
            # Update the weights
            self.w += ALPHA * delta * self.getFeatures(s,a)
            self.w = (self.w - np.mean(self.w))

        # Set s = s'
        s = s_prime
        # Find all possible actions and their Q-values
        actions = self.getActions(s)
        qValues = [self.getQValue(s,a) for a in actions]
        if VERBOSE: ic(actions)
        if VERBOSE: ic([self.getFeatures(s,a) for a in actions])

        if VERBOSE: ic(self.w)

        if VERBOSE: ic(qValues)

        a = self.selectAction(actions, qValues)

        if isTerminalState(s):
            np.save(WEIGHTS_FILE_NAME, self.w)

        # Take the action
        self.takeAction(a)
    
    # Compute the Q-value for the inputted state-action pair.
    def getQValue(self, s, a):
        f = self.getFeatures(s, a)
        return np.dot(self.w, f)

    # Compute the f-values required to compute the Q-value for a state-action pair
    def getFeatures(self, s, a):
        MAXDIST = s.width() + s.height() # world height and wrld grid
        char_x = self.x + a.value[0]
        char_y = self.y + a.value[1]

        # distance to the exit cell
        f_e = aStarDist(s, (char_x,char_y), s.exitcell) #aStar dist to the exit 

        # Distance to the closest monster
        f_m = MAXDIST
        for i in range(len(list(s.monsters.values()))):
            monster = list(s.monsters.values())[i][0]
            f_m = min(f_m, aStarDist(s, (char_x,char_y), (monster.x,monster.y))) #aStar dist to the monster 
        
        # Distance to the closest explosion cell
        f_x = MAXDIST
        for i in range(len(list(s.explosions.values()))):
            explosion = list(s.explosions.values())[i]
            f_x = min(f_x, manhattanDist((char_x,char_y), (explosion.x,explosion.y)))
        
        # Distance to the closest bomb cell
        f_b = MAXDIST
        for i in range(len(list(s.bombs.values()))):
            bomb = list(s.bombs.values())[i]
            f_b = min(f_b, manhattanDist((char_x,char_y), (bomb.x,bomb.y))) #aStar dist to the bombs

        # Binary feature which returns 1 if character is in the same line (x axis or y-axis) as a bomb
        f_b_xy = 0
        for i in range(len(list(s.bombs.values()))):
            bomb = list(s.bombs.values())[i]
            if bomb.x == char_x or bomb.y == char_y: 
                f_b_xy = 1
                break
        
        # Binary feature which returns 1 if character is inside an explosion.
        f_x_in = 0
        for i in range(len(list(s.explosions.values()))):
            explosion = list(s.explosions.values())[i]
            if explosion.x == char_x and explosion.y == char_y:
                f_x_in = 1
                break
        
        # Manhattan distance to the exit cell
        f_e_m = manhattanDist((char_x, char_y), s.exitcell)
        
        # normalization
        f_e = 1 - f_e/MAXDIST
        f_m = 1 - f_m/MAXDIST
        f_x = 1 - f_x/MAXDIST
        f_b = 1 - f_b/MAXDIST
        f_e_m = 1 - f_e_m/MAXDIST

        return np.array([f_e, f_e_m, f_m, f_x, f_x_in, f_b, f_b_xy])
        
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
    
    def selectAction(self, actions, qValues):
        if random() < EPSILON:
            idx = np.argmax(qValues)
        else:
            idx = int(random() * len(actions))
        return actions[idx]
    
    # Returns the reward of state s based on the events in s
    def getReward(self, s):
        for event in s.events:
            if (event.tpe == 0 or event.tpe == 1): # Bomb hit wall or bomb hit monster
                return 1000
            elif (event.tpe == 2 or event.tpe == 3): # Character killed
                return -2000
            elif (event.tpe == 4): # Character found exit
                return 2000
        return -manhattanDist((self.x, self.y), s.exitcell)*10 # Default reward

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

def aStarDist(s, point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    path = AStar.a_star(s, (x1, y1), (x2, y2))
    if path == None:
        return s.width() + s.height() # world height and wrld grid
    else:
        return len(path)

class AStar():

    def a_star(wrld: World, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates the path using A Star
        :param wrld [World]                  The map data.
        :start: tuple[int, int]                 The starting position (grid coord)
        :goal: tuple[int, int]                 The goal position (grid coord)
        :return        list[tuple[int, int]]    The path i.e. list of points (grid coord)
        """

        if (start == goal):
            return []

        frontier = PriorityQueue() # frontier 
        frontier.put(start,0) # adding the start to the frontier 
        came_from = {} # list of linked list of the path to each node
        cost_so_far = {} # cost to each node
        heuristic_so_far = {} # heuristic of each node
        came_from[start] = None # setting the came_from to None for start
        cost_so_far[start] = 0 # setting the cost of the start to 0

        # keep looping until no more nodes in frontier
        while not frontier.empty():
            current = frontier.get() # get the first node
            if current == goal: # reached to the goal
                break 
            for next in AStar.getNeighborsOfEight(current, wrld, goal): # get the list of neighbors 
                # calculate the new cost
                new_cost = cost_so_far[current] + 1 
                heuristic = AStar.heuristic(goal, next)
                # true if the node has not been visited or if the next node costs less 
                if not (next in cost_so_far) or new_cost < cost_so_far[next]: # or heuristic < heuristic_so_far[next]:
                    cost_so_far[next] = new_cost # set the cost 
                    heuristic_so_far[next] = heuristic
                    priority =  new_cost + heuristic # calculate the priority
                    frontier.put(next, priority) # add the node to the priority queue based on the cost 
                    came_from[next] = current # set the path of the node

         # Check if the goal was reached
        if goal not in came_from:
            return None  # Return None if there's no path to the goal

        path = [] # the optimized path 
        current = goal # go back wards
        while True:
            path.insert(0, current) # add the path to the list
            current = came_from[current] # set the curent to the node current came from
            if(current == start): # true if we reach the start
                break
        
        # return the path
        return path
        
    def getNeighborsOfEight(cell: tuple[int,int], wrld: World, goal: tuple[int, int]):
        # List of empty cells
        neighbors = []
        x,y = cell
        # Go through neighboring cells
        for dx in [-1, 0, 1]:
            # Avoid out-of-bounds access
            if ((x + dx >= 0) and (x + dx < wrld.width())):
                for dy in [-1, 0, 1]:
                    # Avoid out-of-bounds access
                    if ((y + dy >= 0) and (y + dy < wrld.height())):
                        # Is this cell safe and not a non-move?
                        if  (wrld.exit_at(x + dx, y + dy) or
                        wrld.empty_at(x + dx, y + dy) or (x + dx, y + dy) == goal):
                            # Yes
                            neighbors.append((dx, dy))
        # All done
        return [(x+dx,y+dy) for (dx,dy) in neighbors]
        
    def heuristic(goal, next):
        goal_dist = AStar.euclidean_distance(goal, next)
        return goal_dist
                
    def euclidean_distance(cell1, cell2):
        return math.dist(cell1, cell2)