# This is necessary to find the main code
# System imports
import sys

from sensed_world import SensedWorld
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
suicidal = False

GAMMA = 0.4 #tune value
ALPHA = 0.05 #tune value
EPSILON = 0.5 #tune value
WEIGHTS_FILE_NAME = "weights.npy"
VERBOSE = True

CREATE_NEW_WEIGHTS = False
NUM_WEIGHTS = 16

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
    BOMB = None
    NOTHING = (0,0)

class TrainCharacter(CharacterEntity):

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        if (CREATE_NEW_WEIGHTS):
            self.w = np.ones(NUM_WEIGHTS)
        else:
            self.w =  np.load(WEIGHTS_FILE_NAME) # np.array([1.,1.,1.,1.]) # (w1, w2, w3, w4)
            # self.w = (self.w - np.mean(self.w)) / np.std(self.w)
        
    def do(self, s_prime):
        global s,a 
        
        # s and a are from the previous iteration
        if (s == None): # this was the first step
            pass
        else: # this was not the first step
            # Get (s, a, s', r)
            r = self.getReward(s_prime, s, a)
            if VERBOSE: ic(a)
            if VERBOSE: ic(r)
            if VERBOSE: print()
            # Compute delta
            delta = r + GAMMA * max([self.getQValue(s_prime,a_prime) for a_prime in self.getActions(s_prime)]) - self.getQValue(s,a)
            # Update the weights
            self.w += ALPHA * delta * self.getFeatures(s,a)
            # self.w = (self.w - np.mean(self.w)) / np.std(self.w)
        
        # Set s = s'
        # s' -> current world
        s = SensedWorld.from_world(s_prime)
        # Find all possible actions and their Q-values
        actions = self.getActions(s)
        features = [self.getFeatures(s,a) for a in actions]
        qValues = [self.getQValue(s,a) for a in actions]

        # This is solely for debugging.
        if VERBOSE: ic(actions)
        if VERBOSE: ic(features)
        if VERBOSE: ic(self.w)
        if VERBOSE: ic(qValues)
        if VERBOSE: print('\n\n\n')

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
        if a == ActionSet.BOMB:
            char_x, char_y = self.x, self.y
        else:
            char_x = self.x + a.value[0]
            char_y = self.y + a.value[1]

        # distance to the exit cell
        f_e = aStarDist(s, (char_x,char_y), s.exitcell) #aStar dist to the exit 

        # Binary feature which returns 1 if the agent moves closer to the target
        f_e_c = 0
        if aStarDist(s, (char_x, char_y), s.exitcell) < aStarDist(s, (self.x, self.y), s.exitcell):
            f_e_c = 1

        # Manhattan distance to the exit cell
        f_e_m = manhattanDist((char_x, char_y), s.exitcell)

        # Distance to the closest monster
        f_m = MAXDIST
        for i in range(len(list(s.monsters.values()))):
            monster = list(s.monsters.values())[i][0]
            f_m = min(f_m, aStarDist(s, (char_x,char_y), (monster.x,monster.y))) #aStar dist to the monster 
        
        # Binary feature which returns 1 if the agent moves within:
        # 2 step of 'stupid' monster, 3 step of 'self-preserving monster', 4 step of 'aggressive monster'
        # Early detection warning...
        f_m_w = 0
        for i in range(len(list(s.monsters.values()))):
            monster = list(s.monsters.values())[i][0]
            if monster.name == 'stupid' and aStarDist(s, (char_x,char_y), (monster.x,monster.y)) <= 2:
                f_m_w = 1
            if monster.name == 'selfpreserving' and aStarDist(s, (char_x,char_y), (monster.x,monster.y)) <= 3:
                f_m_w = 1
            if monster.name == 'aggressive' and aStarDist(s, (char_x,char_y), (monster.x,monster.y)) <= 4:
                f_m_w = 1


        # Binary feature which returns 1 if the agent moves within:
        # 1 step of 'stupid' monster, 2 step of 'self-preserving monster', 3 step of 'aggressive monster'
        f_m_c = 0
        for i in range(len(list(s.monsters.values()))):
            monster = list(s.monsters.values())[i][0]
            if monster.name == 'stupid' and aStarDist(s, (char_x,char_y), (monster.x,monster.y)) <= 1:
                f_m_c = 1
            if monster.name == 'selfpreserving' and aStarDist(s, (char_x,char_y), (monster.x,monster.y)) <= 2:
                f_m_c = 1
            if monster.name == 'aggressive' and aStarDist(s, (char_x,char_y), (monster.x,monster.y)) <= 3:
                f_m_c = 1

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
        #   and within 4 steps of a bomb 
        #   and the bomb is about to go off in the next turn
        f_b_xy = 0
        for i in range(len(list(s.bombs.values()))):
            bomb = list(s.bombs.values())[i]
            if manhattanDist((char_x,char_y), (bomb.x,bomb.y)) <= 4 and (bomb.x == char_x or bomb.y == char_y) and bomb.timer <= 2: 
                f_b_xy = 1
                break
                
        # Binary feature which returns 1 if character is on top of a bomb
        f_b_on = 0
        for i in range(len(list(s.bombs.values()))):
            bomb = list(s.bombs.values())[i]
            if bomb.x == char_x and bomb.y == char_y: 
                f_b_on = 1
                break

        # Binary feature which returns 1 if the move is to drop a bomb
        f_b_drop = 0
        if a == ActionSet.BOMB:
            f_b_drop = 1

        # Binary feature which returns 1 if the move is to do nothing
        f_d_n = 0
        if a == ActionSet.NOTHING:
            f_d_n = 1

        # Binary feature which returns 1 if character is adjacent to an explosion. 
        f_x_in = 0
        for i in range(len(list(s.explosions.values()))):
            explosion = list(s.explosions.values())[i]
            if manhattanDist((char_x,char_y), (explosion.x,explosion.y)) == 1:
                f_x_in = 1
                break

        # Binary feature which returns true if character's next step in AStar path is a wall and it is dropping a bomb.
        # The weight associated with this feature should be super positive to encourage this behavior.
        f_w_n = 0
        if a == ActionSet.BOMB:
            path = AStarWithWalls.a_star(s, (char_x,char_y), s.exitcell)
            if path != [] and s.wall_at(path[0][0], path[0][1]):
                f_w_n = 1

        # Returns the number of walls which a bomb could explode next to the character if the action is dropping a bomb.
        # Basically, the number of walls which will be exploded by the current action.
        f_w_b = 0
        if a == ActionSet.BOMB:
            f_w_b = getBombWalls((char_x, char_y), s)

        # Number of walls next to the character
        f_w = getNumWalls((char_x, char_y), s)
        
        # normalization
        f_e = 1 - f_e/MAXDIST
        # f_m = 1 - f_m/MAXDIST
        # f_x = 1 - f_x/MAXDIST
        # f_b = 1 - f_b/MAXDIST
        # f_e_m = 1 - f_e_m/MAXDIST

        # f_e = 1/(1 + f_e)
        f_m = 1/(1 + f_m)
        f_x = 1/(1 + f_x)
        f_b = 1/(1 + f_b)
        f_e_m = 1/(1 + f_e_m)
        f_w_b = 1/(1+ f_w_b)

        f_w = f_w/8

        features = np.array([f_e, f_e_m, f_e_c, f_m, f_m_w, f_m_c, f_x, f_x_in, f_b, f_b_xy, f_b_on, f_b_drop, f_d_n, f_w, f_w_b, f_w_n])

        return np.round(features, decimals = 3)
        
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
            if action == ActionSet.BOMB:
                if len(list(s.bombs.values())) == 0 and getNumExplosions((self.x,self.y),s) == 0:
                    possibleActions.append(action)
            elif action == ActionSet.NOTHING:
                possibleActions.append(action)
            else: 
                dx,dy = action.value[0], action.value[1]
                if (0 <= self.x + dx < s.width() and 0 <= self.y + dy < s.height()):
                    if (s.exit_at(self.x + dx, self.y + dy) or
                        s.empty_at(self.x + dx, self.y + dy)):
                        possibleActions.append(action)
        return possibleActions
    
    def selectAction(self, actions, qValues):
        if random() > EPSILON:
            idx = np.argmax(qValues)
        else:
            idx = int(random() * len(actions))
        return actions[idx]
    
    # Returns the reward of state s based on the events in s
    def getReward(self, s_prime, s, a):
        events = [event.tpe for event in s_prime.events]
        # If goal is reached, reward the agent.
        if 4 in events:
            return 1000    
        # If the agent was suicidal on a previous turn (so guaranteed to die now)
        # Don't penalize its current action as its irrelevant to its fate.
        global suicidal
        if suicidal:
            return 0
        # If the agent takes a suicidal step - punish it
        if len(list(s.bombs.values())) > 0:
            for i in range(len(list(s.bombs.values()))):
                bomb = list(s.bombs.values())[i]
                if manhattanDist((self.x,self.y), (bomb.x,bomb.y)) <= 4 and (bomb.x == self.x or bomb.y == self.y) and bomb.timer == 1: 
                    suicidal = True
                    s_prime.time = -1
                    break
        # If the agent takes a suicidal step towards monsters - punish it
        if len(list(s.monsters.values())) > 0:
            for i in range(len(list(s.monsters.values()))):
                monster = list(s.monsters.values())[i][0]
                if monster.name == 'selfpreserving' and aStarDist(s, (self.x,self.y), (monster.x,monster.y)) < 2:
                    suicidal = True
                    s_prime.time = -1
                if monster.name == 'aggressive' and aStarDist(s, (self.x,self.y), (monster.x,monster.y)) < 3:
                    suicidal = True
                    s_prime.time = -1
    
        if suicidal:
            return -1000
        # if the agent dies, punish it
        if 2 in events or 3 in events:
            return -1000
        
        # # If the agent blows up a wall, reward it.
        # if 0 in events or 1 in events:
        #     return 200
        # # If the agent places a useful bomb, reward it.
        # elif (len(list(s.explosions.values())) == 0 and 
        #           len(list(s_prime.explosions.values())) != 0 ):
        #     return -100

        # If the action was to drop a bomb:
        if (a == ActionSet.BOMB):
        #    If the bomb was useful: 
            numWalls = getBombWalls((self.x, self.y), s)
            if(numWalls > 0):
                return 40
        #    Else if the bomb was not useful:
            else:
                return -40
        # Otherwise, the action was to move position 
        # If the action moves to one square away from a stupid monster, punish the agent
        for i in range(len(list(s_prime.monsters.values()))):
            monster = list(s_prime.monsters.values())[i][0]
            if manhattanDist((self.x, self.y),(monster.x,monster.y)) <= 1:
                return -40
                             
        currDist = aStarDist(s_prime,(self.x, self.y), s_prime.exitcell) 
        oldChar = list(s.characters.values())[0][0]
        oldDist = aStarDist(s,(oldChar.x, oldChar.y), s.exitcell)
        if currDist < oldDist: # we are closer to the exit
            return 20
        elif currDist == oldDist: # we are the same distance to the exit
            return -25
        else: # we are further to the exit
            return -25

# Returns the number of walls that a bomb can destroy
def getBombWalls(bombPosition, s):
    bombWalls = 0
    for dx in [-4,-3,-2,-1,0,1,2,3,4]:
        if (0 <= bombPosition[0] + dx < s.width()):
            if (s.wall_at(bombPosition[0]+dx, bombPosition[1])):
                bombWalls += 1
    for dy in [-4,-3,-2,-1,0,1,2,3,4]:
        if (0 <= bombPosition[1] + dy < s.height()):
            if (s.wall_at(bombPosition[0], bombPosition[1]+dy)):
                bombWalls += 1
    return bombWalls

# Gets the number of actions from the inputted position
def getActions(position, s):
    """ finds all the possible actions that can be taken
    Output: the list of the possible action that the character can take
    """
    # is this action possible? is there a wall next to me
    possibleActions = [] # stores the list of possible action
    for action in ActionSet:
        if action == ActionSet.BOMB:
            if len(list(s.bombs.values())) == 0 and getNumExplosions(position,s) == 0:
                possibleActions.append(action)
        elif action == ActionSet.NOTHING:
            possibleActions.append(action)
        else: 
            dx,dy = action.value[0], action.value[1]
            if (0 <= position[0] + dx < s.width() and 0 <= position[1] + dy < s.height()):
                if (s.exit_at(position[0] + dx, position[1] + dy) or
                     s.empty_at(position[0] + dx, position[1] + dy)):
                    possibleActions.append(action)
    return possibleActions

# Get the number of walls next to the inputted position
def getNumWalls(position, s):
    numWalls = 0
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            if ((position[0] + dx < 0 or position[0] + dx >= s.width())
                or
                (position[1] + dy < 0 or position[1] + dy >= s.height())):
                numWalls += 1
            elif (s.wall_at(position[0] + dx, position[1] + dy)):
                numWalls += 1
    return numWalls

def getNumExplosions(position, s):
    numBooms = 0
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            if ((0 <= position[0] + dx < s.width())
                and
                (0 <= position[1] + dy < s.height())
                and 
                (s.explosion_at(position[0] + dx, position[1] + dy))):
                numBooms += 1
    return numBooms

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
        return len(AStarWithWalls.a_star(s, (x1, y1), (x2, y2)))
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
    
class AStarWithWalls():

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
            for next, isWall in AStarWithWalls.getNeighborsOfEight(current, wrld, goal): # get the list of neighbors 
                # calculate the new cost
                if isWall:
                    new_cost = cost_so_far[current] + 100
                else:
                    new_cost = cost_so_far[current] + 1
                heuristic = AStarWithWalls.heuristic(goal, next)
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
                            neighbors.append(((dx, dy), False))
                        # Is this cell a 'wall move'?
                        else:
                            # Yes
                            neighbors.append(((dx, dy), True))

        # All done
        return [((x+dx,y+dy),isWall) for (dx,dy),isWall in neighbors]
        
    def heuristic(goal, next):
        goal_dist = AStar.euclidean_distance(goal, next)
        return goal_dist
                
    def euclidean_distance(cell1, cell2):
        return math.dist(cell1, cell2)