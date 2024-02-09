# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

from world import World
from sensed_world import SensedWorld

from priority_queue import PriorityQueue

import math
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

inf = 100 
ass = AStar() 

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
        """
        The recursive minimax function that evaluates the possible outcomes and chooses the best action.
        - wrld: The current state of the world.
        - depth: The current depth in the search tree.
        Output: The best score and corresponding action at this level of the tree.
        """
        # Get list of legal actions (moves)
        possibleActions = self.findPossibleActions(wrld)

        bestScore = float(inf)
        bestAction = None

        # Start the search 
        for action in possibleActions:
            nextWorld = self.simulateActionSimple(wrld,action)
            score = self.maxValue(nextWorld)

            if score < bestScore:
                bestScore = score
                bestAction = action

        return bestAction, bestScore
    
    def maxValue(self, wrld):
        """
        Max-value function of minimax
        - wrld: The current state of the world.
        Output: The maximum utility value
        """
        if self.terminalState(wrld):    
            return self.heuristic(wrld)
        

    def heuristics(self, wrld):
        events = [event.tpe for event in wrld.events]
        if (Event.CHARACTER_FOUND_EXIT in events): # yay we are wining, return good heuristics (negative for us)
            return -inf
        elif (Event.CHARACTER_KILLED_BY_MONSTER or Event.BOMB_HIT_CHARACTER): # no we dont want this, return bad heuristic (posiitve)
            return inf
        charDistToExit = ass.euclidean_distance

    def simulateActionSimple(self, wrld, charAction):
        possibleWorlds = []
        # Assume there is only one monster 
        if (len(wrld.monsters.values()) == 1):
            monster = list(wrld.monsters.values())[0][0]
            actions = self.findPossibleMonsterActions(monster, wrld)
            numWorlds = len(actions)
            for act in actions:
                newWorld = SensedWorld.from_world(wrld)
                newCharacter = list(newWorld.characters.values())[0][0]
                newCharacter.move(charAction.value[0], charAction.value[1])
                newMonster = list(newWorld.monsters.values())[0][0]
                newMonster.move(act.value[0], act.value[1])
                newWorld, _ = newWorld.next()
                possibleWorlds.append((newWorld, 1/numWorlds))
        return possibleWorlds
    
    def findPossibleMonsterActions(self, monster, wrld): 
        """ finds all the possible actions that can be taken
        Output: the list of the possible action that the charater can take
        """
        # is this action possible? is there a wall next to me
        possibleActions = [] # stores the list of possible action
        for action in ActionSet: # Note ActionSet works for the Monster. The BOMB action doubles for the monster's 'no move' action.
            dx,dy = action.value[0], action.value[1]
            if (0 <= monster.x + dx < wrld.width() and 0 <= monster.y + dy < wrld.height()):
                if not wrld.wall_at(monster.x + dx, monster.y + dy):
                    possibleActions.append(action)
        return possibleActions

    def findPossibleActions(self, wrld): 
        """ finds all the possible actions that can be taken
        Output: the list of the possible action that the charater can take
        """
        # is this action possible? is there a wall next to me
        possibleActions = [] # stores the list of possible action
        for action in ActionSet:
            dx,dy = action.value[0], action.value[1]
            if action == ActionSet.BOMB:
                pass # implement this later
            else: 
                if (0 <= self.x + dx < wrld.width() and 0 <= self.y + dy < wrld.height()):
                    if (wrld.exit_at(self.x + dx, self.y + dy) or
                            wrld.empty_at(self.x + dx, self.y + dy)):
                        possibleActions.append(action)
        return possibleActions  

class AStar():

    def a_star(wrld: World, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates the path using A Star
        :param wrld [World]                  The map data.
        :start: tuple[int, int]                 The starting position (grid coord)
        :goal: tuple[int, int]                 The goal position (grid coord)
        :return        list[tuple[int, int]]    The path i.e. list of points (grid coord)
        """

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
            for next in AStar.getNeighborsOfEight(current, wrld): # get the list of neighbors 
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

        path = [] # the optimized path 
        current = goal # go back wards
        while True:
            path.insert(0, current) # add the path to the list
            current = came_from[current] # set the curent to the node current came from
            if(current == start): # true if we reach the start
                break
        
        # return the path
        return path
        
    def getNeighborsOfEight(cell: tuple[int,int], wrld: World):
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
                        wrld.empty_at(x + dx, y + dy)):
                            # Yes
                            neighbors.append((dx, dy))
        # All done
        return [(x+dx,y+dy) for (dx,dy) in neighbors]
        
    def heuristic(goal, next):
        goal_dist = AStar.euclidean_distance(goal, next)
        return goal_dist
                
    def euclidean_distance(cell1, cell2):
        return math.dist(cell1, cell2)

    

    

