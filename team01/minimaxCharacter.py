# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

from events import Event


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

    @staticmethod
    def from_tuple(input_tuple):
        for action in ActionSet:
            if action.value == input_tuple:
                return action
        return None

inf = 100.0 
MAX_DEPTH = 1

VERBOSE = False

class MinimaxCharacter(CharacterEntity):

    def do(self, wrld):
        # Your code here
        """
        Main function called every turn. This is where the minimax computation starts.
        - wrld: The current state of the world.
        Output: The best action determined by minimax (e.g., move direction).
        """
        # Start the expectimax search from the current state
        if VERBOSE: print("**************here in do*************")
        print("***************------------------------------***************")
        action, _ = self.minimax(wrld, depth=0)

        if VERBOSE: print("Action: ", action)

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
            self.move(action.value[0],action.value[1])

    def minimax(self, wrld, depth):
        """
        The recursive minimax function that evaluates the possible outcomes and chooses the best action.
        - wrld: The current state of the world.
        - depth: The current depth in the search tree.
        Output: The best score and corresponding action at this level of the tree.
        """
        # Get list of legal actions (moves)
        possibleActions = self.findPossibleActions(wrld)
        if VERBOSE: print("Possible Actions: ", possibleActions)

        bestScore = inf
        bestAction = None

        # Start the search 
        for action in possibleActions:
            allPossibleWorlds = self.simulateActionSimple(wrld,action)
            if VERBOSE: print("Next world (Action: ", action, "): ", allPossibleWorlds)
            score = self.minValue(allPossibleWorlds, depth)

            if VERBOSE: print ("Score: ", score, "Action: ", action)

            if score <= bestScore:
                bestScore = score
                bestAction = action

        return bestAction, bestScore
    
    def maxValue(self, wrld, depth):
        """
        Max-value function of minimax
        - wrld: The current state of the world.
        Output: The maximum utility value
        """
        if VERBOSE: print("\nMAX NODE\n", "Depth: ", depth)
        if self.terminalState(wrld,depth):    
            if VERBOSE: print("Terminal state Max node, Heuristic: ", self.heuristic(wrld))
            return self.heuristic(wrld)
        
        v = -inf
        possibleActions = self.findPossibleActions(wrld)
        if VERBOSE: print("possibleActions", possibleActions)

        for action in possibleActions:
            allPossibleWorlds = self.simulateActionSimple(wrld,action)
            v = max(v,self.minValue(allPossibleWorlds, depth + 1))
            
            
        if VERBOSE: print ("Max Value: ", v)

        return v
    

    
    def minValue(self, wrld, depth):
        """
        Min-value function of minimax
        - wrld: The current state of the world.
        Output: The minimum utility value
        """
        if VERBOSE: print("\nMIN NODE\n", "Depth: ", depth)
        if self.terminalState(wrld, depth):   
            if VERBOSE: print("Terminal State min node, Heuristic: ", self.heuristic(wrld)) 
            return self.heuristic(wrld)
        
        v = inf

        possibleActions = self.findPossibleActions(wrld)
        if VERBOSE: print("Possible Action", possibleActions)

        for action in possibleActions:
            allPossibleWorlds = self.simulateActionSimple(wrld,action)
            # if VERBOSE: print("Depth", depth, "Next World", next_wrld)
            v = min(v,self.maxValue(allPossibleWorlds, depth+1))
                
        if VERBOSE: print ("Min Value: ", v)
        return v
        
    def terminalState(self, wrld, depth):
        events = [event.tpe for event in wrld.events]
        if (Event.CHARACTER_FOUND_EXIT in events or Event.BOMB_HIT_CHARACTER in events or Event.CHARACTER_KILLED_BY_MONSTER in events
            or depth > MAX_DEPTH):
            return True
        return False


    def heuristic(self, wrld):
        events = [event.tpe for event in wrld.events]
        if (Event.CHARACTER_FOUND_EXIT in events): # yay we are wining, return good heuristics (negative for us)
            if VERBOSE: print("Character found exit ayay")
            return -inf
        elif (Event.CHARACTER_KILLED_BY_MONSTER in events or Event.BOMB_HIT_CHARACTER in events): # no we dont want this, return bad heuristic (posiitve)
            if VERBOSE: print("Character died :()")
            return inf
        character = list(wrld.characters.values())[0][0]
        monster = list(wrld.monsters.values())[0][0]
        character = list(wrld.characters.values())[0][0]
        charDistToExit = len(AStar.a_star(wrld, (character.x, character.y), wrld.exitcell)) # distance to exit cell
        charDistToMonster = 0 # Penalty for being too close to a chasing monster
        charDistToMonster = AStar.euclidean_distance((character.x,character.y), (monster.x,monster.y))
        
        # for i in range(len(list(wrld.monsters.values()))):
        #     monster = list(wrld.monsters.values())[i][0]
        #     distToMonster = len(AStar.a_star(wrld, (character.x, character.y), (monster.x, monster.y)))
        #     if monster.name == "selfpreserving" and distToMonster <= 1:
        #         charDistToMonster += 100
        #     elif monster.name == "aggressive" and distToMonster <= 2:
        #         charDistToMonster += 100
        #         if VERBOSE: print("Dist to Monster: ", charDistToMonster)
        # if VERBOSE: print("Dist to Exit: ", charDistToExit)
        return charDistToExit - charDistToMonster
    
    def simulateActionSimple(self, wrld, charAction):
        newWrld = SensedWorld.from_world(wrld)

        # move the character
        character = list(newWrld.characters.values())[0][0]
        character.move(charAction.value[0], charAction.value[1])

        # move the monster 
        if (len(wrld.monsters.values()) == 1):
            monster = list(wrld.monsters.values())[0][0]
            monsterActions = self.findPossibleMonsterActions(monster, wrld)
            for act in monsterActions:
                newMonster = list(newWrld.monsters.values())[0][0]
                newMonster.move(act.value[0], act.value[1])
            
            newWrld, _ = newWrld.next()
            return newWrld
        return wrld


    # def simulateActionSimple(self, wrld, charAction):
    #     newWorld = SensedWorld.from_world(wrld)
    #     newCharacter = list(newWorld.characters.values())[0][0]
    #     newCharacter.move(charAction.value[0], charAction.value[1])
    #     # Assume there is only one monster 
    #     if (len(wrld.monsters.values()) == 1):
    #         monster = list(wrld.monsters.values())[0][0]
    #         actions = self.findPossibleMonsterActions(monster, wrld)

    #         if (monster.name == 'selfpreserving'):
    #             # If kill action not possible, and same action as last time is possible, take that action
    #             lastAction = ActionSet.from_tuple((monster.dx, monster.dy))
    #             if (self.manhattanDistance((monster.x, monster.y), (self.x+charAction.value[0], self.y+charAction.value[1])) > 2 and lastAction != ActionSet.BOMB and lastAction in actions):
    #                 if VERBOSE: print ("OLD ACTION TO REPEAT:", lastAction)
    #                 actions = [lastAction]
    #             # numWorlds = len(actions)

    #             # Iterate through possible actions
    #             for act in actions:
    #                 newMonster = list(newWorld.monsters.values())[0][0]
    #                 newMonster.move(act.value[0], act.value[1])
    #                 newWorld, _ = newWorld.next()

    #                 # If kill action possible, take that with probability 1.0
    #                 events = [event.tpe for event in newWorld.events]
    #                 if (Event.CHARACTER_KILLED_BY_MONSTER in events):
    #                     possibleWorlds = [(newWorld, 1)]
    #                     break
    #                 else:
    #                     possibleWorlds.append((newWorld))
    #     return possibleWorlds


    # def simulateActionSimple(self, wrld, charAction):
    #     possibleWorlds = []
    #     # Assume there is only one monster 
    #     if (len(wrld.monsters.values()) == 1):
    #         monster = list(wrld.monsters.values())[0][0]
    #         monsterActions = self.findPossibleMonsterActions(monster, wrld)
    #         numWorlds = len(monsterActions)
    #         for act in monsterActions:
    #             newWorld = SensedWorld.from_world(wrld)
    #             newCharacter = list(newWorld.characters.values())[0][0]
    #             newCharacter.move(charAction.value[0], charAction.value[1])
    #             newMonster = list(newWorld.monsters.values())[0][0]
    #             newMonster.move(act.value[0], act.value[1])
    #             newWorld, _ = newWorld.next()
    #             possibleWorlds.append((newWorld))
    #     return possibleWorlds
    
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
    
    def manhattanDistance(self, a, b):
        return abs(a[0]-b[0])+abs(a[1]-b[1])

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