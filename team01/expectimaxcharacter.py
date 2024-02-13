# This is necessary to find the main code
from itertools import islice
import sys

from events import Event

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

MAX_DEPTH = 0
VERBOSE = False
WIN_HEURISTIC = 500
LOSE_HEURISTIC = -500
HEURISTIC_EVAL_BOUND = 80

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

    @staticmethod
    def from_tuple(input_tuple):
        for action in ActionSet:
            if action.value == input_tuple:
                return action
        return None

class ExpectimaxCharacter(CharacterEntity):
    
    def do(self, wrld):
        """
        Main function called every turn. This is where the expectimax computation starts.
        - wrld: The current state of the world.
        Output: The best action determined by expectimax (e.g., move direction).
        """
        # TODO: If the character is closer to the end state than the monster, use A* move.
        pathToGoal = AStar.a_star(wrld, (self.x, self.y), wrld.exitcell)
        useAStar = True
        for i in range(len(list(wrld.monsters.values()))):
            monster = list(wrld.monsters.values())[i][0]
            monsterPathToGoal = AStar.a_star(wrld, (monster.x, monster.y), wrld.exitcell)
            if (len(pathToGoal)+1 >= len(monsterPathToGoal)):
                useAStar = False
        if len(list(wrld.monsters.values())) == 0 or useAStar:
            self.move(pathToGoal[0][0] - self.x, pathToGoal[0][1] - self.y)
        else:
            # Else, start the expectimax search from the current state
            if VERBOSE: print("--------------- ACTION ---------------------")
            if VERBOSE: print("NUM MONSTERS:", len(wrld.monsters.values()))
            action, _ = self.expectimax(wrld, depth=0)
            # Execute the action
            self.takeAction(action)

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
    

    def expectimax(self, wrld, depth):
        """
        The recursive expectimax function that evaluates the possible outcomes and chooses the best action.
        - wrld: The current state of the world.
        - depth: The current depth in the search tree.
        Output: The best score and corresponding action at this level of the tree.
        """
        bestAction, bestScore = None, np.Inf

        actions = self.findPossibleActions(wrld) #NOTE: The bug is this function does not return all the possible actions. Some are missing. Find out why.
        if VERBOSE: wrld.printit()
        for act in actions:
            if VERBOSE: print("Act: ", act)
            score = self.evaluateChanceNode(wrld, act, depth)
            if VERBOSE: print("Score", " ", score)
            if VERBOSE: print()
            if (score < bestScore):
                bestAction, bestScore = act, score

        if VERBOSE: print("BEST: ", bestAction, "SCORE: ", bestScore)
        return bestAction, bestScore

    def evaluateChanceNode(self, wrld, action, depth):
        if (len(list(wrld.monsters.values())) == 1):
            possibleWorlds = self.simulateAction(wrld, action)
        if (len(list(wrld.monsters.values())) == 2):
            possibleWorlds = []
            possibleWorlds1 = self.simulateAction(wrld, action)
            for (possibleWorld1, probability1) in possibleWorlds1:
                if (len(list(possibleWorld1.characters.values())) == 0 or len(list(possibleWorld1.characters.values())[0]) == 0):
                    possibleWorlds += [(possibleWorld1, probability1)]
                else:
                    possibleWorlds2 = self.simulateActionTwo(possibleWorld1, action)
                    possibleWorlds += [(possibleWorld2, probability1*probability2) for (possibleWorld2, probability2) in possibleWorlds2]
        if VERBOSE: print("Num Worlds:", len(possibleWorlds))
        # possible_worlds is [(World, probability), (World, probability), ...]
        score = 0

        for (possibleWorld, probability) in possibleWorlds:
            # print(action)
            if VERBOSE: print("Events:", possibleWorld.events)
            # possibleWorld.printit()
            heuristic = self.heuristic(possibleWorld)
            if VERBOSE: print(heuristic, " ", probability)
            if (depth >= MAX_DEPTH or abs(heuristic) >= HEURISTIC_EVAL_BOUND):
                if VERBOSE: print("HEURISTIC USED")
                score += heuristic*probability
            else:
                if VERBOSE: print("EXPECTIMAX USED")
                _, possibleScore = self.expectimax(possibleWorld, depth+1)
                score += possibleScore * probability

        return score
    
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
    
    def simulateAction(self, wrld, charAction):
        possibleWorlds = []
        # Assume there is only one monster 
        monster = list(wrld.monsters.values())[0][0]
        actions = self.findPossibleMonsterActions(monster, wrld)
        numWorlds = len(actions)
        # Assume every future world is equally likely (stupid idiot monster)
        if (monster.name == 'stupid'):
            for act in actions:
                newWorld = SensedWorld.from_world(wrld)
                newCharacter = list(newWorld.characters.values())[0][0]
                newCharacter.move(charAction.value[0], charAction.value[1])
                newMonster = list(newWorld.monsters.values())[0][0]
                newMonster.move(act.value[0], act.value[1])
                newWorld, _ = newWorld.next()
                possibleWorlds.append((newWorld, 1/numWorlds))
        elif (monster.name == 'selfpreserving'):

            # If kill action not possible, and same action as last time is possible, take that action
            lastAction = ActionSet.from_tuple((monster.dx, monster.dy))
            if (len(AStar.a_star(wrld, (self.x+charAction.value[0], self.y+charAction.value[1]), (monster.x, monster.y))) > 2 and lastAction != ActionSet.BOMB and lastAction in actions):
                if VERBOSE: print ("OLD ACTION TO REPEAT:", lastAction)
                actions = [lastAction]
            numWorlds = len(actions)

            # Iterate through possible actions
            for act in actions:
                newWorld = SensedWorld.from_world(wrld)
                newCharacter = list(newWorld.characters.values())[0][0]
                newCharacter.move(charAction.value[0], charAction.value[1])
                newMonster = list(newWorld.monsters.values())[0][0]
                newMonster.move(act.value[0], act.value[1])
                newWorld, _ = newWorld.next()

                # If kill action possible, take that with probability 1.0
                events = [event.tpe for event in newWorld.events]
                if (Event.CHARACTER_KILLED_BY_MONSTER in events):
                    possibleWorlds = [(newWorld, 1)]
                    break
                else:
                    possibleWorlds.append((newWorld, 1/numWorlds))
        elif (monster.name == 'aggressive'):
            # If kill action not possible, and same action as last time is possible, take that action
            lastAction = ActionSet.from_tuple((monster.dx, monster.dy))
            if (len(AStar.a_star(wrld, (self.x+charAction.value[0], self.y+charAction.value[1]), (monster.x, monster.y))) > 2 and lastAction != ActionSet.BOMB and lastAction in actions):
                if VERBOSE: print ("OLD ACTION TO REPEAT:", lastAction)
                actions = [lastAction]
            numWorlds = len(actions)

            # Iterate through possible actions
            for act in actions:
                newWorld = SensedWorld.from_world(wrld)
                newCharacter = list(newWorld.characters.values())[0][0]
                newCharacter.move(charAction.value[0], charAction.value[1])
                newMonster = list(newWorld.monsters.values())[0][0]
                newMonster.move(act.value[0], act.value[1])
                newWorld, _ = newWorld.next()

                # If kill action possible, take that with probability 1.0
                events = [event.tpe for event in newWorld.events]
                if (Event.CHARACTER_KILLED_BY_MONSTER in events):
                    possibleWorlds = [(newWorld, 1)]
                    break
                # # If can chase the agent, take that with probability 1.0
                # elif AStar.a_star(newWorld, (newCharacter.x, newCharacter.y), (newMonster.x, newMonster.y)):
                #     possibleWorlds = [(newWorld, 1)]
                else:
                    possibleWorlds.append((newWorld, 1/numWorlds))

        return possibleWorlds

    def simulateActionTwo(self, wrld, charAction):
        possibleWorlds = []
        if len(list(wrld.monsters.values())) < 2:
            return [(wrld, 1)]
        # Assume there is only one monster 
        monster = list(wrld.monsters.values())[1][0]
        actions = self.findPossibleMonsterActions(monster, wrld)
        numWorlds = len(actions)
        # Assume every future world is equally likely (stupid idiot monster)
        if (monster.name == 'stupid'):
            for act in actions:
                newWorld = SensedWorld.from_world(wrld)
                newCharacter = list(newWorld.characters.values())[0][0]
                newMonster = list(newWorld.monsters.values())[0][0]
                newMonster.move(act.value[0], act.value[1])
                newWorld, _ = newWorld.next()
                possibleWorlds.append((newWorld, 1/numWorlds))
        elif (monster.name == 'selfpreserving'):

            # If kill action not possible, and same action as last time is possible, take that action
            lastAction = ActionSet.from_tuple((monster.dx, monster.dy))
            if (len(AStar.a_star(wrld, (self.x+charAction.value[0], self.y+charAction.value[1]), (monster.x, monster.y))) > 2 and lastAction != ActionSet.BOMB and lastAction in actions):
                if VERBOSE: print ("OLD ACTION TO REPEAT:", lastAction)
                actions = [lastAction]
            numWorlds = len(actions)

            # Iterate through possible actions
            for act in actions:
                newWorld = SensedWorld.from_world(wrld)
                newCharacter = list(newWorld.characters.values())[0][0]
                newMonster = list(newWorld.monsters.values())[0][0]
                newMonster.move(act.value[0], act.value[1])
                newWorld, _ = newWorld.next()

                # If kill action possible, take that with probability 1.0
                events = [event.tpe for event in newWorld.events]
                if (Event.CHARACTER_KILLED_BY_MONSTER in events):
                    possibleWorlds = [(newWorld, 1)]
                    break
                else:
                    possibleWorlds.append((newWorld, 1/numWorlds))
        elif (monster.name == 'aggressive'):
            # If kill action not possible, and same action as last time is possible, take that action
            lastAction = ActionSet.from_tuple((monster.dx, monster.dy))
            if (len(AStar.a_star(wrld, (self.x+charAction.value[0], self.y+charAction.value[1]), (monster.x, monster.y))) > 4 and lastAction != ActionSet.BOMB and lastAction in actions):
                if VERBOSE: print ("OLD ACTION TO REPEAT:", lastAction)
                actions = [lastAction]
            numWorlds = len(actions)

            # Iterate through possible actions
            for act in actions:
                newWorld = SensedWorld.from_world(wrld)
                newCharacter = list(newWorld.characters.values())[0][0]
                newMonster = list(newWorld.monsters.values())[0][0]
                newMonster.move(act.value[0], act.value[1])
                newWorld, _ = newWorld.next()

                # If kill action possible, take that with probability 1.0
                events = [event.tpe for event in newWorld.events]
                if (Event.CHARACTER_KILLED_BY_MONSTER in events):
                    possibleWorlds = [(newWorld, 1)]
                    break
                # # If can chase the agent, take that with probability 1.0
                # elif AStar.a_star(newWorld, (newCharacter.x, newCharacter.y), (newMonster.x, newMonster.y)):
                #     possibleWorlds = [(newWorld, 1)]
                else:
                    possibleWorlds.append((newWorld, 1/numWorlds))

        return possibleWorlds

    # CODE FOR GENERATING WORLDS MONTE-CARLO STYLE... NOT WORKING AT THE MOMENT
    # worlds = {}
    #         for i in range(100):
    #             newWorld = SensedWorld.from_world(wrld)
    #             # Make character move
    #             newCharacter = list(newWorld.characters.values())[0][0]
    #             newCharacter.move(charAction.value[0], charAction.value[1])
    #             # Make monster move
    #             for i in range(len(list(newWorld.monsters.values()))):
    #                 list(newWorld.monsters.values())[i][0].do(newWorld)
    #             newWorld, _ = newWorld.next()

    #             worldExists = False
    #             for oldWorld in worlds.keys():
    #                 if (self.isSameWorld(oldWorld, newWorld)):
    #                     worlds[oldWorld] += 1
    #                     worldExists = True
    #                     break
    #             if not worldExists:
    #                 worlds[newWorld] = 1
    #         possibleWorlds = [(key, worlds[key]/100) for key in worlds.keys()]
    
    # def isSameWorld(self, wrld1, wrld2):
    #     # Character positions
    #     if len(list(wrld1.characters.values())) != len(list(wrld2.characters.values())): return False
    #     if len(list(wrld1.characters.values())) > 0 and len(list(wrld1.characters.values())[0]) > 0:
    #         char1, char2 = list(wrld1.characters.values())[0][0], list(wrld2.characters.values())[0][0]
    #         if char1.x != char2.x or char1.y != char2.y: return False
    #     # Monster positions
    #     if len(list(wrld1.monsters.values())) != len(list(wrld2.monsters.values())): return False
    #     for i in range(len(list(wrld1.monsters.values()))):
    #         monster1, monster2 = list(wrld1.monsters.values())[i][0], list(wrld2.monsters.values())[i][0]
    #         if monster1.x != monster2.x or monster1.y != monster2.y: return False 
    #     return True
    
    def heuristic(self, wrld):
        events = [event.tpe for event in wrld.events]
        if (Event.CHARACTER_FOUND_EXIT in events): # If exit found, return really negative (good) heuristic
            return LOSE_HEURISTIC
        if (Event.CHARACTER_KILLED_BY_MONSTER in events or Event.BOMB_HIT_CHARACTER in events): # If character kiled, return really positive (bad) heuristic.
            return WIN_HEURISTIC
        else:
            character = list(wrld.characters.values())[0][0]
            U1 = len(AStar.a_star(wrld, (character.x, character.y), wrld.exitcell)) # distance to exit cell
            U2 = 0 # Penalty for being too close to a chasing monster
            for i in range(len(list(wrld.monsters.values()))):
                monster = list(wrld.monsters.values())[i][0]
                distToMonster = len(AStar.a_star(wrld, (character.x, character.y), (monster.x, monster.y)))
                if monster.name == "selfpreserving" and distToMonster <= 2:
                    U2 += 100 #*(2-distToMonster)
                elif monster.name == "stupid" and distToMonster <= 1:
                    U2 += 50 #*(1-distToMonster)
                elif monster.name == "aggressive" and distToMonster <= 2:
                    U2 += 100 #*(3-distToMonster)
            # for monster in list(wrld.monsters.values())[0]:
            #     U2 += 10 / len(AStar.a_star(wrld, (character.x, character.y), (monster.x,monster.y))) ** 2 # distance to each monster
            #     # U2 += 10 / math.dist((character.x, character.y), (monster.x,monster.y))
            return U1 + U2
        
    def manhattanDistance(self, a, b):
        return abs(a[0]-b[0])+abs(a[1]-b[1])
    
    def isEightNeighbor(self, a, b):
        return abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1

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
                    if not wrld.wall_at(self.x + dx, self.y + dy):
                        possibleActions.append(action)
        return possibleActions
                    
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