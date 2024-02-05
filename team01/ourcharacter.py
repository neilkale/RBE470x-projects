# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

from world import World
import math
from priority_queue import PriorityQueue

class OurCharacter(CharacterEntity):

    def do(self, wrld):
        """Pick an action for the character"""
        # Get list of safe moves
        ass = AStar()
        path = ass.a_star(wrld, (self.x, self.y))
        nextcell = path[0]
        dx,dy = nextcell[0] - self.x, nextcell[1] - self.y
        self.move(dx,dy)
    
class AStar():

    def a_star(self, wrld: World, start: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates the path using A Star
        :param mapdata [World]                  The map data.
        :start: tuple[int, int]                 The starting position (grid coord)
        :start: tuple[int, int]                 The goal position (grid coord)
        :return        list[tuple[int, int]]    The path i.e. list of points (grid coord)
        """

        frontier = PriorityQueue() # frontier 
        frontier.put(start,0) # adding the start to the frontier 
        came_from = {} # list of linked list of the path to each node
        cost_so_far = {} # cost to each node
        came_from[start] = None # setting the came_from to None for start
        cost_so_far[start] = 0 # setting the cost of the start to 0
        goal = wrld.exitcell

        # keep looping until no more nodes in frontier
        while not frontier.empty():
            current = frontier.get() # get the first node
            if current == goal: # reached to the goal
                break 
            for next in self.getNeighborsOfFour(current, wrld): # get the list of neighbors 
                # calculate the new cost
                new_cost = cost_so_far[current] + 1 
                # true if the node has not been visited or if the next node costs less 
                if not (next in cost_so_far) or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost # set the cost 
                    priority =  new_cost + AStar.euclidean_distance(goal, next) # calculate the priority
                    frontier.put(next, priority) # add the node to the priority queue based on the cost 
                    came_from[next] = current # set the path of the node
        
        print("WAHHHHHHHHHHHHHHHHHHHHHHHHH!!!!!!!!!!!!!!!!!!!! ASTARRR BLINK") # self explanatory 
        
        path = [] # the optimized path 
        current = goal # go back wards
        # print("GOALL", goal)
        print("Came From", came_from)
        while True:
            path.insert(0, current) # add the path to the list
            print("path", path)
            current = came_from[current] # set the curent to the node current came from
            if(current == start): # true if we reach the start
                break
        
        # return the path
        return path
        
    def getNeighborsOfFour(self, cell: tuple[int,int], wrld: World):
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
                        # Is this cell safe and not a diagonal / non-move?
                        if ((dx==0) != (dy==0) and
                        (wrld.exit_at(x + dx, y + dy) or
                        wrld.empty_at(x + dx, y + dy))):
                            # Yes
                            neighbors.append((dx, dy))
        # All done
        return [(x+dx,y+dy) for (dx,dy) in neighbors]

    def euclidean_distance(cell1, cell2):
        return math.dist(cell1, cell2)