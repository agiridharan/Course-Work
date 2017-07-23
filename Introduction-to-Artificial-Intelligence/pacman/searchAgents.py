# searchAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import sys


class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################


class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """

        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)


class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        self.startingGameState = startingGameState
        self.heuristicInfo = {}

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        x, y = self.startingPosition
        return x, y, False, False, False, False

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        x, y, c1, c2, c3, c4 = state
        if c1 and c2 and c3 and c4:
            isGoal = True
        else:
            isGoal = False

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        "*** YOUR CODE HERE ***"
        successors = []
        x, y, c0, c1, c2, c3 = state
        corners = zip(self.corners, (c0, c1, c2, c3))

        for action in (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST):
            dx, dy = Actions.directionToVector(action)
            nextX, nextY = int(x+dx), int(y+dy)

            if not self.walls[nextX][nextY]:
                # check if x, y is in a corner
                c = []
                i = 0
                while i < 4:
                    if corners[i][0][0] == nextX and corners[i][0][1] == nextY:
                        c.append(True)
                    else:
                        c.append(corners[i][1])
                    i += 1

                successors.append(((nextX, nextY, c[0], c[1], c[2], c[3]), action, 1))

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """

        if actions == None:
            return 999999
        return len(actions)


def test_cornersHeuristic(state, problem):
    corners = problem.corners
    x, y, c0, c1, c2, c3 = state
    worst = 0

    if (x, y) not in problem.heuristicInfo:
        subdictionary = {}
        for c, v in zip(corners, (c0, c1, c2, c3)):
            if not v:
                #dist = mazeDistance((x, y), c, problem.startingGameState)
                dist = abs(x-c[0]) + abs(y-c[1])
                subdictionary[c] = dist
                if dist > worst:
                    worst = dist
        problem.heuristicInfo[(x, y)] = subdictionary
    else:
        for c, v in zip(corners, (c0, c1, c2, c3)):
            if not v:
                if c not in problem.heuristicInfo[(x, y)]:
                    #dist = mazeDistance((x, y), c, problem.startingGameState)
                    dist = abs(x - c[0]) + abs(y - c[1])
                    problem.heuristicInfo[(x, y)][c] = dist
                else:
                    dist = problem.heuristicInfo[(x, y)][c]
                if dist > worst:
                    worst = dist

    return worst


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    # get distance to closest non-visited corner
    from util import PriorityQueue

    # initialize heuristic info
    # This returns a dictionary of dictionaries.
    # key1 = node name, value1 = dictionary of distances for that node to other nodes
    # key2 = node name, value2 = distance to that node
    #
    # example of usage: find distance from 'c0' to 'c3'
    # distance = problem.heuristicInfo['c0']['c1']
    if not problem.heuristicInfo:
        problem.heuristicInfo = buildCornersDictionary(problem)

    # packet needed variables
    x, y, v0, v1, v2, v3 = state
    visited = (v0, v1, v2, v3)
    names = ('c0', 'c1', 'c2', 'c3')

    # explored stores the name of a corner as a key
    explored = {}
    pq = PriorityQueue()

    # Calculate the cost to go to each unvisited corner,
    # packet the info into a state and place into priority queue
    #
    # NOTE: this segment checks the heuristicInfo['hist'] to see if the current
    # coordinates have been visited already. If visited, it checks if the
    # unvisited corners distance have been placed into the 'hist' dictionary.
    # It will then pull the entry form 'hist' if there is one, calculate the distance
    # to the unvisted corner and insert that into the 'hist' dictionary
    if (x, y) in problem.heuristicInfo['hist']:
        # (x, y) has been visited before
        index = 0
        while index < 4:
            # check for the corners we care about
            if not visited[index]:
                if names[index] in problem.heuristicInfo['hist'][(x, y)]:
                    # The cost to this corner has already been calculated
                    start_cost = problem.heuristicInfo['hist'][(x, y)][names[index]]
                else:
                    # The cost to this corner has not been calculated yet.
                    # Calculate and then insert cost
                    start_cost = mazeDistance((x, y), problem.corners[index], problem.startingGameState)
                    problem.heuristicInfo['hist'][(x, y)][names[index]] = start_cost

                # make an updated visited list and packet everything into a state
                start_visited_corners = list(visited)
                start_visited_corners[index] = True
                start_state = x, y, start_visited_corners[0], start_visited_corners[1], start_visited_corners[2], start_visited_corners[3]

                # check for winning condition
                if problem.isGoalState(start_state):
                    return start_cost

                # push onto priority queue
                pq.push((start_state, names[index], start_cost), start_cost)

            index += 1
    else:
        # this (x, y) location has NOT been visited yet
        # create a new dictionary for the new path entries
        pathDict = {}
        index = 0
        while index < 4:
            # check for any corners we care about
            if not visited[index]:
                # make an updated visited list and packet everything into a state
                start_cost = mazeDistance((x, y), problem.corners[index], problem.startingGameState)
                start_visited_corners = list(visited)
                start_visited_corners[index] = True
                start_state = x, y, start_visited_corners[0], start_visited_corners[1], start_visited_corners[2], start_visited_corners[3]

                # check for winning condition
                if problem.isGoalState(start_state):
                    return start_cost

                # insert into dictionary
                pathDict[names[index]] = start_cost

                # push onto priority queue
                pq.push((start_state, names[index], start_cost), start_cost)

            index += 1
        # add pathDict to the 'hist' dict with key (x, y)
        problem.heuristicInfo['hist'][(x, y)] = pathDict

    # run a dijkstra's search through the remaining unvsited corners and return the shortest pathlengh to eat all food
    while not pq.isEmpty():
        curr_state, curr_name, curr_cost = pq.pop()

        # check for goal
        if problem.isGoalState(curr_state):
            return curr_cost

        explored[state] = True

        x, y, c0, c1, c2, c3 = curr_state
        curr_visited_corners = (c0, c1, c2, c3)

        index = 0
        while index < 4:
            if not curr_visited_corners[index]:
                next_visited_corners = list(curr_visited_corners)
                next_visited_corners[index] = True
                next_name = names[index]
                next_cost = problem.heuristicInfo[curr_name][next_name] + curr_cost
                next_state = x, y, next_visited_corners[0], next_visited_corners[1], next_visited_corners[2], next_visited_corners[3]

                # check if state as been explored
                if next_state in explored:
                    continue

                # put next state onto queue
                pq.push((next_state, next_name, next_cost), next_cost)

            index += 1

    return 0


def buildCornersDictionary(problem):
    c0, c1, c2, c3 = problem.corners
    gamestate = problem.startingGameState
    pathways = {}

    # initialize pathways
    pathways['c0'] = {}
    pathways['c1'] = {}
    pathways['c2'] = {}
    pathways['c3'] = {}

    # find paths from c0 -> c1, c2, c3
    pathways['c0']['c1'] = mazeDistance(c0, c1, gamestate)
    pathways['c0']['c2'] = mazeDistance(c0, c2, gamestate)
    pathways['c0']['c3'] = mazeDistance(c0, c3, gamestate)

    # copy path cost back for other corners to c0
    pathways['c1']['c0'] = pathways['c0']['c1']
    pathways['c2']['c0'] = pathways['c0']['c2']
    pathways['c3']['c0'] = pathways['c0']['c3']

    # find paths from c1 -> c2, c3
    pathways['c1']['c2'] = mazeDistance(c1, c2, gamestate)
    pathways['c1']['c3'] = mazeDistance(c1, c3, gamestate)

    # copy paths back for other corners to c1
    pathways['c2']['c1'] = pathways['c1']['c2']
    pathways['c3']['c1'] = pathways['c1']['c3']

    # find path from c2 -> c3
    pathways['c2']['c3'] = mazeDistance(c2, c3, gamestate)

    # copy path back from other corner to c2
    pathways['c3']['c2'] = pathways['c2']['c3']

    pathways['hist'] = {}

    return pathways


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """

    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    position, foodGrid = state
    food_list = foodGrid.asList()
    a = {}
    longest = -99999999
    pos = 0
    if not food_list:
        return 0
    for food in food_list:
        a[food] = mazeDistance(position, food, problem.startingGameState)
        if (a[food] > longest):
            longest = a[food]
            pos = food
    return a[pos]


class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        #check that there is still food on the map
        if (food.count() != 0):
            return search.breadthFirstSearch(problem)
        else:
            return []


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        if (self.food[x][y]):
            return True
        return False


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))


def GetClosestDot(start, walls, foodGrid):
    from util import Queue

    x, y = start
    moveVector = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    assert not walls[x][y], 'point is wall: ' + str(start)

    q = Queue()
    explored = {}

    q.push((start, 0))

    while not q.isEmpty():
        currPos, currPathLen = q.pop()
        x, y = currPos
        explored[(x, y)] = currPathLen

        if foodGrid[x][y]:
            return (x, y), currPathLen

        for nextX, nextY in moveVector:
            nextX += x
            nextY += y
            if walls[nextX][nextY] or (nextX, nextY) in explored:
                continue
            nextPathLen = currPathLen + 1
            q.push(((nextX, nextY), nextPathLen))

    return (-1, -1), 999999


""" Mapping Tools """

# Returns Pair of Dictionaries : 1. (node_id -> node), 2. (food_coord -> node_id)
def GetFoodNodes(foodGrid):
    nodes = {}
    foodmap = {}
    node_id = 0

    for f in foodGrid.asList():
        node = Node(node_id, [f], {}, 1)
        nodes[node_id] = node
        foodmap[f] = node_id
        node_id += 1
    return nodes, foodmap


# Returns Dictionary : node_id -> node
def ConnectNodes(nodes, food_map, foodGrid, walls):

    for n in nodes:
        node = nodes[n]
        food_pos = node.getFoodPositions()
        node_food_pos = food_pos[0]
        conn_food = GetAllConnectingDots(food_pos[0], walls, foodGrid)

        for conn_food_pos, weight in conn_food:
            conn_node_id = food_map[conn_food_pos]
            conn_node = nodes[conn_node_id]

            # check if this node already contains an edge to the connected node
            if conn_node in node.getEdges():
                continue

            # connect nodes with edges
            node.addEdge(conn_node, node_food_pos, weight, conn_food_pos)
            conn_node.addEdge(node, conn_food_pos, weight, node_food_pos)

            nodes[node.id] = node
            nodes[conn_node.id] = conn_node

    return nodes


# Returns Pair of Dictionary : 1. node_id -> node, 2. food(x, y) -> node_id
def CompressNodeMap(nodes, foodmap):
    from util import Queue
    q = Queue()
    for n in nodes:
        q.push(nodes[n])
    new_nodes = dict(nodes)
    new_foodmap = dict(foodmap)

    while not q.isEmpty():
        node = q.pop()

        if node.getEdgeCount() == 2:
            for neighbor_id in node.getNeighborsId():
                neighbor_node = nodes[neighbor_id]
                edge_start, neighbor_weight, edge_end, edge_node = node.getEdges()[neighbor_id]

                if neighbor_node.getEdgeCount() <= 2 and neighbor_weight == 1:
                    node.mergeNode(neighbor_node)
                    new_nodes[node.id] = node
                    new_nodes.pop(neighbor_id, None)

                    for food in neighbor_node.getFoodPositions():
                        new_foodmap[food] = node.id

        elif node.getEdgeCount() == 1:
            other_node = nodes[node.getNeighborsId()[0]]
            other_node.mergeNode(node)
            new_nodes[other_node.id] = other_node
            new_nodes.pop(node.id, None)
            q.push(other_node)

            for food in node.getFoodPositions():
                new_foodmap[food] = other_node.id

    return new_nodes, new_foodmap


# Returns Pair of Dictionary : 1. node_id -> node, 2. food(x, y) -> node_id
def CreateFoodMap(start, walls, foodGrid):

    if foodGrid.count() == 0:
        return None

    # get singleton food nodes
    # Note: none of the nodes are connected
    singletons, food_map = GetFoodNodes(foodGrid)

    # connect nodes
    node_map = ConnectNodes(singletons, food_map, foodGrid, walls)

    # compress map
    node_map, food_map = CompressNodeMap(node_map, food_map)

    # return map
    return node_map, food_map


def GetAllConnectingDots(start, walls, foodGrid):
    from util import Queue

    x, y = start
    moveVector = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    assert not walls[x][y], 'point is wall: ' + str(start)

    food = []
    q = Queue()
    explored = {}

    q.push((start, 0))

    while not q.isEmpty():
        currPos, currPathLen = q.pop()
        x, y = currPos
        explored[(x, y)] = currPathLen

        if not (x == start[0] and y == start[1]) and foodGrid[x][y]:
            food.append((currPos, currPathLen))
            continue

        for nextX, nextY in moveVector:
            nextX += x
            nextY += y
            if walls[nextX][nextY] or (nextX, nextY) in explored:
                continue
            nextPathLen = currPathLen + 1
            q.push(((nextX, nextY), nextPathLen))

    return food


def overwrite_line(message):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(message)
    sys.stdout.flush()

