# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util, sys

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # imports
    from util import Stack

    # declare variables
    stack = Stack()
    explored = {}
    maxNodeId = 0
    start = problem.getStartState()
    stack.push((start, [], maxNodeId))

    # check if start state is goal state
    if problem.isGoalState(start):
        return []


    while not stack.isEmpty():

        # pop next node from stack
        currState, currPath, currId = stack.pop()

        # check if explored already
        if currState in explored:
            continue

        # evaluate node for goal
        if problem.isGoalState(currState):
            return currPath

        explored[currState] = True

        # get list of next moves
        nextMoves = problem.getSuccessors(currState)

        for nextNode in nextMoves:

            nextState, nextAction, nextCost = nextNode

            # check if nextState has already been visited
            if nextState not in explored:

                # Add action to path
                nextPath = list(currPath)
                nextPath.append(nextAction)

                # incroment node id
                maxNodeId += 1

                # Add to stack
                stack.push((nextState, nextPath, maxNodeId))

    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # imports
    from game import Directions
    from util import Queue

    # declare variables
    maxNodeId = 0
    explored = {}
    q = Queue()
    start = problem.getStartState()
    q.push((start, [], maxNodeId))
    explored[start] = True

    # check if initial state is goal state
    if problem.isGoalState(start):
        return []

    while not q.isEmpty():
        # pop current node
        currState, currPath, currId = q.pop()

        # evaluate node for goal
        if problem.isGoalState(currState):
            return currPath

        nextMoves = problem.getSuccessors(currState)

        for nextNode in nextMoves:
            nextState, nextAction, nextCost = nextNode

            if nextState not in explored:

                # make copy of path
                nextPath = list(currPath)
                nextPath.append(nextAction)

                # add to explored list
                explored[nextState] = True

                # incroment node id
                maxNodeId += 1

                # push onto queue
                q.push((nextState, nextPath, maxNodeId))

    return []


def uniformCostSearch(problem):

    # imports
    from util import PriorityQueue

    # declare variables
    explored = {}
    start = problem.getStartState()
    pq = PriorityQueue()
    pq.push((start, [], 0), 0)

    # check start state for goal
    if problem.isGoalState(start):
        return []

    # loop keep pulling from queue until empty or goal found
    while not pq.isEmpty():

        # get next node and check for goal
        currState, currPath, currCost = pq.pop()

        if currState in explored:
            continue

        if problem.isGoalState(currState):
            return currPath

        # get next moves
        explored[currState] = True
        nextMoves = problem.getSuccessors(currState)

        # compile next move
        for nextNode in nextMoves:
            # unpack node
            nextState, nextAction, nextCost = nextNode

            if nextState not in explored:

                # set cost
                nextCost += currCost

                # append proper move to path
                nextPath = list(currPath)
                nextPath.append(nextAction)

                # push new move onto queue
                pq.push((nextState, nextPath, nextCost), nextCost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # create variables that have the permitted Pacman's moves.
    from util import PriorityQueue

    # set variables
    maxNodeId = 0
    pq = PriorityQueue()
    explored = {}
    start = problem.getStartState()
    pq.push((start, 0, [], maxNodeId), 0)

    while not pq.isEmpty():

        # get next node
        currState, currCost, currPath, currId = pq.pop()

        if currState in explored:
            continue

        if problem.isGoalState(currState):
            return currPath

        explored[currState] = True

        # get next moves
        nextNodes = problem.getSuccessors(currState)

        for node in nextNodes:
            nextState, nextAction, nextCost = node

            # check that the temp successor state is not already in the list of visited
            # nodes to prevent going to a previous state.
            if nextState not in explored:

                maxNodeId += 1

                nextPath = list(currPath)
                nextPath.append(nextAction)

                # get the cost and push onto priority queue
                cost = nextCost + currCost
                priority = heuristic(nextState, problem) + nextCost + currCost

                pq.push((nextState, cost, nextPath, maxNodeId), priority)

    return []


def foodAStarSearch(start, goal, walls):
    from util import PriorityQueue

    moveVector = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    pq = PriorityQueue()
    explored = {}
    maxWidth = abs(start[0] - goal[0])
    maxHeight = abs(start[1] - goal[1])

    if start == goal:
        return 0

    pq.push((start, 0, 0), 0)
    explored[start] = True
    nodesPopped = 0

    while not pq.isEmpty():
        currNode = pq.pop()
        currPos, currCost, currSteps = currNode

        for move in moveVector:
            nextPos = (currPos[0] + move[0], currPos[1] + move[1])

            if nextPos not in explored:
                # check that move is within bounds
                nextWidth = abs(nextPos[0] - goal[0])
                nextHeight = abs(nextPos[1] - goal[1])
                if nextWidth <= maxWidth and nextHeight <= maxHeight:
                    if not walls[nextPos[0]][nextPos[1]]:

                        #check for goal
                        if nextPos == goal:
                            return currSteps + 1

                        explored[nextPos] = True
                        nextCost = ((nextPos[0] - goal[0]) ** 2 + (nextPos[1] - goal[1]) ** 2) ** 0.5
                        pq.push((nextPos, nextCost, currSteps + 1), nextCost)

    return 999999


def overwrite_line(message):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(message)
    sys.stdout.flush()



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
