# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    def __init__(self):
        self.INFINITY = 999999

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        numCurrFood = len(currentGameState.getFood().asList())

        if action == Directions.STOP:
            return 0

        numNextFood = 0
        if successorGameState.isLose():
            return -self.INFINITY
        elif successorGameState.isWin():
            return self.INFINITY

        dist_food = []
        for food in newFood.asList():
            numNextFood += 1
            dist = manhattanDistance(food, newPos)
            dist_food.append(dist)

        if numNextFood == 0 or numNextFood != numCurrFood:
            return self.INFINITY
        else:
            return self.INFINITY - min(dist_food)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.INFINITY = 999999
        self.max = -self.INFINITY
        self.min = self.INFINITY


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        """*** YOUR CODE HERE ***"""
        return self.getMax(gameState, 1)

    def getMax(self, gamestate, depth):
        actions = gamestate.getLegalActions(0)
        max_action = [Directions.STOP]
        max_value = -self.INFINITY
        num_ties = 0

        if len(actions) == 0:
            return self.evaluationFunction(gamestate)

        for next_action in actions:
            nextgamestate = gamestate.generateSuccessor(0, next_action)

            # Calculate cost
            if nextgamestate.getNumAgents() > 1:
                next_value = self.getMin(nextgamestate, 1, depth)
            elif depth < self.depth:
                next_value = self.getMax(nextgamestate, depth + 1)
            else:
                next_value = self.evaluationFunction(nextgamestate)

            # Compare value with max value
            if next_value == max_value:
                max_action.append(next_action)
                num_ties += 1
            elif next_value > max_value:
                max_value = next_value
                max_action = [next_action]
                num_ties = 0

        if depth == 1:
            index = random.randint(0, num_ties)
            return max_action[index]
        else:
            return max_value

    def getMin(self, gamestate, ghost_index, depth):
        actions = gamestate.getLegalActions(ghost_index)
        min_value = self.INFINITY

        if len(actions) == 0:
            return self.evaluationFunction(gamestate)

        for next_action in actions:
            nextgamestate = gamestate.generateSuccessor(ghost_index, next_action)

            # calculate value
            if ghost_index == nextgamestate.getNumAgents() - 1:
                if depth < self.depth:
                    next_value = self.getMax(nextgamestate, depth + 1)
                else:
                    next_value = self.evaluationFunction(nextgamestate)

            else:
                next_value = self.getMin(nextgamestate, ghost_index + 1, depth)

            # compare value with min value
            if next_value < min_value:
                min_value = next_value

        return min_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, -self.INFINITY, self.INFINITY, 0)

    def max_value(self, state, a, b, depth):
        if depth == self.depth:
            isLastLevel = True
        else:
            isLastLevel = False

        if depth == 0:
            isTopLevel = True
        else:
            isTopLevel = False

        best_move = Directions.STOP
        v = -self.INFINITY
        actions = state.getLegalActions(0)

        if len(actions) == 0:
            if isTopLevel:
                return best_move
            return self.evaluationFunction(state)

        for next_action in actions:
            successor = state.generateSuccessor(0, next_action)

            if successor.getNumAgents() > 1: # there are ghosts
                v = max(v, self.min_value(successor, a, b, depth+1, 1))
            else:
                if isLastLevel:
                    v = max(v, self.evaluationFunction(successor))
                else:
                    v = max(v, self.max_value(successor, a, b, depth+1))

            if v > b:
                if isTopLevel:
                    return next_action
                return v
            if v > a:
                a = v
                best_move = next_action

        if isTopLevel:
            return best_move
        return v

    def min_value(self, state, a, b, depth, ghost_index):

        if depth == self.depth:
            isLastLevel = True
        else:
            isLastLevel = False
        if ghost_index == state.getNumAgents() - 1:
            isLastGhost = True
        else:
            isLastGhost = False

        v = self.INFINITY
        actions = state.getLegalActions(ghost_index)

        if len(actions) == 0:
            return self.evaluationFunction(state)

        for next_action in actions:
            successor = state.generateSuccessor(ghost_index, next_action)

            if isLastGhost:
                if isLastLevel:
                    v = min(v, self.evaluationFunction(successor))
                else:
                    v = min(v, self.max_value(successor, a, b, depth))
            else:
                v = min(v, self.min_value(successor, a, b, depth, ghost_index+1))

            if v < a:
                return v
            b = min(b, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, 0)

    def expectimax(self, state, depth):

        best_action = Directions.STOP
        best_value = -999999
        actions = state.getLegalActions(0)

        if len(actions) == 0:
            if depth == 0:
                return best_action
            return self.evaluationFunction(state)

        for action in actions:
            successor = state.generateSuccessor(0, action)

            if successor.getNumAgents() > 1:
                value = self.expectimin(successor, depth + 1, 1)
            elif depth == self.depth:
                value = self.evaluationFunction(successor)
            else:
                value = self.expectimax(successor, depth + 1)

            if value > best_value:
                best_value = value
                best_action = action

        if depth == 0:
            return best_action
        else:
            return best_value

    def expectimin(self, state, depth, agent_index):

        average = 0
        actions = state.getLegalActions(agent_index)
        num_actions = len(actions)

        if num_actions == 0:
            if agent_index < state.getNumAgents() - 1:
                average = self.expectimin(state, depth, agent_index + 1)
            elif depth == self.depth:
                average = self.evaluationFunction(state)
            else:
                average = self.expectimax(state, depth)
            return average

        for action in actions:
            successor = state.generateSuccessor(agent_index, action)

            if agent_index < successor.getNumAgents() - 1:
                average += (self.expectimin(successor, depth, agent_index + 1) * 1.0) / num_actions
            elif depth == self.depth:
                average += (self.evaluationFunction(successor) * 1.0) / num_actions
            else:
                average += (self.expectimax(successor, depth) * 1.0) / num_actions

        return average


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: basically like a single layer neural net with
      Vi being the vector of inputs and Vb the vector of bias:
      Vi = < distToClosestFood, numFoodLeft, distToPowerPellets, numPowerPellets, distToUnscarredGhost, distToScaredGhosts >
      Vb = < b0, b1, b2, b3, b4 >

      The bias vector will be a weighted bias that represents how much we care about that value.
      We will then return the dot product of the two vectors.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose(): # you lost anyways so why calculate anything more?
        return -999999

    # < closestFood, numFood, closestPowerPellet, numPowerPellet, closestUnscaredGhost, closestScaredGhost >
    Vb = [9.9, 9999, 45, 9999, 99, 2.5]
    Vi = []

    # Get pacman position
    pacman = currentGameState.getPacmanPosition()

    # --- calculate values for Vi and append --- #
    # NOTE: every input value is inserted as (1.0 / (value + 1.0)), with the exception of numFood
    #       which is inserted as 20.0 when 0 and closestUnscaredGhost which is inserted as
    #       1.0 - (1.0/ (closestUnscaredGhost + 1.0)).
    #       Lastly, the added 1.0 in (1.0 / (value + 1.0)) is there to prevent a divide by 0 error
    # Calculate Closest Food
    closestFood = 999999
    foodGrid = currentGameState.getFood().asList()
    for food in foodGrid:
        closestFood = min(manhattanDistance(food, pacman), closestFood)
    # append
    if closestFood == 999999:
        Vi.append(1.0)
    else:
        Vi.append(1.0/closestFood+1.0)

    # calculate numFood
    numFood = len(foodGrid)
    if numFood == 0:
        Vi.append(20.0) # add extra weight for numFood == 0
    else:
        Vi.append(1.0/(len(foodGrid)+1.0))

    # calculate distance to nearest Power pellot
    closestPellet = 999999
    pelletGrid = currentGameState.getCapsules()
    for pellet in pelletGrid:
        closestPellet = min(manhattanDistance(pellet, pacman), closestPellet)
    # append
    Vi.append(1.0/(closestPellet+1.0))

    # num power pellets left
    numPowerPellets = len(pelletGrid)
    if numPowerPellets == 0:
        Vi.append(1.0)
    else:
        Vi.append(1.0/(numPowerPellets+1.0))

    # calculate distance to unscared ghost and scared ghosts
    closestScaredGhost = 999999
    closestUnscaredGhost = 999999
    ghosts = currentGameState.getGhostStates()
    i = 1
    for ghost in ghosts:
        dist = manhattanDistance(currentGameState.getGhostPosition(i), pacman)
        if ghost.scaredTimer > 0:
            closestScaredGhost = min(dist, closestScaredGhost)
        else:
            closestUnscaredGhost = min(dist, closestUnscaredGhost)

    # append
    if closestScaredGhost == 999999: # gives pacman insentive to eat ghost instead of hang out next to it
        Vi.append(1.0)
    else:
        Vi.append(1/(closestScaredGhost+1.0))
    # append
    Vi.append(1.0-(1.0/(closestUnscaredGhost+1.0)))

    sum = 0.0
    for input, bias in zip(Vi, Vb):
        sum += (input * bias)
    return sum

# Abbreviation
better = betterEvaluationFunction

