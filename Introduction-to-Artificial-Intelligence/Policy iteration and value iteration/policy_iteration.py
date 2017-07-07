
class Actions:
    EAST = 'east'
    NORTH = 'north'
    SOUTH = 'south'
    WEST = 'west'

    def asList(self):
        return [self.EAST, self.NORTH, self.SOUTH, self.WEST]


class TransitionModel:
    _state_space = 0
    _transitions = None
    _actions = None
    _rewards = None

    def __init__(self, state_space):
        if state_space < 0: raise Exception("state_space must be a positive integer.")
        self._state_space = state_space
        self._actions = Actions()
        self._transitions = {}
        self._rewards = []

        prob_east = open("prob_east.txt", "r")
        prob_north = open("prob_north.txt", "r")
        prob_south = open("prob_south.txt", "r")
        prob_west = open("prob_west.txt", "r")

        actions = (self._actions.asList())
        probabilities = (prob_east, prob_north, prob_south, prob_west)
        for a, f in zip(actions, probabilities):
            self._transitions[a] = [[0.0 for x in xrange(self._state_space)] for y in xrange(self._state_space)]
            for line in f.readlines():
                s, s_, prob = map(float, line.strip().split())
                self._transitions[a][int(s) - 1][int(s_) - 1] = prob
            f.close()

        reward_file = open("rewards.txt", "r")
        for l in reward_file.readlines():
            val = l.strip().split()
            if val.count > 0:
                self._rewards.append(float(val.pop()))
        reward_file.close()

    def getRewards(self):
        return self._rewards

    def getVector(self, action):
        if action not in self._actions.asList():
            raise Exception("Invalid argument.")
        if action == self._actions.EAST:
            return 9
        if action == self._actions.NORTH:
            return 1
        if action == self._actions.SOUTH:
            return -1
        if action == self._actions.WEST:
            return -9
        return -999999

    def P(self, x, y, action):
        if x < 0 or y < 0 or action not in self._actions.asList():
            raise Exception('Invalid argument')
        if x >= self._state_space or y >= self._state_space:
            raise Exception('Invalid argument')

        return self._transitions[action][x][y]


class MDP:

    _state_space = 0
    _rewards = None
    _actions = None
    _tm = None

    def getRewards(self):
        return self._tm.getRewards()

    def getActions(self):
        return self._actions

    def getStateSpace(self):
        return self._state_space

    def getTransitionModel(self):
        return self._tm

    def __init__(self, state_space):
            if state_space < 0: raise Exception("state_space must be a positive integer.")
            self._state_space = state_space
            self._tm = TransitionModel(self._state_space)
            self._actions = Actions()

    def init_matrix(self, val, size):
        return [[val for x in xrange(size)] for y in xrange(size)]


class PolicyIteration:
    _mdp = None
    _policy = None
    _util = None
    _util_ = None
    _discount = 0.99

    def __init__(self, mdp):
        if mdp is None:
            raise Exception("Argument null.")
        self._mdp = mdp
        size = self._mdp.getStateSpace()
        self._util = [0] * size
        self._util_ = [0] * size
        self._policy = [Actions.NORTH for x in xrange(size)]

    def calculate(self):
        if self._mdp is None:
            raise Exception("MDP null.")

        unchanged = False
        policy = self._policy
        util = self._util
        size = self._mdp.getStateSpace()
        actions = self._mdp.getActions().asList()

        while not unchanged:
            util = self._eval(policy, util)
            unchanged = True
            for s in xrange(size):
                best_value = -999999.99
                best_action = ''
                for a in actions:
                    t = self._PU(a, s, util)
                    if t > best_value:
                        best_value = t
                        best_action = a

                if best_value > self._PU(policy[s], s, util):
                    policy[s] = best_action
                    unchanged = False
        self._util_ = util
        return policy

    def getUtility(self):
        return self._util_

    def _eval(self, policy, util):
        if self._mdp is None:
            raise Exception("MDP null.")

        u = list(util)
        size = self._mdp.getStateSpace()
        R = self._mdp.getRewards()
        d = self._discount

        for s in xrange(size):
            r = R[s]
            pu = d * self._PU(policy[s], s, u)
            u[s] = r + pu

        return u

    def _PU(self, action, state, util):
        if self._mdp is None:
            raise Exception("MDP null.")
        num = 0
        for i in xrange(mdp.getStateSpace()):
            num += self._mdp.getTransitionModel().P(state, i, action) * util[i]

        return num

mdp = MDP(81)
p = PolicyIteration(mdp)
policy = p.calculate()
util = p.getUtility()
f = open("PolicyIteration.txt", "w")

for s in xrange(mdp.getStateSpace()):
    d = {}
    if util[s] > 0:
        if s < 72:
            d["East"] = util[s + 9]
        if s > 8:
            d["West"] = util[s - 9]
        if s % 9 != 0:
            d["North"] = util[s - 1]
        if s % 9 != 8:
            d["South"] = util[s + 1]
        string = "(" + str(s + 1) + ", " + str(util[s]) + ", " + max(d, key=d.get) + ")"
        f.write(str(s+1)+" "+str(util[s])+" "+max(d, key=d.get)+"\n")
        print string
f.close()
