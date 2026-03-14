# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        """
        repeat k times:
            create newValues
            for each state s:
                if terminal:
                    newValues[s] = 0
                else:
                    compute value of each action
                    newValues[s] = max(actionValues)
            replace values with newValues
        """
        for i in range(self.iterations):

            newValues = util.Counter()

            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                    continue
                    
                bestValue = -float('inf')

                for action in self.mdp.getPossibleActions(state):
                    q = self.computeQValueFromValues(state, action)
                    if q > bestValue:
                        bestValue = q
                
                newValues[state] = bestValue
            
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        """
        q = 0
        for (nextSate, prob) in transition states:
            get reward
            q += prob * (reward + discount * values[nextState])
        return q
        """
        qValue = 0

        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        for nextState, prob in transitions:
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.values[nextState])

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        """
        if terminal:
            return none
        
        bestAction = None
        bestValue = -inf

        for action in possible actions:
            qValue = computeQValueFromValues

            if qValue > bestValue:
                bestValue = qValue
                bestAction = action
            
        return bestAction
        """
        if self.mdp.isTerminal(state):
            return None
        
        bestAction = None
        bestValue = -float('inf')

        for action in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, action)

            if q > bestValue:
                bestValue = q
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        - compute predecessprs
        - initialize pq
        - iterate
        - update predecessors
        """
        predecessors = {}

        for s in self.mdp.getStates():
            predecessors[s] = set()

        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                    if prob > 0:
                        predecessors[nextState].add(s)

        pq = util.PriorityQueue()

        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue

            bestQ = max(self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s))
            diff = abs(self.values[s] - bestQ)
            pq.push(s, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                return
            
            s = pq.pop()

            if not self.mdp.isTerminal(s):
                bestQ = max(self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s))
                self.values[s] = bestQ

            for p in predecessors[s]:
                if self.mdp.isTerminal(p):
                    continue
                bestQ = max(self.computeQValueFromValues(p, a) for a in self.mdp.getPossibleActions(p))
                diff = abs(self.values[p] - bestQ)

                if diff > self.theta:
                    pq.update(p, -diff)