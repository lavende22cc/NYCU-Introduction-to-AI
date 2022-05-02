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


"""
part 2-1
"""

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
        "*** YOUR CODE HERE ***"
        # Begin your code

        """
        V(s) = max_{a in actions} Q(s,a)
        policy(s) = arg_max_{a in actions} Q(s,a)
        """
        for i in range(self.iterations):
            nextQValues = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                bestAction = self.computeActionFromValues(state)  # this is policy(s)
                bestValue = self.computeQValueFromValues(state, bestAction)  # this is V(s)
                nextQValues[state] = bestValue
            self.values = nextQValues
        # End your code


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
        "*** YOUR CODE HERE ***"
        # Begin your code
        QValue = 0

        Transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for i in range(len(Transitions)):
            nextState, Prob = Transitions[i]
            nextReward = self.mdp.getReward(state, action, nextState)
            QValue += Prob * (nextReward + self.discount * self.getValue(nextState))
        return QValue
        # End your code

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        if self.mdp.isTerminal(state):
            return None

        QValues = util.Counter() # A dict(), but defalt value is 0.


        # We will iterate all possible actions to count the Qvalues.
        PossibleActions = self.mdp.getPossibleActions(state)
        for possibleAction in PossibleActions:
            QValues[possibleAction] = self.computeQValueFromValues(state, possibleAction)


        # Iterate all possible actions to get the best action which has max Qvalue.
        bestAction = None
        bestValue = -1000000000.0
        for possibleAction in PossibleActions:
            if  QValues[possibleAction] > bestValue:
                bestValue = QValues[possibleAction]
                bestAction = possibleAction

        return bestAction
        # End your code

    def getPolicy(self, state):
        """
        The policy is the best action in the given state
        according to the values computed by value iteration.
        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        """
        The q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        return self.computeQValueFromValues(state, action)
