import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=.01, gamma=0.9):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        #self.epsilon = 0.5
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.epsilon = 1 / i_episode
        return self.action_from_epsilon_greedy(state, self.Q, self.epsilon, self.nA)
    
    def action_from_epsilon_greedy(self, state, Q_s, epsilon, nA):
        probs = self.update_probs(state, Q_s, epsilon, nA)#Q_s needs to be replaced by Q[s]
        if state in Q_s:
            action = np.random.choice(np.arange(nA), p=probs)  
        else:
            action = env.action_space.sample()
        return action
    
    def update_probs(self, state, Q_s, epsilon, nA):
        probs = np.ones(nA) * epsilon / nA
        best_a = np.argmax(Q_s[state])
        probs[best_a] = 1 - epsilon + epsilon / nA
        return probs 
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q = self.update_Q(state, next_state, action, reward, self.Q, self.alpha, self.gamma)
        #self.Q[state][action] += 1
        
    
    def update_Q(self, state, next_state, action, reward, Q, alpha, gamma):
        Q_old = Q[state][action]
        best_action = np.argmax(Q[state]) #find the best next action given the current state
        Q[state][action] = Q_old + alpha * (gamma * reward + Q[next_state][best_action] - Q_old)
        return Q
