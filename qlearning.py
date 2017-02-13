import numpy as np

br = 100    #big reward
sr = 0      #small reward

# init reward table
R = np.array([[-1, -1, -1, -1, sr, -1],
              [-1, -1, -1, sr, -1, br],
              [-1, -1, -1, sr, -1, -1],
              [-1, sr, sr, -1, sr, -1],
              [sr, -1, -1, sr, -1, br],
              [-1, sr, -1, -1, sr, br]])
# alpha = 0.9
gamma = 0.8

# init Q table
Q = np.zeros(R.shape)

#pick a starting state
state = np.random.randint(0, 5)

for i in range(1000):
    # list of all posible actions in this state
    possible_actions = [i for i, x in enumerate(R[state, :]) if x >= 0]
    # choose radomly one action form the list of all posible actions
    action = np.random.choice(possible_actions)

    # action is our next state form now
    next_state = action

    # update Q matrix
    Q[state,action] = R[state,action] + gamma*np.max(Q[next_state,:])

    # if R[state,action] == 100: break

    # set next state as current state
    state = next_state


# print Q matrix scaled in percents (and floored)
print(np.array(Q/np.max(Q)*100).astype(int))

