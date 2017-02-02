import numpy as np
import h5py

debug = False

termination = h5py.File('./termination.h5', 'r').get('data')
reward = h5py.File('./reward.h5', 'r').get('data')
activations = h5py.File('./activations.h5', 'r').get('data')
actions = h5py.File('./actions.h5', 'r').get('data')
qvals = h5py.File('./qvals.h5', 'r').get('data')
states = h5py.File('./states.h5', 'r').get('data')

if debug:
    print('Shape of termination: \n', (np.array(termination)).shape)
    print('Shape of activations: \n', (np.array(activations)).shape)
    print('Shape of actions: \n', (np.array(actions)).shape)
    print('Shape of reward: \n', (np.array(reward)).shape)
    print('Shape of qvals: \n', (np.array(qvals)).shape)
    print('Shape of states: \n', (np.array(states)).shape)

startTrajectory = []  # first index in a trajectory that leads to success
endTrajectory = []  # last index in a trajectory that leads to success

initialIndex = 0
for i in range(len(termination) - 1):
    if debug:
        print(str(i) + ' , ' + str(termination[i]) + ' , ' + str(reward[i]))
    if termination[i + 1] == 1 and reward[i] == 0:
        startTrajectory.append(initialIndex)
        endTrajectory.append(i)
        if debug:
            print('Success: ' + str(initialIndex) + ' , ' + str(i))
        initialIndex = i + 2
    elif termination[i + 1] == 1:
        initialIndex = i + 2

rewardClean = []
activationsClean = []
qvalsClean = []
statesClean = []
actionsClean = []

totalStates = 0

for i in range(len(startTrajectory)):
    totalStates += endTrajectory[i] - startTrajectory[i] + 1
    for j in range(startTrajectory[i], endTrajectory[i]):
        rewardClean.append(reward[j])
        activationsClean.append(activations[j, :])
        qvalsClean.append(qvals[j])
        actionsClean.append(actions[j])
        statesClean.append(states[j, :])

rCleanFile = h5py.File('rewardClean.h5', 'w')
rCleanFile.create_dataset('data', data=rewardClean)

aCleanFile = h5py.File('actionsClean.h5', 'w')
aCleanFile.create_dataset('data', data=actionsClean)

actCleanFile = h5py.File('activationsClean.h5', 'w')
actCleanFile.create_dataset('data', data=activationsClean)

qCleanFile = h5py.File('qvalsClean.h5', 'w')
qCleanFile.create_dataset('data', data=qvalsClean)

sCleanFile = h5py.File('statesClean.h5', 'w')
sCleanFile.create_dataset('data', data=statesClean)

print('Done! Total states kept: ' + str(totalStates))
