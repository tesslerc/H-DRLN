# max_(r=0,1) { max_x ( P(A_r / i = [0, x]) + P(A_(1-r) / i = [x, N]) ) }
# A_0 = distribution 0 , A_1 = distribution 1

# do clustering on all the data!

import numpy as np
import h5py
from sklearn import mixture
import matplotlib.pyplot as plt
import math
import random

# look for best subgroup instead of cutting in half
# for t in range(len(subgroup)):
#   for s in range(t, len(subgroup)):
#        currentValue = probability(subgroup[t, s])
#       if currentValue > previousMax
#           previousMax = currentValue
#           maxRange = (t, s)


def ClusteringCompare(usage_a, uniformity_a, numGaussians_a, usage_b, uniformity_b, numGaussians_b):
    score_a = 1.0 * usage_a * usage_a * usage_a * uniformity_a * numGaussians_a
    score_b = 1.0 * usage_b * usage_b * usage_b * uniformity_b * numGaussians_b
    #if usage_b > 0:
    print('Comparing ' + str(numGaussians_a) + ' gaussians, ' + str(usage_a) + '% usage, ' + str(uniformity_a) + ' uniformity = ' + str(score_a) + '; against ' + str(numGaussians_b) + ' gaussians, ' + str(usage_b) + '% usage, ' + str(uniformity_b) + ' uniformity = ' + str(score_b) + '.')
    if score_a > score_b:
        return True
    #if (usage_a > 0.75 and (usage_a + (numGaussians_a - numGaussians_b) * 0.05 > usage_b) or (usage_a > usage_b)):
    #    return True
    return False


def P(r, probability, minProbability, minLength, maxLength):
    maxProb = 0
    startEnd = {'start': 0, 'end': 0}
    inputLength = len(probability)

    for t in range(inputLength - minLength):
        loopMaxValue = t + maxLength
        if maxLength == 0 or loopMaxValue > inputLength:
            loopMaxValue = inputLength
        for s in range(t + minLength, loopMaxValue):
            curProb = np.sum(probability[t:s, r]) / (s-t)
            if ((curProb >= maxProb and (s - t) >= (startEnd['end'] - startEnd['start'])) or ((s - t) > (startEnd['end'] - startEnd['start']))) and curProb > minProbability:
                # We give priority to:
                #   1. Longest trajectories with average probability > threshold & maxlength >= length >= min length
                #   2. Over several such trajectories - select argmax average probability
                #   3. a
                maxProb = curProb
                startEnd['start'] = t
                startEnd['end'] = s
    return startEnd, maxProb

    # npProb = 0
    # if index > 0:
    #     npProb += np.sum(probability[0:index, r]) / index / 2
    # if index < len(probability):
    #     npProb += np.sum(probability[index:len(probability), 1 - r]) / (len(probability) - index) / 2
    # return npProb
numGaussians = 2
bestUsage = 0
bestUniformity = 0
bestGaussians = 1
minProbability = 0.85
minLength = 5
maxLength = 0  # 0 = disabled

print('Loading data.')
reward = h5py.File('./rewardClean.h5', 'r').get('data')  # [0:10000]
endIndices = []
stopIndex = 0
maxTrajectory = 0

for i in range(len(reward)):
    if reward[i] == 0:
        if (i - stopIndex) > maxTrajectory:
            maxTrajectory = i - stopIndex
        endIndices.append(i + 1)
        stopIndex = i + 1
'''
terminal = h5py.File('./termination.h5', 'r').get('data')[0:200]
for i in range(len(terminal)):
    if terminal[i] == 1:
        if (i - 1 - stopIndex) > maxTrajectory:
            maxTrajectory = i - 1 - stopIndex
        endIndices.append(i - 1)
        stopIndex = i - 1
        # print(str(reward[i - 1]))
'''
data = h5py.File('./activationsClean.h5', 'r').get('data')[1:stopIndex + 1, :]
act = h5py.File('./actionsClean.h5', 'r').get('data')
states = h5py.File('./statesClean.h5', 'r').get('data')[1:stopIndex + 1, :]
tsne = h5py.File('./lowd_activations_800.h5', 'r').get('data')[1:stopIndex + 1, :]
print('Data loaded.')

colors = np.array([[20,0,0],[0,40,0],[0,0,50],[100, 60, 20], [90, 60, 160], [25, 50, 0], [0, 25, 50], [0, 50, 25]])
nth_iteration = 1
while True:
    print('Fitting ' + str(numGaussians) + ' gaussians on data.')
    # gmm = mixture.GaussianMixture(n_components=numGaussians, covariance_type='full', max_iter=1000, warm_start=(numGaussians > 2)).fit(data)
    all_indices = range(len(data))
    random.shuffle(all_indices)
    partial_indices = all_indices[0:min(50000, len(data))]
    partial_data = data[partial_indices, :]
    gmm = mixture.BayesianGaussianMixture(n_components=numGaussians, covariance_type='full', max_iter=1000, warm_start=False).fit(partial_data)  # data
    print('Searching for optimal cut index.')

    probability = gmm.predict_proba(data)
    #finalImage = np.zeros((3, maxTrajectory * 84, len(endIndices) * 84))
    finalDots = np.zeros((3, maxTrajectory, len(endIndices)))
    finalGaussian = np.zeros((3, maxTrajectory, len(endIndices)))

    usagePerSkill = []

    clusterStartTmp = []
    clusterEndTmp = []

    tsneOutTmp = np.zeros((numGaussians, len(data)))
    '''
    tsneMatch = []  # color based on best gaussian matched
    tsneMatch.append([])  # x
    tsneMatch.append([])  # y
    tsneMatch.append([])  # color
    '''
    for i in range(numGaussians):
        clusterStartTmp.append([])
        clusterEndTmp.append([])
        usagePerSkill.append(0)

    for j in range(len(endIndices)):
        startIndex = 0

        if j != 0:
            startIndex = endIndices[j - 1] + 1

        finalDots[:, 0:(endIndices[j] - startIndex), j:(j + 1)] = np.ones((3, (endIndices[j] - startIndex), 1))
	
        finalGaussian[:, 0:(endIndices[j] - startIndex), j:(j + 1)] = np.ones((3, (endIndices[j] - startIndex), 1))
        _maxGaussian = 0

	for __ in range(startIndex, endIndices[j] - 1):
            for _ in range(1, min(3, numGaussians)):
                if probability[__, _] > probability[__, _maxGaussian]:
                    _maxGaussian = _
            finalGaussian[_maxGaussian, __ - startIndex, j] = 0

            '''
            tsneMatch[0].append(tsne[__, 0])
            tsneMatch[1].append(tsne[__, 1])
            tsneMatch[2].append(_maxGaussian)
            '''

        #finalImage[0, 84 * 0:84 * (endIndices[j] - startIndex), 84 * j:84 * (j + 1)] = states[startIndex:endIndices[j], :].reshape((endIndices[j] - startIndex) * 84, 84)
        #finalImage[1, 84 * 0:84 * (endIndices[j] - startIndex), 84 * j:84 * (j + 1)] = states[startIndex:endIndices[j], :].reshape((endIndices[j] - startIndex) * 84, 84)
        #finalImage[2, 84 * 0:84 * (endIndices[j] - startIndex), 84 * j:84 * (j + 1)] = states[startIndex:endIndices[j], :].reshape((endIndices[j] - startIndex) * 84, 84)
        for i in range(numGaussians):
            # #print('Cutting based on Gaussian #' + str(i + 1))

            maxIndex, maxValue = P(i, probability[startIndex:endIndices[j]], minProbability, minLength, maxLength)

            # im = states[startIndex + maxIndex['start']:startIndex + maxIndex['end'], :]
            # finalImage[i, 84 * maxIndex['start']:84 * maxIndex['end'], 84 * j:84 * (j + 1)] = np.zeros(((maxIndex['end'] - maxIndex['start']) * 84, 84))
            if maxValue > 0:
                usagePerSkill[i] = usagePerSkill[i] + maxIndex['end'] - maxIndex['start']
                clusterStartTmp[i].append(maxIndex['start'] + startIndex)
                clusterEndTmp[i].append(maxIndex['end'] + startIndex)

                '''
                colorCoded = np.ones((3, maxIndex['end'] - maxIndex['start'], 1), dtype=np.int)
                colorCoded[0,:,:] = colorCoded[0,:,:] * colors[i % 8, 0]
                colorCoded[1,:,:] = colorCoded[1,:,:] * colors[i % 8, 1]
                colorCoded[2,:,:] = colorCoded[2,:,:] * colors[i % 8, 2]
                finalDots[:, maxIndex['start']:maxIndex['end'], j:(j + 1)] = finalDots[:, maxIndex['start']:maxIndex['end'], j:(j + 1)] + colorCoded
                '''

                finalDots[i % 3, maxIndex['start']:maxIndex['end'], j:(j + 1)] = finalDots[i % 3, maxIndex['start']:maxIndex['end'], j:(j + 1)] * 0.8
				
                tsneOutTmp[i, maxIndex['start'] + startIndex:maxIndex['end'] + startIndex] = 1
                # First state
                #print('Gaussian #' + str(i) + ' start')
                #plt.imshow(states[maxIndex['start'] + startIndex, :].reshape(84, 84), cmap='gray')
                #plt.show()
                # Last state
                #plt.imshow(states[maxIndex['end'] + startIndex, :].reshape(84, 84), cmap='gray')
                #plt.show()
            else:
                clusterStartTmp[i].append(-1)
                clusterEndTmp[i].append(-1)

    states_used = 0.0
    for t in range(len(endIndices)):
        for s in range(maxTrajectory - 1):
            if np.sum(finalDots[:, s, t]) not in [0.0, 3.0]:
                states_used = states_used + 1.0
    usage = states_used * 1.0 / endIndices[len(endIndices) - 1]
    np_ups = np.array(usagePerSkill)
    uniformity = min(np_ups) * 1.0 * numGaussians / np.sum(np_ups)  # between 0 and 1 where 1 is best uniformity
    if ClusteringCompare(usage, uniformity, numGaussians, bestUsage, bestUniformity, bestGaussians) is True:
        bestUsage = usage
        bestGaussians = numGaussians
        bestUniformity = uniformity
        clusterStart = clusterStartTmp
        clusterEnd = clusterEndTmp
        tsneOut = tsneOutTmp
        model = gmm
	
    if (bestGaussians + 2) < numGaussians or numGaussians == 5 and nth_iteration == 6:  # TODO
        break
    if nth_iteration > 5:
        numGaussians = numGaussians + 1
        nth_iteration = 1
    else:
        nth_iteration = nth_iteration + 1

activationsOut = []
terminationOut = []
actionsOut = []

'''
tsneOut = []
tsneOut.append([])  # x
tsneOut.append([])  # y
tsneOut.append([])  # color
'''
for j in range(np.array(clusterStart).shape[0]):
    activationsOut.append([])
    terminationOut.append([])
    actionsOut.append([])
    for i in range(np.array(clusterStart[j]).shape[0]):
        if clusterStart[j][i] != -1:
            for index in range(clusterStart[j][i], clusterEnd[j][i] + 1):
                activationsOut[j].append(data[index, :])
                terminationOut[j].append(0)
                actionsOut[j].append(act[index])
                '''
                tsneOut[0].append(tsne[index, 0])
                tsneOut[1].append(tsne[index, 1])
                tsneOut[2].append(j)
                '''

            terminationOut[j][np.array(terminationOut[j]).shape[0] - 1] = 1
        # print('Gaussian: ' + str(j) + ', StartIdx: ' + str(clusterStart[j][i]) + ', EndIdx: ' + str(clusterEnd[j][i]))

		
print('Best match is: ' + str(bestGaussians) + ' gaussians, with ' + str(bestUsage) + ' percent usage of states.')
for i in range(np.array(activationsOut).shape[0]):
    print('Now looking at skill #' + str(i) + ' activations looks like: ' + str(np.array(activationsOut[i]).shape))

dataOut = h5py.File('SkillsData.h5', 'w')
dataOut.create_dataset('numberSkills', data=bestGaussians)
for i in range(np.array(clusterStart).shape[0]):
    dataOut.create_dataset('activations_' + str(i), data=activationsOut[i])
    dataOut.create_dataset('termination_' + str(i), data=terminationOut[i])
    dataOut.create_dataset('actions_' + str(i), data=actionsOut[i])

dataOut = h5py.File('tsneData.h5', 'w')
dataOut.create_dataset('data', data=tsneOut)

probability = model.predict_proba(data)
tsneMatch = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    tsneMatch[i] = np.argmax(probability[i])

dataOut = h5py.File('tsneMatch.h5', 'w')
dataOut.create_dataset('data', data=tsneMatch)

# clustersOut = h5py.File('clusters.h5', 'w')
# clustersOut.create_dataset('startIndices', data=clusterStart)
# clustersOut.create_dataset('endIndices', data=clusterEnd)

# with h5py.File('./clusters.h5', 'r') as hf:
#    print('List of arrays in this file: \n', hf.keys())

# a = finalImage.transpose()
a = finalDots.transpose()
#a = finalGaussian.transpose()
i = 0
# while i < a.shape[0]:
#    j = 0
#    while j < a.shape[1]:
#        a[i:i + 84, j:j + 84, :] = a[i:i + 84, j:j + 84, :].transpose(1, 0, 2)
#        j += 84
#    i += 84

# print(a.shape[0])

plt.figure(figsize=(20, 10))
plt.imshow(a)
# plt.savefig('ColorBasedOnCuts_clean_20000.png')
plt.show()

'''
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

while optimizing
    gradient = 0
    for all items in array
        act = zeros(#actions)
        act[action[i]] = 1
        gradient = gradient + dW(expectated, act)
    location(n) = location(n-1) - gradient / norm(gradient)


T = W.transpose().dot(x)
U = softmax(T)
O = Kullback-Leiblier(U)
dW = dO/dW = (dU/dW).dot(dO/dU) = (dT/dW).dot(dU/dT).dot(dO/dU) = x.dot(dU/dT).dot(dO/dU)
'''
