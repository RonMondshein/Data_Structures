import math
import random

from AVLTREE_fullcode import AVLTreeList
index = 2
n = 1500 * math.pow(2, index)
""""
#insert
AVLtree = AVLTreeList()
allSteps = 0
for i in range((int)(n)):
    if AVLtree.length() == 0:
        steps = AVLtree.insert(0, 0)
    else:
        j = random.randint(0, AVLtree.length() - 1)
        steps = AVLtree.insert(j, j)
    allSteps += steps

print(allSteps)


"""""
#delete
AVLtree = AVLTreeList()
allSteps = 0
for i in range((int)(n)):
    if AVLtree.length() == 0:
        AVLtree.insert(0, 0)
    else:
        j = random.randint(0, AVLtree.length() - 1)
        AVLtree.insert(j, j)

for i in range((int)(n)):
    if AVLtree.length() == 0:
        steps = AVLtree.delete(0)
    else:
        j = random.randint(0, AVLtree.length() - 1)
        steps = AVLtree.delete(j)
    allSteps += steps

print(allSteps)

""""
#insert & delete
AVLtree = AVLTreeList()
allSteps = 0
for i in range((int)(n/2)):
    if AVLtree.length() == 0:
        AVLtree.insert(0, 0)
    else:
        j = random.randint(0, AVLtree.length() - 1)
        AVLtree.insert(j, j)

for i in range((int)(n/4)):
    if AVLtree.length() == 0:
        steps1 = AVLtree.delete(0)
    else:
        j = random.randint(0, AVLtree.length() - 1)
        steps1 = AVLtree.delete(j)
        j = random.randint(0, AVLtree.length() - 1)
        steps2 = AVLtree.insert(j, j)
    allSteps = allSteps +steps1 +steps2

print(allSteps)
"""