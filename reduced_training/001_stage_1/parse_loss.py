import re
import numpy as np
import matplotlib.pyplot as plt

unpool_method = 'MAX'
max_iter = 6000
test_iter = 500

f = open('training_log/caffe.INFO.cvlab-ara.max', 'r')
s = f.read()	# read the whole file into one string

# extract training loss
pos = [m.end()+1  for m  in re.finditer('Train net output #0: seg-loss =', s)]
trainLoss_end = [s[pos[i]:pos[i]+10].find(' ') for i in xrange(len(pos))]
trainLoss = [float(s[pos[i]:pos[i]+trainLoss_end[i]]) for i in xrange(len(pos))]

# extract validation loss
pos_ = [m.end()+1  for m  in re.finditer('Test net output #1: seg-loss =', s)]
valLoss_end = [s[pos_[i]:pos_[i]+10].find(' ') for i in xrange(len(pos_))]
valLoss = [float(s[pos_[i]:pos_[i]+valLoss_end[i]]) for i in xrange(len(pos_))]

loss = [trainLoss, valLoss]

# extract validation accuracy
_pos = [m.end()+1  for m  in re.finditer('Test net output #0: seg-accuracy =', s)]
acc_end = [s[_pos[i]:_pos[i]+10].find('\n') for i in xrange(len(_pos))]
acc = [float(s[_pos[i]:_pos[i]+acc_end[i]]) for i in xrange(len(_pos))]

# draw loss plots
x1 = xrange(len(pos))
x2 = xrange(0, max_iter+1, test_iter)
f, lossplt = plt.subplots(3)
f.suptitle('unpooling method: ' + unpool_method, fontsize=18)
lossplt[0].plot(x1, loss[0])
lossplt[0].set_title('Training loss')
lossplt[0].annotate('%f' % loss[0][-1], xy=(x1[-1], loss[0][-1]), textcoords='data')
lossplt[1].plot(x2, loss[1])
lossplt[1].set_title('Validation loss')
lossplt[2].plot(x2, acc)
lossplt[2].set_title('Validation Accuracy')
lossplt[2].set_ylim([0, 1])
i = 0
for xy in zip(x2, loss[1]):
        lossplt[1].annotate('%f' % loss[1][i], xy=xy, textcoords='data')
        i = i + 1
i = 0
for xy in zip(x2, acc):
	lossplt[2].annotate('%f' % acc[i], xy=xy, textcoords='data')
	i = i + 1

plt.show()

'''
fig = plt.plot(x, loss)
plt.title('Loss change during iterations')
plt.xlabel('iter')
plt.ylabel('loss')
plt.show(fig)
'''
