import re
import numpy as np
import matplotlib.pyplot as plt

f = open('training_log/caffe.cvlab-ara.mschoi.log.INFO.20150922-022356.23789', 'r')
s = f.read()
pos = [m.end()+1  for m  in re.finditer('Train net output #0: seg-loss =', s)]
loss_end = [s[pos[i]:pos[i]+10].find(' ') for i in xrange(len(pos))]
loss = [float(s[pos[i]:pos[i]+loss_end[i]]) for i in xrange(len(pos))]

x = xrange(len(pos))
fig = plt.plot(x, loss)
plt.title('Loss change during iterations')
plt.xlabel('iter')
plt.ylabel('loss')
plt.show(fig)
