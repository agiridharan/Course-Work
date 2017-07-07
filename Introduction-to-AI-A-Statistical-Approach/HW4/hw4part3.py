import numpy as np
np.set_printoptions(threshold = np.inf)

traindata = np.loadtxt("hw4train.txt", dtype = int)
testdata = np.loadtxt("hw4test.txt", dtype = int)

datas = []

for i in range(6):
   data = np.empty([len(traindata), 820], dtype = int)
   data[:] = traindata
   for entry in data:
      if entry[-1] == (i+1):
         entry[-1] = 1
      else:
         entry[-1] = -1
   datas.append(data)

def perceptron(w, data):
   for dpoint in data:
      if np.dot(w, dpoint[:-1])*dpoint[-1]<= 0:
         w = np.add(w, np.multiply(dpoint[-1], dpoint[:-1]))
   return w
     

w = np.zeros(819, dtype = int)

ws = []

for data in datas:
   ws.append(perceptron(w, data))
   
confusion = np.zeros((7,6),dtype = float)

for x in testdata:
   labels = []
   for i in range(len(ws)):
      if np.sign(np.dot(ws[i], x[:-1])) > 0:
         labels.append(i)
   
   if not len(labels) == 1:
      confusion[6,x[-1]-1] += 1
   else:
      confusion[labels[0],x[-1]-1] += 1

n = np.bincount(testdata.T[-1])[1:]

confusion = confusion.T
for i in range(len(confusion)):
   np.divide(confusion[i], float(n[i]), confusion[i])

confusion = confusion.T
print confusion         