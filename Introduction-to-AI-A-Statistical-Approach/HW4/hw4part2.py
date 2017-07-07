import numpy as np

traindata = np.loadtxt("hw4train.txt", dtype = int)
toremove = []

for i in range(len(traindata)):
   if traindata[i][-1] == 1:
      traindata[i][-1] = -1
   elif traindata[i][-1] == 2:
      traindata[i][-1] = 1
   else:
      toremove.append(i)
traindata = np.delete(traindata, toremove, axis=0)

dictionary = np.loadtxt("hw4dictionary.txt", dtype = str)

def avgperceptron(w, data, passes):
   for i in range(1,passes):
      data = np.concatenate((data, data))      
   result = w
   
   for dpoint in data:
      if np.dot(w, dpoint[:-1])*dpoint[-1] <= 0:
         w = np.add(w, np.multiply(dpoint[-1], dpoint[:-1]))
      result = np.add(result, w)
      
   return result
w = np.zeros(819, dtype = int)
w = avgperceptron(w, traindata, 3)

maxes = []
mins = []

for i in range(3):
   currmax = np.argmax(w)
   maxes.append(currmax)
   currmin = np.argmin(w)
   mins.append(currmin)
   
   w = np.delete(w, currmax)
   w = np.delete(w, currmin)
   
print "Decreasing Maxes:", maxes
print dictionary[maxes]
print "Increasing Mins=:", mins
print dictionary[mins]