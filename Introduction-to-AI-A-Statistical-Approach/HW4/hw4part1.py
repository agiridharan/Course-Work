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

testdata = np.loadtxt("hw4test.txt", dtype = int)
toremove = []

for i in range(len(testdata)):
   if testdata[i][-1] == 1:
      testdata[i][-1] = -1
   elif testdata[i][-1] == 2:
      testdata[i][-1] = 1
   else:
      toremove.append(i)
testdata = np.delete(testdata, toremove, axis=0)

w = np.zeros(819, dtype = int)

def error(w, data):
   size = len(data)
   errors = 0
   for dpoint in data:
      if np.dot(w, dpoint[:-1]) <= 0 :
         if not dpoint[-1] == -1:
            errors += 1
      else:
         if not dpoint[-1] == 1:
            errors += 1
   return float(errors)/size

def voteerror(w, data):
   size = len(data)
   errors = 0
   for dpoint in data:
      onesum = 0      
      for per in w:
         dot = np.dot(per[0], dpoint[:-1])
         sign = np.sign(dot)
         if sign == 0:
            sign = -1
         onesum += per[1]*sign
      sign = np.sign(onesum)
      if sign == 0:
         sign = -1
      
      if not sign == dpoint[-1]:
         errors += 1
   return float(errors)/size
      

def perceptron(w, data):
   for dpoint in data:
      if np.dot(w, dpoint[:-1])*dpoint[-1]<= 0:
         w = np.add(w, np.multiply(dpoint[-1], dpoint[:-1]))
   return w

def voteperceptron(w, data, passes):
   for i in range(1,passes):
      data = np.concatenate((data, data))      
   m = 0
   c = [1]
   ws = [w]
   
   for dpoint in data:
      if np.dot(w, dpoint[:-1])*dpoint[-1] <= 0:
         ws.append(np.add(w, np.multiply(dpoint[-1], dpoint[:-1])))
         w = ws[-1]
         m += 1
         c.append(1)
      else:
         c[m] += 1
   return zip(ws, c)

def avgperceptron(w, data, passes):
   for i in range(1,passes):
      data = np.concatenate((data, data))      
   result = w
   
   for dpoint in data:
      if np.dot(w, dpoint[:-1])*dpoint[-1] <= 0:
         w = np.add(w, np.multiply(dpoint[-1], dpoint[:-1]))
      result = np.add(result, w)
      
   return result

print "Perceptron"
w = perceptron(w, traindata)
print "Pass 1 training error:", error(w, traindata)

w = perceptron(w, traindata)
print "Pass 2 training error:", error(w, traindata)
print "Pass 2 testing error:", error(w, testdata)

w = perceptron(w, traindata)
print "Pass 3 training error:", error(w, traindata)
print "Pass 3 testing error:", error(w, testdata)

w = perceptron(w, traindata)
print "Pass 4 training error:", error(w, traindata)
print "Pass 4 testing error:", error(w, testdata)

w = np.zeros(819, dtype = int)
print "Voted Perceptron"
ws = voteperceptron(w, traindata, 1)
print "Pass 1 training error:", voteerror(ws, traindata)

ws = voteperceptron(w, traindata, 2)
print "Pass 2 training error:", voteerror(ws, traindata)
print "Pass 2 testing error:", voteerror(ws, testdata)

ws = voteperceptron(w, traindata, 3)
print "Pass 3 training error:", voteerror(ws, traindata)
print "Pass 3 testing error:", voteerror(ws, testdata)

ws = voteperceptron(w, traindata, 4)
print "Pass 4 training error:", voteerror(ws, traindata)
print "Pass 4 testing error:", voteerror(ws, testdata)

w = np.zeros(819, dtype = int)
print "Average Perceptron"
w = avgperceptron(w, traindata, 1)
print "Pass 1 training error:", error(w, traindata)

w = avgperceptron(w, traindata, 2)
print "Pass 2 training error:", error(w, traindata)
print "Pass 2 testing error:", error(w, testdata)

w = avgperceptron(w, traindata, 3)
print "Pass 3 training error:", error(w, traindata)
print "Pass 3 testing error:", error(w, testdata)

w = avgperceptron(w, traindata, 4)
print "Pass 4 training error:", error(w, traindata)
print "Pass 4 testing error:", error(w, testdata)
