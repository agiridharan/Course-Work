import numpy as np
from scipy import stats

class Node:
   def __init__(self, function, strfunc, values):
       self.left = None
       self.right = None
       self.parent = None
       self.function = function
       self.values = values
       self.strfunc = strfunc
       return
   
   def __eq__(self, other):
        return np.array_equal(self.values, other.values)


def probability(entry):
    result = map(float, np.bincount(entry))
    for i in range(len(result)):
        result[i] = result[i]/len(entry)
    return result

def pure(probs):
   if len(probs) == 1 or probs[0] == 1 or probs[1] == 1:
      return True
   return False

train = np.loadtxt("hw3train.txt", dtype = float)
traindata = train[:, 0:-1]
trainlabels = np.array(train[:,-1], dtype = int)

firvalues = []
for i in range(len(traindata)):
    firvalues.append(i)
   
root = Node(None, "root", firvalues)
leaves = [root]

for leaf in leaves:
    labels = []
    for val in leaf.values:
        labels.append(trainlabels[val])
    prob = probability(labels)
    if pure(prob):
        retval = trainlabels[leaf.values[0]]
        leaf.fuction = lambda x: retval
        leaves.remove(leaf)

while leaves:
   currdata = []
   for i in leaves[0].values:
      currdata.append(traindata[i])
   currdata = np.array(currdata)
   splits = []
   
   for feature in range(len(currdata.T)):
      ts = []
      featvals = []
      for i in currdata.T[feature]:
         featvals.append(i)
      featvals = np.unique(featvals)   
      tvalues = []
      for i in range(1,len(featvals)):
         tvalues.append((featvals[i] + featvals[i-1])/2)
      #if len(featvals) == 1:
       #  tvalues.append(featvals[0])
      tvalues = set(tvalues)
    
      for t in tvalues:
         true = []
         false = []
         for i in range(len(currdata.T[feature])):
            if currdata.T[feature,i] < t:
               true.append(trainlabels[i])
            else:   
               false.append(trainlabels[i])
                    
         ptrue = float(len(true))/len(currdata.T[feature])
         pfalse = float(len(false))/len(currdata.T[feature])
            
         condent = stats.entropy(probability(true)) * ptrue + stats.entropy(probability(false))*pfalse
         ts.append([condent,t])
      if tvalues:
         ts = np.matrix(ts)
         amin = np.argmin(ts.T[0])
         splits.append([feature, ts[amin,1], ts[amin,0]])

   splits = np.matrix(splits)
   amin = np.argmin(splits.T[2])
   
   fnum = splits[amin,0]
   tnum = splits[amin,1]

   func = lambda x: x[fnum]<tnum
   leaves[0].function = func
   leaves[0].strfunc = str(fnum+1) + " < " + str(tnum)
    
   leftnodevals = []
   rightnodevals = []
    
   for i in leaves[0].values:
      if func(traindata[i]):
         leftnodevals.append(i)
      else:
         rightnodevals.append(i) 
    
   if rightnodevals: 
      rightnode = Node(None, None, rightnodevals)
      rightnode.parent = leaves[0]
      leaves[0].right = rightnode
      leaves.append(rightnode)
   if leftnodevals:
      leftnode = Node(None, None, leftnodevals)
      leftnode.parent = leaves[0]
      leaves[0].left = leftnode
      leaves.append(leftnode)

   leaves.remove(leaves[0])
   toremove = []
   funcs = []
   for leaf in leaves:
      labels = []
      for val in leaf.values:
         labels.append(trainlabels[val])
      prob = probability(labels)
      if pure(prob):
         retval = trainlabels[leaf.values[0]]
         func = lambda: retval
         funcs.append(func)
         leaf.strfunc = "return"
         toremove.append(leaf)
   for i in range(len(toremove)):
      toremove[i].function = funcs.pop()
      leaves.remove(toremove[i])
         

level = 1
nodes = [root]

while level < 4:
   newnodes = []
   print "level", level
   for node in nodes:
      print node.strfunc, ":",len(node.values), "training points are here"
      if node.left:
         newnodes.append(node.left)
      if node.right:
         newnodes.append(node.right)
   nodes = newnodes
   level += 1

errcnt = 0.0
for i in range(len(traindata)):
   curr = root
   while not curr.strfunc == "return":
      if curr.function(traindata[i]):
         curr = curr.left
      else:
         curr = curr.right
   if not curr.function() == trainlabels[i]:
      print curr.function()
      errcnt += 1

trainerror = errcnt / len(traindata)
print "Training Error:", trainerror

test = np.loadtxt("hw3test.txt", dtype = float)
testdata = test[:, 0:-1]
testlabels = np.array(test[:,-1], dtype = int)

errcnt = 0.0
for i in range(len(testdata)):
   curr = root
   while not curr.strfunc == "return":
      if curr.function(testdata[i]):
         curr = curr.left
      else:
         curr = curr.right
   if not curr.function() == testlabels[i]:
      errcnt += 1

testerror = errcnt / len(testdata)

print "Test Error:", testerror


    
    
    
    
    
        