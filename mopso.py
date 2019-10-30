try:
    import numpy as np
except ImportError:
    raise ImportError,"The numpy module is required to run this program."
    
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError,"The matplotlib.pyplot module is required to run this program."
    
try:
    import random
except ImportError:
    raise ImportError,"The random module is required to run this program."
    
try:
    import time
except ImportError:
    raise ImportError,"The time module is required to run this program."

#:::: This module is used only for the ZDT3 objective function ::::#
try:
    import math
except ImportError:
    raise ImportError,"The math module is required to run this program."



#:::::::::::::::::::::#
#:::: MOPSO class ::::#
#:::::::::::::::::::::#
class MOPSO():
    def __init__(self):
        #:::: MOPSO parameters ::::#
        nPOP = 12
        nGEN = 350
        nDiv = 30
        nSim = 0
        maxSim = nPOP * nGEN
        c1 = 2.8
        c2 = 1.3
        nRep = 100
        
        #:::: Features of the optimization problem ::::#     
        nOF = 2   
        nOV = 30
        
        upB = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        lowB = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stepOV = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        nStep = (upB - lowB)/stepOV
        vmax = 2*(upB - lowB)/nStep

        #:::::::::::::::::::::::::::::::::::::#
        #:::: Initialization of the swarm ::::#
        #:::::::::::::::::::::::::::::::::::::#
    
        #:::: Initialize the speed of each particle VEL[i] = 0 ::::#
        V = np.zeros((nPOP, nOV))
        X = np.zeros((nPOP, nOV))
        PXbest = np.zeros((nPOP, nOV))
        PFbest = np.zeros((nPOP, nOF))
        
        #:::: Initialize population ::::#
        for i in xrange(0, nPOP):
            X[i,:] = initPOP(lowB, upB, stepOV, nOV)
        
        #:::::::::::::::::::::::::::::::::::::::#
        #:::: First evaluation of the swarm ::::#
        #:::::::::::::::::::::::::::::::::::::::#
        
        #:::: Calculate the computing time  ::::%
        timeStart = time.clock()
        
        #:::: Evaluate each particle in the swarm ::::#
        F = np.zeros(shape=(nPOP, nOF))
        for i in xrange(0,nPOP):
            F[i,:] = costFunction(X[i,:])
            #::: Initialize the momory of each particle ::::#
            PXbest[i,:] = X[i,:]
            PFbest[i,:] = F[i,:]
        
        #:::: Store the particles' positions that represent non-dominated vectors ::::#
        #:::: in an archive, also called the repository ::::#
        RepX, RepF = nondominatedVectors(X, F)
        
        #:::: Generate hypercubes of the search space explored so far, and locate the ::::# 
        #:::: particles using these hypercubes as a coordinate system where each ::::# 
        #:::: particle’s coordinates are defined according to the values of its objective function ::::#
        grid = generateHypercubes(RepF, nDiv)
        index, subindices = gridIndices(RepF, grid, nDiv)
        
        nSim = nSim + nPOP
        
        print "Generation: " + str(nSim/nPOP) + ", that is to say " + str(nSim) + " simulations"
        
        #:::::::::::::::::::::::::::::::::::::::::::::::::::#
        #:::: Here is the portion that makes MOPSO work ::::#
        #:::::::::::::::::::::::::::::::::::::::::::::::::::#
        
        #:::: WHILE maximum number of cycles has not been reached do ::::#
        while (nSim < maxSim):
            #:::: Linearly decreasing inertia weight ::::#
            w = 0.9 - ((0.9 - 0.4)/maxSim)*nSim 
            
            #:::: Compute the speed of each particle ::::#
            #:::: Compute the new position of each particle by adding the speed ::::#
            for i in xrange(0, nPOP):
                #:::: Rep[h] is a value that is taken from the repository ::::#
                h = selectH(RepF, index)
                V[i,:] = w*V[i,:] + random.random()*c1*(PXbest[i,:] - X[i,:]) + random.random()*c2*(RepX[h,:] - X[i,:]);
                X[i,:] = X[i,:] + V[i,:]
        
                #:::: Check that the particles do not fly out of the search space ::::#
                X[i,X[i,:] < lowB] = lowB[X[i,:] < lowB]
                X[i,X[i,:] > upB] = upB[X[i,:] > upB]
                #:::: Change the velocity direction ::::#
                V[i,X[i,:] < lowB] = -V[i,X[i,:] < lowB]
                V[i,X[i,:] > upB] = -V[i,X[i,:] > upB]
                #:::: Constrain the velocity ::::#
                V[i, V[i,:] > vmax] = vmax[V[i,:] > vmax]
                V[i, V[i,:] < -vmax] = vmax[V[i,:] < -vmax]

                
            #:::: Evaluate each particle in the swarm ::::#
            for i in xrange(0,nPOP):
                F[i,:] = costFunction(X[i,:])
                #:::: When the current position of the particle is better than the ::::#
                #:::: position contained in its memory, the particle’s position is updated ::::#
                if (all(F[i,:] <= PFbest[i,:]) & any(F[i,:] < PFbest[i,:])):
                    PFbest[i,:] = F[i,:]
                    PXbest[i,:] = X[i,:]
                elif (all(PFbest[i,:] <= F[i,:]) & any(PFbest[i,:] < F[i,:])):
                    #:::: Do nothing ::::#
                    #:::: If the current position is dominated by the position in memory, ::::#
                    #:::: then the position in memory is kept ::::#
                    pass
                else:
                    #:::: If neither of them is dominated by the other, then we select ::::# 
                    #:::: one of them randomly ::::#
                    if (random.randint(0,1) == 0):
                        PFbest[i,:] = F[i,:]
                        PXbest[i,:] = X[i,:]
            
            #:::: Update the contents of Rep together with the geographical representation ::::#
            #:::: of the particles within the hypercubes ::::#
            newRepX, newRepF = nondominatedVectors(X, F)
            
            newIndex, newSubindices = gridIndices(newRepF, grid, nDiv)
        
            RepX, RepF, index, subindices = archiveController(RepX, RepF, newRepX, newRepF, index, subindices, newIndex, newSubindices);
  
            #:::: if the individual inserted into the external population lies outside ::::#
            #:::: the current bounds of the grid, then the grid has to be recalculated ::::#
            #:::: and each individual within it has to be relocated ::::#
            if (sum(sum((grid[:,-2] > RepF).astype(int))) >= 1  & sum(sum((grid[:,1] < RepF).astype(int))) >= 1):
                grid = generateHypercubes(RepF, nDiv)
                index, subindices = gridIndices(RepF, grid, nDiv)
            
            #:::: If the external population has reached its maximum allowable ::::#
            #:::: capacity, then the adaptive grid procedure is invoked ::::#
            if (len(RepF[:,0]) > nRep):
                RepF, RepX, index, subindices = removeParticles(RepF, RepX, nRep, index, subindices)

            nSim = nSim + nPOP

            gen_no=nSim/nPOP
        
            print "Generation: " + str(gen_no) + ", that is to say " + str(nSim) + " simulations"
            plt.plot(RepF[:,0], RepF[:,1], 'ro')
            plt.xlabel('$f_{1}$')
            plt.ylabel('$f_{2}$')    
            plt.savefig("generations/"+str(nSim/nPOP)+".png")
            plt.close()
        
        #:::: Calculate the computing time  ::::%
        totalTime = time.clock() - timeStart
        
        print "MOPSO lasted " + str(totalTime)

        #::::::::::::::::::::::::::::::::#
        #:::: End of MOPSO - results ::::#
        #::::::::::::::::::::::::::::::::#

        plt.plot(RepF[:,0], RepF[:,1], 'ro')
        plt.xlabel('$f_{1}$')
        plt.ylabel('$f_{2}$')    
        plt.savefig("generations/end.png")
        plt.close()
        
        

#::::::::::::::::::::::::::::::#
#:::: Functions being used ::::#
#::::::::::::::::::::::::::::::#

def initPOP(lowB, upB, stepDV, nDV):
    nStep = []
    valueDV = []
    initialPar = []
    for it in xrange(0,nDV):
        nStep.append((upB[it] - lowB[it])/stepDV[it] + 1)
    
    for it in xrange(0,nDV):
        valueDV.append(np.linspace(lowB[it], upB[it], nStep[it]))
    
    for it in xrange(0,nDV):
        initialPar.append(valueDV[it][random.randint(0,nStep[it]-1)])
    
    return initialPar
    
def costFunction(X):
    #:::: ZDT1 ::::#
    n = X.shape[0]
    G  = 1 + (9*(np.sum(X) - X[0])/(n-1)) # or G  = 1 + 9*(np.sum(X[2:n]))/(n-1)
    F1 = X[0]
    F2 = G*(1 - np.sqrt(np.divide(X[0],G)))
    F = np.array([F1, F2])

    #:::: ZDT3 ::::#
    #n = X.shape[0]
    #G  = 1 + 9*(np.sum(X[2:n]))/(n-1)
    #F2 = G*(1 - np.sqrt(np.divide(X[0],G)) - np.divide(X[0],G)*math.sin(10*math.pi*X[0]))
    #F1 = X[0]
    #F = np.array([F1, F2])
    
    return F
    
def nondominatedVectors(X, F):
    X = np.array(X)
    F = np.array(F)
    nF = len(F[:,0])
    nD = np.zeros((nF,1))
    index = []
    
    for i in xrange(0,nF):
        nD[i][0] = 0
        for j in xrange(0,nF):
            if j != i:
                if (all(F[j,:] <= F[i,:]) & any(F[j,:] < F[i,:])):
                    nD[i][0] = nD[i][0] + 1
        if nD[i][0] == 0:
            index.append(i)
    
    repX = X[index,:]
    repF = F[index,:]
    
    it = 0
    nMax = len(repX[:,0])
    while (it < nMax):
        uIdx = sum((2 == (0 == (repX - repX[it,:])).sum(1)).astype(int)) 
        if uIdx > 1:
            repX = np.delete(repX, it, axis = 0)
            repF = np.delete(repF, it, axis = 0)
            nMax = len(repX[:,0])
            it = 0
        else:
            it = it + 1
            nMax = len(repX[:,0])
        
    return repX, repF
    
def generateHypercubes(F, nDiv):
    grid = np.zeros((0, nDiv))
    for i in xrange(0, len(F[0,:])):
        grid = np.vstack((grid, np.append(np.append(-np.inf, np.linspace(F[np.argmin(F[:,i]),i], F[np.argmax(F[:,i]),i], num = nDiv-2)), np.inf)))
            
    return grid

def gridIndices(F, grid, nDiv):
    nOFs = len(F[0,:]) 
    nFs = len(F[:,0])    

    subIndices = []
    for i in xrange(0, nFs):
        subIdx = []
        for j in xrange(0, nOFs):
            subIdx = np.append(subIdx, min(np.where(F[i,j] <= grid[j,:])[0]))
        subIndices.append(subIdx)
    
    subIndices = np.array(subIndices)
    
    coordinates = []
    for i in xrange(0, nFs):
        coordinates.append(tuple([int(coord) for coord in  subIndices[i,:]]))
    
    coordinates = tuple(coordinates)
    
    dimension = []
    for i in xrange(0, nOFs):
        dimension.append(len(grid[i,:]))
    
    dimension = tuple(dimension)
    
    index = []
    for i in xrange(0, nFs):
        index.append(np.ravel_multi_index(coordinates[i], dims = dimension, order = 'C'))
    
    return index, subIndices

def selectH(RepF, index):
    x = 10
    uniqueIdx = np.zeros((len(index),2))
    for i in xrange(0, len(index)):
        nIdx = np.array(index == index[i])
        nIdx = nIdx.astype(np.int)
        nIdx = nIdx.sum(axis = 0)
        uniqueIdx[i,:] = np.array(np.hstack((nIdx, i)))
    
    fitness = np.zeros((len(RepF[:,0]),1))
    for i in xrange(0, len(RepF[:,0])):
        fitness[i] = x/uniqueIdx[i,0]
        
    rouletteWheel = np.cumsum(fitness, axis = 0)
    finalIdx = np.where(((rouletteWheel.max() - rouletteWheel.min())*random.random()) <= rouletteWheel)[0]
    h = finalIdx[0]
    
    return h

def deleteH(RepF, index):
    x = 10
    uniqueIdx = np.zeros((len(index),2))
    for i in xrange(0, len(index)):
        nIdx = np.array(index == index[i])
        nIdx = nIdx.astype(np.int)
        nIdx = nIdx.sum(axis = 0)
        uniqueIdx[i,:] = np.array(np.hstack((nIdx, i)))
    
    fitness = np.zeros((len(RepF[:,0]),1))
    for i in xrange(0, len(RepF[:,0])):
        fitness[i] = x*uniqueIdx[i,0]
        
    rouletteWheel = np.cumsum(fitness, axis = 0)
    finalIdx = np.where(((rouletteWheel.max() - rouletteWheel.min())*random.random()) <= rouletteWheel)[0]
    h = finalIdx[0]
    
    return h
    
def archiveController(X, F, arcX, arcF, idxRep, subIdx, newIdxRep, newSubIdx):
    X = np.concatenate((X, arcX), axis = 0)    
    F = np.concatenate((F, arcF), axis = 0)  
    idxR = np.concatenate((idxRep, newIdxRep), axis = 0)  
    subidxR = np.concatenate((subIdx, newSubIdx), axis = 0)  
    nF = len(F[:,0])
    nD = np.zeros((nF,1))
    index = []
    
    for i in xrange(0,nF):
        nD[i,0] = 0
        for j in xrange(0,nF):
            if j != i:
                if (all(F[j,:] <= F[i,:]) & any(F[j,:] < F[i,:])):
                    nD[i,0] = nD[i,0] + 1
        if nD[i,0] == 0:
            index.append(i)
    
    RepX = X[index,:]
    RepF = F[index,:]
    indexRep = idxR[index]
    subindicesRep = subidxR[index,:]    

    it = 0
    nMax = len(RepX[:,0])
    while (it < nMax):
        uIdx = sum((2 == (0 == (RepX - RepX[it,:])).sum(1)).astype(int)) 
        if uIdx > 1:
            RepX = np.delete(RepX, it, axis = 0)
            RepF = np.delete(RepF, it, axis = 0)
            indexRep = np.delete(indexRep, it, axis = 0)
            subindicesRep = np.delete(subindicesRep, it, axis = 0)   
            nMax = len(RepX[:,0])
            it = 0
        else:
            it = it + 1
            nMax = len(RepX[:,0])
            
    return RepX, RepF, indexRep, subindicesRep
    
def removeParticles(RepF, RepX, nRep, index, subindices):
    h = np.zeros((nRep,1))
    for i in xrange(0, nRep):
        h[i,0] = deleteH(RepF, index)
        RepF = np.delete(RepF, h[i,0], axis = 0)
        RepX = np.delete(RepX, h[i,0], axis = 0)
        index = np.delete(index, h[i,0], axis = 0)
        subindices = np.delete(subindices, h[i,0], axis = 0)
    
    return RepF, RepX, index, subindices
    
    
def main():
    MOPSO()

if __name__ == '__main__':
    main()
