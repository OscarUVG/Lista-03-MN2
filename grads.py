import numpy as np
import matplotlib.pyplot as plt
import itertools

def checkStopCrit(stopCrit, x1, x2, f, df, eps):
    e = 0
    if(stopCrit == 'abs_x'):
        e = np.linalg.norm(x2 - x1)
    elif(stopCrit == 'rel_x'):
        e = np.linalg.norm(x2 - x1)/np.linalg.norm(x1)
    elif(stopCrit == 'abs_f'):
        e = abs(f(x2) - f(x1))
    elif(stopCrit == 'rel_f'):
        e = abs((f(x2) - f(x1))/f(x1))
    elif(stopCrit == 'norm_df'):
        e = np.linalg.norm(df(x1))
    return e < eps, e
        
def descNaiveSteepest(f, df, x0, alpha, maxIter, eps, stopCrit):
    seq_x = [x0]
    seq_e = [-1]
    stop = False
    j = 0
    while(not stop and j < maxIter):
        xNext = seq_x[-1] - alpha * df(seq_x[-1])
        seq_x.append(xNext)
        stop, e = checkStopCrit(stopCrit, seq_x[-2], seq_x[-1], f, df, eps)
        seq_e.append(e)
        j += 1
    return j!=maxIter, seq_x, seq_e

def descNaiveRandom(f, df, x0, alpha, maxIter, eps, stopCrit):
    seq_x = [x0]
    seq_e = [-1]
    stop = False
    j = 0
    while(not stop and j < maxIter):
        
        dfx = df(seq_x[-1])
        dSelected = False
        while(not dSelected):
            d = np.random.uniform(-1,1,len(x0))
            if(np.dot(dfx,d) < 0): dSelected = True
        d = dfx / np.linalg.norm(d)
        
        xNext = seq_x[-1] + alpha * d
        seq_x.append(xNext)
        stop, e = checkStopCrit(stopCrit, seq_x[-2], seq_x[-1], f, df, eps)
        seq_e.append(e)
        j += 1
    return j!=maxIter, seq_x, seq_e

# alpha=1e-2, c=1e-5, rho=0.9
def descBacktracking(f, df, x0, alpha, maxIter, eps, stopCrit, c, rho):
    seq_x = [x0]
    seq_e = [-1]
    stop = False
    j = 0
    while(not stop and j < maxIter):
        d = -df(seq_x[-1])
        alphaFound = False
        while(not alphaFound):
            if(f(seq_x[-1] + alpha*d) > f(seq_x[-1]) + c*alpha*(np.dot(df(seq_x[-1]),d))): alpha *= rho
            else: alphaFound = True
            
        xNext = seq_x[-1] - alpha * df(seq_x[-1])
        seq_x.append(xNext)
        stop, e = checkStopCrit(stopCrit, seq_x[-2], seq_x[-1], f, df, eps)
        seq_e.append(e)
        j += 1
    return j!=maxIter, seq_x, seq_e

# c=0.8, rho=0.9
def descNewtonExacto(f, df, ddf, x0, alpha, maxIter, eps, stopCrit, c, rho):
    seq_x = [x0]
    seq_e = [-1]
    stop = False
    j = 0
    while(not stop and j < maxIter):
        L, V = np.linalg.eig(ddf(seq_x[-1]))
        if(all([l>0 for l in L])):
            d = -np.linalg.solve(ddf(seq_x[-1]), df(seq_x[-1]))
        else:
            newL = np.heaviside(L,0) + 1e-2*np.heaviside(-L,0)
            d = -np.linalg.solve(V @ np.diag(newL) @ V.T, df(seq_x[-1]))
        
        alphaFound = False
        while(not alphaFound):
            if(f(seq_x[-1] + alpha*d) > f(seq_x[-1]) + c*alpha*(np.dot(df(seq_x[-1]),d))): alpha *= rho
            else: alphaFound = True
        
        xNext = seq_x[-1] + alpha * d
        seq_x.append(xNext)
        stop, e = checkStopCrit(stopCrit, seq_x[-2], seq_x[-1], f, df, eps)
        seq_e.append(e)
        j += 1
    return j!=maxIter, seq_x, seq_e


def test():
    f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    dfx0 = lambda x: 200*(x[1]-x[0]**2)*(-2*x[0]) - 2*(1-x[0])
    dfx1 = lambda x: 200*(x[1]-x[0]**2)
    df = lambda x: np.array([dfx0(x), dfx1(x)])
    dfx00 = lambda x: 600*x[0]**2 - 400*x[1] + 2
    dfx01 = lambda x: -400*x[0]
    dfx11 = lambda x: 200
    ddf = lambda x: np.matrix([[dfx00(x), dfx01(x)],
                              [dfx01(x), dfx11(x)]])
    
    g = lambda x: 100*(x[0]**2-x[1])**2 + (x[0]-1)**2 + (x[2]-1)**2 + 90*(x[2]**2-x[3])**2
    dgx0 = lambda x: 200*(x[0]**2-x[1])*(2*x[0]) + 2*(x[0]-1)
    dgx1 = lambda x: -200*(x[0]**2-x[1])
    dgx2 = lambda x: 2*(x[2]-1) + 180*(x[2]**2-x[3])*(2*x[2])
    dgx3 = lambda x: -180*(x[2]**2-x[3])
    dg = lambda x: np.array([dgx0(x), dgx1(x), dgx2(x), dgx3(x)])
    
    h = lambda x: np.array([100*(x[j+1] - x[j]**2)**2 + (1-x[j])**2 for j in range(99)]).sum()
    dhx0 = lambda x: [200*(x[1] - x[0]**2)*(-2*x[0]) - 2*(1-x[0])]
    dhx1_x98 = lambda x: [200*(x[j] - x[j-1]**2) + 200*(x[j+1] - x[j]**2)*(-2*x[j]) - 2*(1-x[j])  for j in range(1,99)]
    dhx99 = lambda x: [200*(x[99] - x[98]**2)]
    dh = lambda x: np.array(dhx0(x) + dhx1_x98(x) + dhx99(x))
    
    # --------------
    x0 = [-1.2,1]
    alpha = 1e-2
    maxIter = 100000
    eps = 1e-8
    
    conv, seq_x, seq_e  = descNewtonExacto(f, df, ddf, x0, alpha, maxIter, eps, 'abs_x', c=0.8, rho=0.9)
    print(conv, len(seq_x))
    print(seq_x[-1])
    print(f(seq_x[-1]))
    
    coords = np.linspace(-2,2,100)
    grid_x, grid_y = np.meshgrid(coords, coords)
    plt.contour(grid_x, grid_y, f([grid_x,grid_y]), levels=50)
    plt.plot(np.array(seq_x)[:,0], np.array(seq_x)[:,1], c='red')
    plt.show()

def gaussianas(sig=1):
    seq_mu = np.random.uniform(0,8,(8,2))
    f = lambda x: np.array([-np.exp(-np.linalg.norm(x-mu)**2/(2*sig)) for mu in seq_mu]).sum()
    df = lambda x: np.array([np.exp(-np.linalg.norm(x-mu)**2/(2*sig))*(x-mu)/sig for mu in seq_mu]).sum(axis=0)
    
    grid = np.linspace(0,8,200)
    z = [[f([x, y]) for x in grid] for y in grid]
    plt.contour(grid, grid, z, levels=30)
    
    cant_points = 12
    points = np.random.uniform(0.5,7.5,(cant_points, 2))
    plt.scatter(points[:,0], points[:,1], c='red')                       
    for point in points:
        conv, z, e = descNaiveSteepest(f, df, point, alpha=1e-1, maxIter=10000, eps=1e-10, stopCrit='abs_x')
        plt.plot(np.array(z)[:,0], np.array(z)[:,1], c='red', linestyle='dashed')
        plt.scatter(z[-1][0], z[-1][1], c='purple')
    
    plt.show()
    
    
#test()
gaussianas()
