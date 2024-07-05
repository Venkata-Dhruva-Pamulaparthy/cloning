import numpy as np
from random import choice, sample
import scipy as sp
# heapq provides the Heap Queue structure which allows to manipulate efficiently sorted list.
# Here we use a list pof copies of the system, sorted according to their next time of evolution.
import heapq 
import time

# procedure to remove a element of index i from a heap, keeping the heap structure intact

#model params
alpha_R=1 #site empty: right boundary
alpha_L=2#site empty: left boundary
beta_L=1 #site occupied: left boundary
beta_R=0.6 #site occupied: right boundary



def heapq_remove(heap, index):
    # Move slot to be removed to top of heap
    while index > 0:
        up = int((index + 1) / 2 - 1)
        heap[index] = heap[up]
        index = up
    # Remove top of heap and restore heap property
    heapq.heappop(heap)




def bias(x_state,tau_min):
    bias=0
 
    if x_state==0:
        #print('here')
        psi_L = (alpha_L**2) * tau_min * np.exp(-alpha_L*tau_min) #gamma(2,lamda)
        psi_R = (alpha_R**2) * tau_min * np.exp(-alpha_R*tau_min)
        a  = psi_R/ sp.special.gammaincc(2,alpha_R*tau_min)   # upper_incomplete_gamma(2,lamda)
        b  = psi_L/ sp.special.gammaincc(2,alpha_L*tau_min)
        p     =    a/(a+b)
        j    = np.random.choice(range(2),1,False,[p,1-p])
        #print(p)
        if j==0 :
            bias= -1
            
    else:
           #print('or here')
           psi_L = (beta_L**2) * tau_min * np.exp(-beta_L*tau_min)
           psi_R = (beta_R**2) * tau_min * np.exp(-beta_R*tau_min)
           a  = psi_R/ sp.special.gammaincc(2,beta_R*tau_min)
           b  = psi_L/ sp.special.gammaincc(2,beta_L*tau_min)
           p     =    a/(a+b)
           #print(p) 
           j    = np.random.choice(range(2),1,False,[p,1-p])
           if j==0:
                bias= 1
      
    return  bias
    
def flip(state):
    if state == 1:
        new_state = 0
    else:
        new_state = 1
    return new_state


def cloning_factor(s, theta, dt):
    return np.exp(theta * s)
    
class Dist0(sp.stats.rv_continuous):
       def __init__(self, xtol=1e-14, seed=None):
          super().__init__(a=0, xtol=xtol, seed=seed)
        
       def _cdf(self,t):
           return  1- (sp.special.gammaincc(2,alpha_L*t)*sp.special.gammaincc(2,alpha_R*t))
            
      
        
class Dist1(sp.stats.rv_continuous):
        
        def __init__(self, xtol=1e-14, seed=None):
          super().__init__(a=0, xtol=xtol, seed=seed)
        
        def _cdf(self,t):
            return 1 - (sp.special.gammaincc(2,beta_L*t) * sp.special.gammaincc(2,beta_R*t))
        

DIM_ENSEMBLE = 500
observation_time = 500.
dist0=Dist0()
dist1=Dist1()

def escape(state):
    if state == 1:
        return dist0.rvs(1)
    else:
        return dist1.rvs(1)


def main(s):

    t = 0
    C = 0.

    # initialization of the population
    # each clone in the ensemble is described by a 3-uple (time,dt,state) 
    #   time  = next time at which it will evolve
    #   dt = time since last evolution
    #   state = 0 or 1 = empty or occupied 

    ensemble = [(0.001, 0.001, 0.) for count in range(DIM_ENSEMBLE)]
    #print(ensemble)

    # orders the population into a Heap Queue
    heapq.heapify(ensemble)
    #k=0
    while t < observation_time:
        # we pop the first element of populat, which is always the next to evolve
        (t, dt, state) = heapq.heappop(ensemble)
        #if k%1000==0:
        #     print("....",t,C/(t-0.001))
        #k+=1    
        #new_state = flip(state)
        theta = bias(state,dt)
        
        Y = cloning_factor(s, theta, dt)
        # the copy we poped out is to be replaced by p copies
        y = int(np.floor(Y + np.random.random())) 

        if y == 0:
            # one copy chosen at random replaces the current copy
            to_clone = choice(ensemble)
            heapq.heappush(ensemble, to_clone)
        elif y == 1:
            # the current copy is evolved without cloning
            new_state = flip(state)

            #interval until next evolution
            Deltat = escape(new_state)
            to_clone = (t + Deltat, Deltat, new_state)
            heapq.heappush(ensemble, to_clone)
        else: # p>1 : make y clones ; population size becomes N+y-1 ; remove y-1 clones uniformly 
            pcount = y
            while pcount > 0:
                pcount -= 1
                new_state = flip(state)
                Deltat = escape(new_state) # interval until next evolution
                toclone = (t + Deltat, Deltat, new_state)
                heapq.heappush(ensemble, toclone)

            # we first chose uniformly the p-1 distinct indices to remove, among the N+p-1 indices 
            listsize = DIM_ENSEMBLE + y - 1
            indices = sample(range(listsize), y - 1) 
            # the list of indices to remove is sorted from largest to smallest, so as to remove largest indices first
            indices.sort(reverse = True)
            for i in indices:
                heapq_remove(ensemble, i)

        C += np.log((DIM_ENSEMBLE + Y - 1.) / DIM_ENSEMBLE)

    return C / (t-0.001)


if __name__ == '__main__':
  s_array=  np.arange(-3,3.5,0.5)
  scgf_heap=[] 
  print("start")  
  for s in s_array:
      ti = time.time()
      scgf=main(s)
      scgf_heap.append(scgf)
      tf=time.time()
      print("------------------",scgf,tf-ti)