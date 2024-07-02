import numpy as np
import scipy as sp
import scipy as sp
from matplotlib import pyplot as plt
import time



#Cloning params
N=1000
T = 1000

s_array=  np.arange(-3,3.5,0.5)

#model params
alpha_R=1 #site empty: right boundary
alpha_L=2#site empty: left boundary
beta_L=1 #site occupied: left boundary
beta_R=0.6 #site occupied: right boundary



##config update function

def update_x(x_state,tau_min):
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
            bias= 1
            
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
                bias= -1
    
    x_state=1-x_state   #flip config  
    return(x_state,bias)



scgf=[]

###########################################Distributions costructed with standard scipy library

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
        

def update_t(clone,dist0,dist1):
             #print("chosen",clone)
             if clone[1]==0:
               tau_new = dist0.rvs(size = 1)
               clone[2] +=tau_new
               clone[3] =tau_new 
             else: 
                 tau_new = dist1.rvs(size = 1)
                 clone[2] +=tau_new
                 clone[3] =tau_new 
                    
             return clone
            
#################################Cloning!            
def main():
    
    ######## looping over s-values
    for s in s_array:
        #set-up clones
        t_0=np.random.rand(N)    # some initial_time
        clone_state= np.zeros((N,4))
        clone_state[:,1]=np.random.choice([0,1],N) # state
        clone_state[:,2]=t_0 # time
        clone_state[:,3]=np.zeros(N) # waiting_time tau
        clone_state[:,0]=t_0 # initial_time
        C=0
        
        k=0
        dist0 = Dist0()
        dist1 = Dist1()
        samples0 = np.array(dist0.rvs(size=N))
        samples1 = np.array(dist1.rvs(size=N))
    
        ##############  step1
        
        clone_state[:,3][clone_state[:,1]==0]=samples0[clone_state[:,1]==0]
        clone_state[:,3][clone_state[:,1]==1]=samples1[clone_state[:,1]==1]
        clone_state[:,2] +=  clone_state[:,3]
        
        min_indx =np.argmin(clone_state[:,2])
        
    
        ti = time.time()
        tii = time.time()
        t_ = clone_state[min_indx,2]-clone_state[min_indx,0]
        ##cloning algorith: main Loop
        while t_ <T:
            
            
            #step2 & 3: Update configuration and time for chosen clone
            
           
            clone_state[min_indx,1],bias=update_x(clone_state[min_indx,1],clone_state[min_indx,3]) #config update  
            clone_state[min_indx] = update_t(clone_state[min_indx],dist0,dist1)  # time update
            
           
            
            
            ##############Step4: cloning step
           
            y=np.floor(np.exp(-s*bias)+np.random.uniform(0,1)) 
           
            c=0
            
            if int(y)==0:
         
               c=np.random.randint(N-1,size=1)
               if c>=min_indx:
                  c += 1   
               clone_state[min_indx]=clone_state[c]
            
               #in case time to be updated after cloning
               #if clone_state[min_indx,1]==0:
                #  clone_state[min_indx,3] += dist0.rvs(1)
                 # clone_state[min_indx,2] += clone_state[min_indx,3] 
               #else: 
                #   clone_state[min_indx,3] += dist1.rvs(1)
                 #  clone_state[min_indx,2] += clone_state[min_indx,3]
               
            else:
                x_state= clone_state[min_indx,2]         
                clones = np.repeat(np.array([clone_state[min_indx]]),int(y),axis=0) #copy clones
                
                # in case time is updated after cloning
                #if x_state ==0:
                 #  new_tau = dist0.rvs(size=int(y))
                  # clones[:,3] = new_tau                         
                   #clones[:,2] += new_tau                        
                #else:
                 #   new_tau = dist1.rvs(size=int(y))
                  #  clones[:,3] = new_tau
                   # clones[:,2] += new_tau
                    
                clone_state = np.append(clone_state,clones,axis=0) #add copies to ensemble
                c = np.random.choice(N+int(y),size=int(y),replace=False)
                clone_state=  np.delete(clone_state,c,axis=0) # delete to maintain N clones
                

        
            ###step5 update cloning factor 
            C += np.log((N+np.exp(-s*bias)-1)/N)
            
            ## choose next clone
            min_indx=np.argmin(clone_state[:,2])
            
            t_ = clone_state[min_indx,2]-clone_state[min_indx,0]
            #print(t_,clone_state[min_indx,2],clone_state[min_indx,0])
            if k%1000==0:
               #compute time per-1000 steps 
               tff = time.time()  
               print(f"--------------step--------------------,sim_time:{t_}, scgf_update:{C/t_},bias:{s},run_time:{(tff-tii)}")
               tii = time.time()   
            k+=1
        
        tf =time.time()
        tt=tf-ti
        ##storing scgf
        scgf.append(C/t_)
        print(f"------------------------------scgf-----------,scgf: {C/t_}, run_time:{tt},bias:{s}")
        #print(-C/T)
    
    #save data and plot
    np.save("scgf_cloning_paper.npy",np.array(scgf))    
    print("SCGF",scgf)            
    plt.plot(s_array,scgf)
    plt.show()

if __name__=='__main__':
     main()
    
