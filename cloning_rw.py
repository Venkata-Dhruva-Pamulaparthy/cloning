import numpy as np
import scipy as sp
import matplotlib as plt

#from matplotlib import pyplot as plt
import time



#Cloning params
N=1000
T = 1000


s_array=  np.arange(-2,2.2,0.2)

#model params
alpha_R=0.6 #site empty: right boundary
alpha_L=0.4#site empty: left boundary
#beta_L=0.2 #site occupied: left boundary
#beta_R=0.4 #site occupied: right boundary



##config update function

def update_x(x_state,tau_min):
    bias=0
    psi_L = (alpha_L**2) * tau_min * np.exp(-alpha_L*tau_min) #gamma(2,lamda)
    psi_R = (alpha_R**2) * tau_min * np.exp(-alpha_R*tau_min)
    a  = psi_R/ sp.special.gammaincc(2,alpha_R*tau_min)   # upper_incomplete_gamma(2,lamda)
    b  = psi_L/ sp.special.gammaincc(2,alpha_L*tau_min)
    p     =    a/(a+b)
    j    = np.random.choice(range(2),1,False,[p,1-p])
        #print(p)
    if j==0 :
            bias= 1
            x_state += 1
            
    else:
            bias=  -1
            x_state -=1
           
    
  
    return(x_state%2,bias)



scgf=[]

###########################################Distributions costructed with standard scipy library

class Dist0(sp.stats.rv_continuous):
       def __init__(self, seed=None):
          super().__init__(a=0, seed=seed)
        
       def _cdf(self,t):
           return  1- (sp.special.gammaincc(2,alpha_L*t)*sp.special.gammaincc(2,alpha_R*t))
            
      

def update_t(clone,dist0):
             #print("chosen",clone)
             a,_= clone[2],clone[3]
             tau_new = dist0.rvs(size = 1)
             clone[2] = a + tau_new
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
        samples0 = np.array(dist0.rvs(size=N))
        #samples1 = np.array(dist1.rvs(size=N))
    
        ##############  step1
        
        clone_state[:,3] = samples0
        clone_state[:,2] +=  clone_state[:,3]
        
        min_indx =np.argmin(clone_state[:,2])
        
    
        ti = time.time()
        tii = time.time()
        t_ = clone_state[min_indx,2]-clone_state[min_indx,0]
        ##cloning algorith: main Loop
        while t_ <T:
            
            
            #step2 & 3: Update configuration and time for chosen clone
            
           
            clone_state[min_indx,1],bias=update_x(clone_state[min_indx,1],clone_state[min_indx,3]) #config update  
            clone_state[min_indx] = update_t(clone_state[min_indx], dist0)  # time update
            
           
            
            
            ##############Step4: cloning step
           
            y=np.floor(np.exp(s*bias)+np.random.uniform(0,1)) 
           
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
                #x_state= clone_state[min_indx,2]         
                clones = np.repeat(np.array([clone_state[min_indx]]),int(y)-1,axis=0) #copy clones
                   
                clone_state = np.append(clone_state,clones,axis=0) #add copies to ensemble
                c = np.random.choice(N+int(y)-1,size=int(y)-1,replace=False)
                    
                clone_state=  np.delete(clone_state,c,axis=0) # delete to maintain N clones
            

        
            ###step5 update cloning factor 
            C += np.log((N+np.exp(s*bias)-1)/N)
            
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
    