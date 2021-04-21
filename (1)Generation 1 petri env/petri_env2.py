# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 09:23:26 2021

@author: isofi
"""
#%% Import libreries
import pandas as pd
import numpy as np
import random
import copy
import itertools
from itertools import product 


#%% initiation 
class Petri_Dyna :
    
    def __init__(self,exp_rate=0.3, lr=0.1, n_steps=5, episodes=1):
        
        
        while True:
            try : NPLACES=int(input ("Total Number of Places in the Petri : "))
            except:print (" oops Thats not a valid input ")
            else :break
        while True:
            try :NTRANSITION=int(input ("Total Number of transition in the Petri :"))      
            except: print (" oops Thats not a valid input ")
            else :break
        
  
        self.PLACES_NAMES=[]
        self.PLACES_OBJECTS=[]
        self.NPLACES=NPLACES
        self.NTRANSITION=NTRANSITION

        self.DP=pd.DataFrame(np.zeros(shape=((NTRANSITION,NPLACES))))
        self.DM=pd.DataFrame(np.zeros(shape=((NTRANSITION,NPLACES))))
        self.D=pd.DataFrame(np.zeros(shape=((NTRANSITION,NPLACES))))
        
        self.lr = lr
        self.steps = n_steps
        self.episodes = episodes  
        self.exp_rate = exp_rate
        self.reward=0
        
        self.goal=200
        self.delivered=0
        self.state =[]
        self.Next_state=[]
        self.state_space=[]
        
        self.action_space=[]
        self.state_action_space=[]
        self.state_space =list(itertools.product((-4,-3,-2,-1,0,1,2,3,4), repeat = self.NPLACES))
        self.Q_tabel=pd.DataFrame(np.zeros((len(self.state_space), self.NTRANSITION)))
        #self.Q_tabel.insert(loc=0, column="State", value=self.state_space, allow_duplicates=False)

#%% Core       

    class Place:    
        def __init__(self,name,ID,token,In_arcs,Out_arcs):
            
            self.pname =name
            self.pid=ID
            self.token=token
            self.in_tran=In_arcs
            self.out_tran=Out_arcs                         
       
        def __str__(self):
            return(f"The defined place {self.pname} : number of initial tokens: {self.token}\
                    input:{self.in_tran},output{self.out_tran}" )
                    
 
    def def_place (self):
        

        in_arcs=[]
        out_arcs=[]   
        
        ID= int (input(" Place ID: ").upper())
        name="P"+str(ID)
        token= int( input(f"Initial number of token in the place {name}: "))    
        for i in range (0,int(input (f"Number of input transition arcs to {name}  : "))):
            in_arcs.append(input(f" input arc number {i+1} ").upper())        
            for i in range (0,int(input (f"Number of output transition arcs to {name}  : "))):
                out_arcs.append(input(f" output arc number {i+1} ").upper())
                
        self.PLACES_OBJECTS.append(self.Place(name,ID, token, in_arcs, out_arcs))
        self.state.append(token)
        if name not in self.PLACES_NAMES : self.PLACES_NAMES.append(name)
        
        for i in range(len(in_arcs)) : 
            if in_arcs[i] not in self.action_space:
                self.action_space.append(in_arcs[i])
                
            try :self.DP.loc[int(in_arcs[i][-1])-1,ID-1]=1  
            except IndexError : 
                print("Erro creating the matrix ")
                
                          

        for i in range(len(out_arcs)) : 
            if out_arcs[i] not in self.action_space:
                self.action_space.append(out_arcs[i])
                
            try :self.DM.loc[int(out_arcs[i][-1])-1,ID-1]=1
            except IndexError : 
                print("Erro creating the matrix ")

            
        self.D=self.DP-self.DM
  
    def Petri_definition(self):
        
        for i in range (self.NPLACES):
            self.def_place() 

   
                
    def possible_firing(self) :
        
        situation=[True]*(self.NTRANSITION)  
        
        for i in range (self.NTRANSITION):     
            transition_array =np.zeros(self.NTRANSITION)
            transition_array[i]=1  
            
            print(transition_array)
  
            out=transition_array.dot(self.D.values)+self.state
            out=1

            for j in range (len(out)):
                if out[j]<0:
                    situation[i]=False             
        return situation
    
    
    def firing (self,Transition_ID):
        
        global Next_state

        current_state=np.array(self.state)
        transition_array =np.zeros(len(self.action_space))
        transition_array[Transition_ID]+=1
        Next_state=transition_array.dot(self.D.values)+current_state
        
        return (transition_array.dot(self.D.values)+current_state)

        
    def Reward(self):
        for i in range (self.NPLACES):
            if self.state[i]<0 : 
                return -100      
        if self.delivered>=self.goal:
            return +1000
        else :
            return -1

        


    def chooseAction(self):
        # epsilon-greedy
      
  
        if random.uniform(0, 1) <  self.exp_rate:
            action = random.sample(self.action_space, 1)[0] # Explore action space
            print(f" random action is {action}")                 
        else:
            # greedy action
            current_position = self.state_space.index(tuple (self.state))

            # if all actions have same value, then select randomly
            if len (set (agent.Q_tabel.T[current_position]))==1:
                action = random.sample(self.action_space, 1)[0]
                print(f" random action for same  is {action}")
                
            else:
                action = self.action_space[np.argmax(self.Q_tabel.T[current_position])]
                print(f" optimal action is {action}")  
                
        return (int (action[-1])-1)
    
    def Train  (self):
        
        self.steps_per_episode = []     
        for ep in range(self.episodes): 
            
            while  self.delivered<self.goal:
                
                action = self.chooseAction()
                self.state_action_space.append((self.state, action))      
  
                self.Next_state=(self.firing(action))
                
                if all([self.Next_state[i] >= 0 for i in range(self.NPLACES)]):    
                    self.state= copy.copy(self.Next_state)
                    print ("New State ! ")  
                    if self.state[-1]>2:
                        self.delivered+=1
                        self.state[-1]=1
                else :print ("firing Halted ")
                

                current_position = self.state_space.index(tuple (self.state))
                next_position=self.state_space.index(tuple (self.Next_state))
                self.reward=self.Reward()
                
                print (self.reward)

                
                self.Q_tabel[action][current_position] += self.lr*(self.reward + np.max(list(self.Q_tabel.iloc[next_position].values)) - self.Q_tabel[action][current_position])
                
                
                

agent=Petri_Dyna()

#%% Testing and debuging 
#agent=DynaAgent()

#agent.Petri_definition()

#agent.delivered=0
#agent.Train()


#action=0
#Next_state=[2,1,1,2]

#agent.Q_tabel[0][0]=10

#next_position=agent.state_space.index(tuple (Next_state))


#print(np.max(list(agent.Q_tabel.iloc[0].values)))

#print =np.max(list(agent.Q_tabel.iloc[next_position].values()))


#print ((agent.Q_tabel[next_position].values()))

#print (np.max(list(agent.Q_tabel.iloc[282].values)))


#print(agent.state_space.index(tuple (agent.state)))
#print(agent.state_space.index(tuple (agent.Next_state)))


print(agent.Q_tabel.iloc[4029])


