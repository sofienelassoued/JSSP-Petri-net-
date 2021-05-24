# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:58:25 2021

@author: isofi
"""

#%% Import libreries 

import pandas as pd
import numpy as np
import random
import copy
import itertools
from itertools import product 
import bisect



#%% Petri_net class 

class Petri_env:
    
    def __init__(self,NPLACES,NTRANSITION):
        
        self.state=[]
        self.action_space=[]
        
        self.PLACES_NAMES=[]
        self.PLACES_OBJECTS=[]
        self.NPLACES=NPLACES
        self.NTRANSITION=NTRANSITION

        self.DP=pd.DataFrame(np.zeros(shape=((NTRANSITION,NPLACES))))
        self.DM=pd.DataFrame(np.zeros(shape=((NTRANSITION,NPLACES))))
        self.D=pd.DataFrame(np.zeros(shape=((NTRANSITION,NPLACES))))
        
        self.delivered=0
        
        
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
                bisect.insort(self.action_space, in_arcs[i])

                
            try :self.DP.loc[int(in_arcs[i][-1])-1,ID-1]=1  
            except IndexError : 
                print("Erro creating the matrix ")
                
                          

        for i in range(len(out_arcs)) : 
            if out_arcs[i] not in self.action_space:
                bisect.insort(self.action_space, out_arcs[i])
                
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
    
    
    def firing (self,Transition):
        
        delivery=False
        current_state=np.array(self.state)

        Transition_ID =self.action_space.index(Transition)
        transition_array =np.zeros(len(self.action_space))
        transition_array[Transition_ID]+=1
        Next_state=transition_array.dot(self.D.values)+current_state
        
        if all([Next_state[i] >= 0 for i in range(self.NPLACES)]):
        
            if Next_state[-1]>1:
                self.delivered+=1
                Next_state[-1]=1
                delivery=True
            return (Next_state)
        
        else : 
            print ("firing Halted ")
            return (current_state)
            

        
#%% DynQ Class    
             
class DynaQ :
    
    def __init__(self,petri_env,exp_rate=0.1, lr=0.3, n_steps=5, episodes=10):
      
        self.env=petri_env
        self.lr = lr
        self.steps = n_steps
        self.episodes = episodes  
        self.exp_rate = exp_rate
                 
        self.goal=200
        self.terminal=False
         
        self.state = petri_env.state
        self.actions=petri_env.action_space
        self.state_actions=[]

        self.state_space =list(itertools.product((range(0,100)), repeat = petri_env.NPLACES))
        
        self.model = {}
        self.Q_values={}
        for i in range (len(self.state_space)):        
          self.Q_values[self.state_space[i]]={}
          for a in petri_env.action_space:
              self.Q_values[self.state_space[i]][a] = 0
              
    def Reward(self,Next_state):
        
        
        positive_state =all([Next_state[i] >= 0 for i in range(self.env.NPLACES)]) 
        
        # update the delivered based on the new state
        
        if self.env.delivered>=self.goal: 
            
            self.terminal=True
            reward=+1000

        elif positive_state == False :
            reward=-100
            
        #elif Delivery==False:
            #reward=-1
            
        else : reward=-1
                
        return reward
            
  
        
    def chooseAction(self):
        # epsilon-greedy
        action = ""
        
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
            print (f"Action choosed with epsilon Greedy {action}")
            
        else:
            # greedy action
            current_position = tuple(self.state)
            # if all actions have same value, then select randomly
            if len(set(self.Q_values[current_position].values())) == 1:
                action = np.random.choice(self.actions)
                print (f" Action choosed for same Q values {action}")
                
            
            else:
                for a in self.actions:
                    
                    Qvalues=list(self.Q_values[current_position].values())
                    action=self.actions[Qvalues.index(max(Qvalues))]
                print (f" Action choosed for Optimal policy {action}")
        return(action)
    
    
    def Train(self):       
        self.steps_per_episode = [] 

        for ep in range(self.episodes):  
            
            self.terminal=False
            self.env.delivered=0
            
            print(f"-------Episode Number {ep}-----------------")
            while not self.terminal:
                
                action = self.chooseAction()
                self.state_actions.append((self.env.state, action))
                
                Next_state=self.env.firing(action)
                reward=self.Reward(Next_state)
                self.Q_values[tuple(self.env.state)][action] += self.lr*(reward + np.max(list(self.Q_values[tuple (Next_state)].values())) - self.Q_values[tuple(self.env.state)][action])
                
                # update model
                
                if tuple(self.env.state) not in self.model.keys():
                    self.model[tuple(self.env.state)] = {}
                    
                self.model[tuple(self.env.state)][action] = (Next_state,reward)           
                print (f"(current State {self.env.state}, Next_state {Next_state} collected reward {reward}))" )
               
                self.env.state = Next_state
                

         
#%% Initiation          
         
while True:
    try : NPLACES=int(input ("Total Number of Places in the Petri : "))
    except:print (" oops Thats not a valid input ")
    else :break
while True:
    try :NTRANSITION=int(input ("Total Number of transition in the Petri :"))      
    except: print (" oops Thats not a valid input ")
    else :break   
      
env=Petri_env(NPLACES,NTRANSITION) 
env.Petri_definition()   
agent=DynaQ(env)

#%% reset 

#saved_env=copy.deepcopy(env)
#agent=DynaQ(env)
#agent.env.state=[2,2,2,1]

#%% Main loop

#agent.env.state=[1,1,1,1,1,1,1]

agent=DynaQ(env)

#agent.Train()





    
    

        
