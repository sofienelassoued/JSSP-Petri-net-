# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import itertools
from itertools import product 

#%% Petri env 

path = "D:\Sciebo\Semester 4 (Project Thesis)\Git Projects\My petrinet simulator/petri1.html"

class Petri_env:
    
    def __init__(self,path):
    
        self.Forwards_incidence = pd.read_html(path,header=0,index_col=0)[1]
        self.Backwards_incidence= pd.read_html(path,header=0,index_col=0)[3]
        self.Combined_incidence = pd.read_html(path,header=0,index_col=0)[5]
        self.Inhibition_matrix = pd.read_html(path,header=0,index_col=0)[7]
        self.Marking =pd.read_html(path,header=0,index_col=0)[9]
            
        self.Places_names= self.Forwards_incidence.index.tolist()
        self.NPLACES=len( self.Places_names)
        self.Places_obj=[]
        
        self.Transition_names= self.Forwards_incidence.columns.tolist()   
        self.NTRANSITIONS=len( self.Transition_names)
        self.Transition_obj=[]
        
        self.state=self.Marking.loc["Current"]
        self.acion_space=self.Transition_obj
        self.summary=[]
        
        
        self.goal=20
        self.terminal=False
        
        
    def load_model(self):
        
        class Place:
            def __init__(self,name,token,In_arcs,Out_arcs):
            
               self.pname =name
               self.token=token
               self.in_tran=In_arcs
               self.out_tran=Out_arcs                         
       
            def __str__(self):
                return(f"Place name {self.pname}  Tokens: {self.token}  Input:{self.in_tran}  Output{self.out_tran}" )
                   
        class Transition:
            def __init__(self,name,time,In_arcs,Out_arcs):
            
                self.tname =name
                self.time=0
                self.in_tran=In_arcs
                self.out_tran=Out_arcs                         
       
            def __str__(self):
                return(f"Tansition name {self.tname}  Timer: {self.time}  Input:{self.in_tran}  Output{self.out_tran}" )
            
        for i in self.Places_names:  # outer loop for every place        
        
            In_arcs=[]
            Out_arcs=[]      
            name=i
            token=self.Marking.loc["Current"][i]
                
                
            for j in self.Forwards_incidence.columns.tolist() :      
                if self.Forwards_incidence.loc[i,j]==1:
                    In_arcs.append(j)
            
            for k in self.Backwards_incidence.columns.tolist() :      
                if self.Backwards_incidence.loc[i,k]==1:
                    Out_arcs.append(k)   
                         
            self.Places_obj.append(Place(name,token,In_arcs, Out_arcs))
            
            
        for i in self.Transition_names:  # outer loop for every transition        
           
            In_arcs=[]
            Out_arcs=[]         
            name=i
            time=0     
            
            for j in self.Forwards_incidence.index.tolist() :      
                if self.Forwards_incidence.loc[j,i]==1:
                    Out_arcs.append(j)
            
            for k in self.Backwards_incidence.index.tolist() :      
                if self.Backwards_incidence.loc[k,i]==1:
                    In_arcs.append(k)
                                 
            self.Transition_obj.append(Transition(name,time,In_arcs, Out_arcs))
               
    def possible_firing(self) :
        
        
        situation=[True]*(self.NTRANSITIONS)  
        
        for i in range (self.NTRANSITIONS):     
            transition_array =np.zeros(self.NTRANSITIONS)
            transition_array[i]=1            
            next_step=transition_array.dot(self.Combined_incidence.T.values)+self.state
            for j in range (len(next_step)):
                if next_step[j]<0:
                    situation[i]=False
                    
        self.summary = pd.DataFrame(situation,index=self.Transition_names,columns=["Firing enabled"])
        #print (summary)
            
        return situation
    
    def fire_transition (self,Transition):
        
        current_state=np.array(self.state)  
        
        #print(current_state)
        firing_array =pd.DataFrame(np.zeros(self.NTRANSITIONS),index=self.Transition_names).T
        firing_array[Transition]=1
        
        Next_state=(firing_array.values.dot(self.Combined_incidence.T.values)+ current_state)[0]
            
        if all([Next_state[i] >= 0 for i in range(self.NPLACES)]):  
            
            for i in range(self.NPLACES) :       
                self.Places_obj[i].token=Next_state[i] 
               
            #print ("firing successful ")  
            print(f" firing successful! Current State {current_state} , Action {Transition} , Next State {Next_state}")
            return (Next_state,True)    
        else : 
            #print ("firing Halted ")
            print(f"firing Halted! Current State {current_state} , Action {Transition} , Next State {Next_state}")
            return (current_state,False)
        
        

    def Reward(self,fire_transition):
        
        reward=0
        
        Next_state,delivery= fire_transition
   
        if self.state[-1]>=self.goal:  # Goal achieved  
        
            reward=+100
            print("Goal achieved !! ")
        
            self.terminal=True
            

        elif delivery == False :   # firing halted
            reward=-10
            
        elif delivery==True: # firing successful 
                  
            reward=1
            
        else : reward=-1
               
        return reward
    


petri=Petri_env(path)
petri.load_model()


#%% Dyna Q

class DynaQ :
    
    def __init__(self,Petri_env,exploration=0.5, lr=0.3, n_steps=1, episodes=1):
        
        self.env=Petri_env 
        self.n_steps=n_steps
        self.learning_rate=lr
        self.episodes=episodes
        self.exploration=exploration
        
        
        self.Qvalues={}
        self.state_space ={}
        self.state_actions=[]
        self.states_history=[]
        self.state =self.env.state
        self.action_space=self.env.Transition_names
      
              
    def chooseAction(self):
        # epsilon-greedy
        action = ""
        
        if np.random.uniform(0, 1) <= self.exploration:
            action = np.random.choice(self.action_space)
            print (f"Action choosed with epsilon Greedy {action}")
            
        else:
            
            try : 
           
                # greedy action
                current_position = tuple(self.state)
                # if all actions have same value, then select randomly
                if len(set(self.Qvalues[current_position].values())) == 1:
                    action = np.random.choice(self.action_space)
                    print (f" Action choosed for same Q values {action}")
                
            
                else:
                    for a in self.action_space:
                    
                        Qvalues=list(self.Qvalues[current_position].values())
                        action=self.action_space[Qvalues.index(max(Qvalues))]
                    print (f" Action choosed for Optimal policy {action}")
                        
            except KeyError :
                  
                  print("new_state")
                  
                  self.Qvalues[current_position]= {}                      
                  for a in self.action_space:
                      self.Qvalues[current_position][a] = 0   
                  action = np.random.choice(self.action_space)
        
        return(action)
    
    
    def reset(self):
        
        self.env.delivered=0
        self.env.state=self.state = self.env.Marking.loc["Current"]
        self.env.terminal=False  
        self.Qvalues={}
        self.state_space ={}
        self.state_actions=[]
        
    
    
    def Train(self):       
        self.steps_per_episode = [] 

        for ep in range(self.episodes):  
            
            while not self.env.terminal:
                
                action = self.chooseAction()
                self.state_actions.append((self.env.state, action))
                
                
                Next_state,delivery =self.env.fire_transition(action)
                reward=self.env.Reward(self.env.fire_transition(action)) 
                
                try : 
                    
                    print (Next_state,self.state)
                    self.Qvalues[tuple(self.state)][action] += self.learning_rate*(reward + np.max(list(self.Qvalues[tuple (Next_state)].values()))-self.Qvalues[tuple(self.state)][action])
                
                except KeyError :
                    
                    self.Qvalues[tuple(self.state)]= {}                      
                    for a in self.action_space:
                        self.Qvalues[tuple(self.state)][a] = 0
                        
   
                self.env.state =self.state =Next_state              
                self.states_history.append(self.state.tolist())
                
              
                
                if all([i==False for i in self.env.possible_firing()]) :
                    
                    print("Terminal or dead lock")
                    self.env.terminal=True
     
        
    
              

agent=DynaQ(petri)

#%% Test 


agent.reset()
agent.Train()

print (agent.env.summary)

#%%

print (agent.Qvalues)
#print(agent.states_history)





