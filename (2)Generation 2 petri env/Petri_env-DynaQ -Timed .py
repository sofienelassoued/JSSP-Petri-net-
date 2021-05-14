# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        self.Places_dict={}
        
        self.Transition_names= self.Forwards_incidence.columns.tolist()   
        self.NTRANSITIONS=len( self.Transition_names)
        self.Transition_obj=[]
        self.Transition_dict={}
        
        
        self.state=self.Marking.loc["Current"]
        self.acion_space=self.Transition_obj
        self.summary=[]
        
        
        self.goal=10
        self.terminal=False
        self.process_time = {"P16":3,"P12":1,"P14":2,"P6":3,"P2":1,"P4":2}
         
        
        
    def load_model(self):
        
        class Place:
            def __init__(self,name,token,time,In_arcs,Out_arcs):
            
               self.pname =name
               self.token=token
               self.time=time
               self.in_tran=In_arcs
               self.out_tran=Out_arcs                         
       
            def __str__(self):
                return(f"Place name {self.pname}  Tokens: {self.token} Process Time: {self.time} Input:{self.in_tran}  Output{self.out_tran}" )
                   
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
            time=0
            token=self.Marking.loc["Current"][i]
            

                
            for j in self.Forwards_incidence.columns.tolist() :      
                if self.Forwards_incidence.loc[i,j]==1:
                    In_arcs.append(j)
            
            for k in self.Backwards_incidence.columns.tolist() :      
                if self.Backwards_incidence.loc[i,k]==1:
                    Out_arcs.append(k)   
                    
            if i in list(self.process_time.keys()):
                time=self.process_time[i]
             
            self.Places_obj.append(Place(name,token,time,In_arcs, Out_arcs))
            self.Places_dict.update({name: [token,time,In_arcs, Out_arcs]})
            

            
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
            self.Transition_dict.update({name: [time,In_arcs, Out_arcs]})
               
  
    
    def fire_transition (self,Transition):
        
        ptime=0
        current_state=np.array(self.state)  
        firing_array =pd.DataFrame(np.zeros(self.NTRANSITIONS),index=self.Transition_names).T
        firing_array[Transition]=1
        
        Next_state_values=(firing_array.values.dot(self.Combined_incidence.T.values)+ current_state)[0]
        Next_state=pd.DataFrame(Next_state_values,index=self.Places_names,columns=["Current"],dtype="int64")
            
        if all([Next_state.loc[i].values >= 0 for i in Next_state.index]):           
        
            for i in (self.Transition_dict[Transition][2]):        
                if ptime < self.Places_dict[i][1]:
                    ptime = self.Places_dict[i][1]
                    
            #print(f" firing successful! Current State {current_state} , Action {Transition} , Next State {Next_state.values}, Max_Process_timing ={ptime}")
   
            return (Next_state["Current"],True,ptime) 
        
        else : 
           #print(f"firing Halted! Current State {current_state} , Action {Transition} , Next State {Next_state.values} ,Max_Process_timing ={ptime}")
            return (self.state,False,ptime)   
        
        
    def possible_firing(self) :
        
        situation=[]         
        for i in self.Transition_names:     
            situation.append(self.fire_transition(i)[1])  
            
        summary = pd.DataFrame(situation,index=self.Transition_names,columns=["Firing enabled"])    
        return summary


    def Reward(self,fire_transition):   
        reward=0
        Next_state,delivery,time= fire_transition
        
        if  int(self.state["OB"])>self.goal:  # Goal achieved  
            reward=+100
            print("Goal achieved !! ")  
            self.terminal=True
        
      
        elif self.terminal==True : # dead lock
            reward=-1000
            print ("Dead lock")
            

        elif delivery == False :   # firing halted
            reward=-100
            #print("in process firing halted" )
            
        else :# firing sccessful                   
            reward=-time*10
            #print("in process firing successful" )
      
        return reward
    


petri=Petri_env(path)
petri.load_model()


#%% Dyna Q

class DynaQ :
    
    def __init__(self,Petri_env,exploration=0.1,Gamma=0.9, lr=0.3, episodes=100):
        
  
        self.Gamma=Gamma
        self.env=Petri_env 
        self.learning_rate=lr
        self.episodes=episodes
        self.exploration=exploration
        
        self.state_space ={}
        self.Qvalues={}        
        self.state_actions=[]
        self.state =self.env.state
        self.action_space=self.env.Transition_names
        
        self.states_history=[]
        self.state_actions_history=[]
        self.episode_actions_history=[]
        
        self.episode_clock=0
        self.episode_reward=0
        self.Nsteps=[]     
        self.episodes_timing = []
        self.episodes_Reward = []
        
         
    def chooseAction(self,Forced=False):
        # epsilon-greedy
        action = ""
        enabled_actions=[]

        
        transition_summary=self.env.possible_firing()["Firing enabled"]        
        for i in  transition_summary.index:
            if transition_summary.loc[i]==True : 
                enabled_actions.append(i)
                
        if Forced ==False :  
            enabled_actions=self.action_space # force transition 
            
                
        if len (enabled_actions)==0:
            #print("No action possible")
            self.env.terminal=True  
            print("No action possible")
            return(self.env.terminal)
            
        
        elif np.random.uniform(0, 1) <= self.exploration:
            action = np.random.choice(enabled_actions)
            #print (f"Action choosed with epsilon Greedy {action}")
          
        else:
            
            try : 
           
                # greedy action
                current_position = tuple(self.state.values)
                # if all actions have same value, then select randomly
                if len(set(self.Qvalues[current_position].values())) == 1:
                    action = np.random.choice(enabled_actions)
                    #print (f" Action choosed for same Q values {action}")            
            
                else:
                    for a in self.action_space:
                    
                        Qvalues=list(self.Qvalues[current_position].values())
                        action=self.action_space[Qvalues.index(max(Qvalues))]
                    #print (f" Action choosed for Optimal policy {action}")
                        
            except KeyError :
                  
                  print("new_state")
                  
                  self.Qvalues[current_position]= {}                      
                  for a in self.action_space:
                      self.Qvalues[current_position][a] = 0   
                  action = np.random.choice(enabled_actions)                
                  
        self.episode_actions_history.append(action)
        
        return(action)
    
    
    def reset(self):
        
        self.env.terminal=False  
        self.env.state=self.state = self.env.Marking.loc["Current"]
        self.episode_actions_history=[]
        self.episode_clock=0
        self.episode_reward=0
        
        
    def Train(self):   
        
        for ep in range(self.episodes):       
            print(f" **----------Episode number:{ep} exploration:{self.exploration}------------**")
                        
            self.exploration+= 0.8/self.episodes     
            self.Nsteps.append(len(self.episode_actions_history))
            self.episodes_timing.append(self.episode_clock)
            self.episodes_Reward.append(self.episode_reward)
            self.reset()
                 
            while not self.env.terminal:
                
                
                transition_summary=self.env.possible_firing()["Firing enabled"] 
                if all([transition_summary[i]==False for i in transition_summary.index]) :
                    self.env.terminal=True                
                
                action = self.chooseAction(False) 
                self.state_actions_history.append((tuple(self.env.state.values), action))           
                if action ==True: 
                    break         
                                 
                            
                Next_state,delivery,time =self.env.fire_transition(action)
                reward=self.env.Reward(self.env.fire_transition(action))
                      
                current_position=tuple (self.state.values)
                next_position =tuple(Next_state.values)
                self.episode_clock+=time
                self.episode_reward+=reward
                
                try :                     
                    self.Qvalues[current_position][action] += self.learning_rate*(reward + self.Gamma*(np.max(list(self.Qvalues[next_position].values())))-self.Qvalues[current_position][action])
                   
                
                except KeyError :  
                    
                    self.Qvalues[current_position]= {}                      
                    for a in self.action_space:
                        self.Qvalues[current_position][a] = 0   
                        

                self.env.state =self.state =Next_state      
                self.states_history.append(self.state.values[0])
                

        self.Nsteps.pop(0)
        self.episodes_timing.pop(0)
        self.episodes_Reward.pop(0)
        print(f"Number of steps : {self.Nsteps}")
        print(f"Timing : {self.episodes_timing}")
        print(f"Reward :{self.episodes_Reward}")
        
        plt.ylim(0, 900)
        plt.figure(figsize=[10, 6])
        plt.plot(range(self.episodes-1), self.Nsteps, label="Number of Steps")
        plt.plot(range(self.episodes-1), self.episodes_timing, label="timing")

agent=DynaQ(petri)

#%%Display Model 

#print (agent.env.Forwards_incidence)
#print (agent.env.Combined_incidence)
#print (agent.env.Marking)
#for i in agent.env.Places_obj:  
    #print(i)

#for i in agent.env.Transition_obj:  
   # print(i)

#%% Test 

agent.Train()


#%%
#print (agent.Qvalues) 
#print ((agent.Qvalues.keys()))  
#print (len(agent.Qvalues.keys()))
#print(agent.states_history)
#print(agent.episode_actions_history)
#print(len(agent.episode_actions_history))
#print(agent.state_actions_history)

#%% 

            
                
                
                
                
                
                
                
                
                
                