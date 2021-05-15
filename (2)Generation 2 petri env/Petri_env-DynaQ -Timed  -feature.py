# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%% Petri env 



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
        
        
        self.marking=self.Marking.loc["Current"]
        self.acion_space=self.Transition_obj
        self.summary=[]
        
        
        self.goal=5
        self.terminal=False
        self.process_time = {"P16":3,"P12":1,"P14":2,"P6":3,"P2":1,"P4":5}
        
        self.simulation_clock=0
         
        
        
    def load_model(self,dimesion=20): # number of channels embeddings
        
        class Place:
            def __init__(self,name,token,In_arcs,Out_arcs,time,features,enabled=False):
            
               self.pname =name
               self.token=token  
               self.In_arcs=In_arcs
               self.Out_arcs=Out_arcs 
               
               self.enabled=enabled
               self.token_enabled_time=0
               self.process_time=time
               self.features=features
                     
       
            def __str__(self):
                return(f"Place name {self.pname}  Tokens: {self.token} Process Time: {self.process_time} Input:{self.In_arcs}  Output{self.Out_arcs } , currently enabled:{self.enabled}" )
                   
        class Transition:
            def __init__(self,name,time,In_arcs,Out_arcs):
            
                self.tname =name
                self.time=0
                self.In_arcs=In_arcs
                self.Out_arcs=Out_arcs                         
       
            def __str__(self):
                return(f"Tansition name {self.tname}  Timer: {self.time}  Input:{self.In_arcs}  Output{self.Out_arcs }" )
          
            
          
        for i in self.Places_names:  # outer loop for every place        
        
            In_arcs=[]
            Out_arcs=[]      
            name=i
            time=0
            token=self.Marking.loc["Current"][i]
            enabled=False
            feature=[-1]*dimesion

            for j in self.Forwards_incidence.columns.tolist() :      
                if self.Forwards_incidence.loc[i,j]==1:
                    In_arcs.append(j)
            
            for k in self.Backwards_incidence.columns.tolist() :      
                if self.Backwards_incidence.loc[i,k]==1:
                    Out_arcs.append(k)   
                    
            if i in list(self.process_time.keys()):
                time=self.process_time[i]
                
            
             
            self.Places_obj.append(Place(name,token,In_arcs, Out_arcs,time,feature,enabled))
            self.Places_dict.update({name: [token,In_arcs, Out_arcs,time,feature,enabled]})
            
            
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
        

        possible=False
        in_process=False
        

        current_marking=np.array(self.marking)  
        firing_array =pd.DataFrame(np.zeros(self.NTRANSITIONS),index=self.Transition_names).T
        firing_array[Transition]=1
        
        Next_marking_values=(firing_array.values.dot(self.Combined_incidence.T.values)+ current_marking)[0]
        Next_marking=pd.DataFrame(Next_marking_values,index=self.Places_names,columns=["Current"],dtype="int64")
        
        possible = all([Next_marking.loc[i].values >= 0 for i in Next_marking.index])
 
        
        for i in (self.Transition_dict[Transition][1]):
            for p in self.Places_obj:
                if p.pname==i and (p.token_enabled_time-p.process_time)<0 :                  
                    in_process=True 
 

        if  not possible  :
           
            print("firing Halted! ")
            return (self.marking,False)
                    
        elif in_process:
            
            print(f"Upstream {i} Still in process , firing halted ")
            return (self.marking,False)
                    
        else :
                  
            for i in (self.Transition_dict[Transition][2]): #Loop on downstream places

                for k in self.Places_obj: # activate enabled status of place
                    if k.pname==i:
                        k.enabled=True
                        
                        
            for i in (self.Transition_dict[Transition][1]): #Loop on upstream places and reset Clock
                 
                for k in self.Places_obj:
                     if k.pname==i:
                         k.token_enabled_time=0
                         k.enabled=False
                
  

            print(" firing successful! ")
   
            return (Next_marking["Current"],True)
        



    def possible_firing(self) :
        
        situation=[]      
        current_marking=np.array(self.marking)  
    
 
        for i in self.Transition_names:        
            possible=False 
            firing_array =pd.DataFrame(np.zeros(self.NTRANSITIONS),index=self.Transition_names).T
            firing_array[i]=1
            Next_marking_values=(firing_array.values.dot(self.Combined_incidence.T.values)+ current_marking)[0]     
            Next_marking=pd.DataFrame(Next_marking_values,index=self.Places_names,columns=["Current"],dtype="int64")
            possible = all([Next_marking.loc[i].values >= 0 for i in Next_marking.index])
            situation.append(possible)  
            
        summary = pd.DataFrame(situation,index=self.Transition_names,columns=["Firing enabled"])    
        return summary


    def Reward(self,Next_marking,delivery):   
        reward=0

        
        if  int(self.marking["OB"])>self.goal:  # Goal achieved  
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
            reward=-self.simulation_clock
            #print("in process firing successful" )
      
        return reward
    


petri=Petri_env(path)
petri.load_model()


#%% Dyna Q

class DynaQ :
    
    def __init__(self,Petri_env,exploration=0.9,Gamma=0.9, lr=0.3, episodes=1000):
        
  
        self.Gamma=Gamma
        self.env=Petri_env 
        self.learning_rate=lr
        self.episodes=episodes
        self.exploration=exploration
        
        self.marking_space ={}
        self.Qvalues={}        
        self.marking_actions=[]
        self.marking =self.env.marking
        self.action_space=self.env.Transition_names
        
        self.markings_history=[]
        self.marking_actions_history=[]
        self.episode_actions_history=[]
        
 
        self.episode_timing=0
        self.episode_reward=0
        self.Nsteps=[]     
        self.All_episodes_timings = []
        self.All_episodes_Rewards= []
        
         
    def chooseAction(self,Forced=False):
        # epsilon-greedy
        action = ""
        enabled_actions=[]

        
        transition_summary=self.env.possible_firing()["Firing enabled"]        
        for i in  transition_summary.index:
            if transition_summary.loc[i]==True : 
                enabled_actions.append(i)
                
        #if Forced ==False :  
            #enabled_actions=self.action_space # force transition 
            
                
        if len (enabled_actions)==0:

            self.env.terminal=True  
            print("No action possible")
            return(self.env.terminal)
            
        
        elif np.random.uniform(0, 1) <= self.exploration:
            action = np.random.choice(enabled_actions)
            print (f"Action choosed with epsilon Greedy {action}")
          
        else:
            
            try : 
           
                # greedy action
                current_position = tuple(self.marking.values)
                # if all actions have same value, then select randomly
                if len(set(self.Qvalues[current_position].values())) == 1:
                    action = np.random.choice(enabled_actions)
                    print (f" Action choosed for same Q values {action}")            
            
                else:
                    for a in self.action_space:
                    
                        Qvalues=list(self.Qvalues[current_position].values())
                        action=self.action_space[Qvalues.index(max(Qvalues))]
                    print (f" Action choosed for Optimal policy {action}")
                        
            except KeyError :
                  
                  print("new_State")
                  
                  self.Qvalues[current_position]= {}                      
                  for a in self.action_space:
                      self.Qvalues[current_position][a] = 0   
                  action = np.random.choice(enabled_actions)                
                  
        self.episode_actions_history.append(action)
        
        return(action)
    
    
    def reset(self):
        
        self.env.terminal=False  
        self.env.marking=self.marking = self.env.Marking.loc["Current"]
        self.episode_actions_history=[]
        self.episode_timing=0
        self.episode_reward=0
        self.env.simulation_clock=0
        
        
    def step(self):
        
        print (f"*** Simulation Clock {self.env.simulation_clock}  **** ")
        self.env.simulation_clock+=1  
        
        if self.env.simulation_clock>1000:
            self.env.terminal=True
            print("unfinishing episode ")
            
        self.env.marking =self.marking #synchronise agent and env marking
        

        for p in self.env.Places_obj:  #update marking in places objects 
           
            p.token=self.marking[p.pname] 
            if p.enabled==True:
                p.token_enabled_time+=1
            
            for j in range (p.token):    # update the feature in places objects     
                p.features[j]=self.env.simulation_clock-p.token_enabled_time

                                        
        for t in self.env.Transition_obj: #Synchronising dic and Obj Transition      
            self.env.Transition_dict[t.tname]= [t.time,t.In_arcs,t.Out_arcs]
                     
        for p in self.env.Places_obj: #Synchronising dic and Obj Places     
            self.env.Places_dict[p.pname]= [p.token,p.In_arcs, p.Out_arcs,p.process_time,p.features]
       
        for k in self.env.Places_obj: # initialise enabled status of place exept in propcess
            
            if (p.token_enabled_time-p.process_time) <=0:  # not in process
                k.enabled=False
                
   
        
        
    def Train(self):   
        
        for ep in range(self.episodes):       
            print(f" **----------Episode number:{ep} exploration:{self.exploration}------------**")
                        
            self.exploration-= 0.7/self.episodes     
            self.Nsteps.append(len(self.episode_actions_history))
            self.All_episodes_timings.append(self.episode_timing)
            self.All_episodes_Rewards.append(self.episode_reward)
            self.reset()
                 
            while not self.env.terminal:
                
                
                transition_summary=self.env.possible_firing()["Firing enabled"] 
                if all([transition_summary[i]==False for i in transition_summary.index]) :
                    self.env.terminal=True                
                
                action = self.chooseAction(False) 
                self.marking_actions_history.append((tuple(self.env.marking.values), action))           
                if action ==True: 
                    break         
                                 
                            
                Next_marking,delivery =self.env.fire_transition(action)
                reward=self.env.Reward(Next_marking,delivery)
    
                self.step() 
                      
                current_position=tuple (self.marking.values)
                next_position =tuple(Next_marking.values)
                self.episode_timing=self.env.simulation_clock
                self.episode_reward+=reward
                
                try :                     
                    self.Qvalues[current_position][action] += self.learning_rate*(reward + self.Gamma*(np.max(list(self.Qvalues[next_position].values())))-self.Qvalues[current_position][action])
                   
                
                except KeyError :  
                    
                    self.Qvalues[current_position]= {}                      
                    for a in self.action_space:
                        self.Qvalues[current_position][a] = 0   
                        

                self.env.marking =self.marking =Next_marking      
                self.markings_history.append(self.marking.values[0])
                #for i in agent.env.Places_obj:print(i.features)
                

        self.Nsteps.pop(0)
        self.All_episodes_timings.pop(0)
        self.All_episodes_Rewards.pop(0)
        print(f"Number of steps : {self.Nsteps}")
        print(f"Timing : {self.All_episodes_timings}")
        print(f"Reward :{self.All_episodes_Rewards}")
        
        plt.ylim(0, 900)
        plt.figure(figsize=[10, 6])
        plt.plot(range(self.episodes-1), self.Nsteps, label="Number of Steps")
        plt.plot(range(self.episodes-1), self.All_episodes_timings, label="timing")

agent=DynaQ(petri)

#%%Display Model 

#print (agent.env.Forwards_incidence)
#print (agent.env.Combined_incidence)
#print (agent.env.marking)
#for i in agent.env.Places_obj:print(i)
#for i in agent.env.Transition_obj:print(i)

#%% Test 

agent.Train()


#%%
#print (agent.Qvalues) 
#print ((agent.Qvalues.keys()))  
print (len(agent.Qvalues.keys()))
#print(agent.markings_history)
#print(agent.episode_actions_history)
#print(len(agent.episode_actions_history))
#print(agent.marking_actions_history)


#%%        
#print(agent.env.marking)