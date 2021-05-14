import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


class PetriEnv(gym.Env):

  '''
  
  Description:
      
  Observation:
      
  Actions:
      
      Type: Discrete (Number of available transitions U idle )
      
  Reward:
      
  Starting State:
      
  Episode Termination:
      

  '''
 
  metadata = {'render.modes': ['human']}
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def __init__(self):
      super (PetriEnv, self).__init__()
      
      self.path = "D:\Sciebo\Semester 4 (Project Thesis)\Programming\Petrinet modelisation/petri1.html"
      self.Forwards_incidence = pd.read_html(self.path,header=0,index_col=0)[1]
      self.Backwards_incidence= pd.read_html(self.path,header=0,index_col=0)[3]
      self.Combined_incidence = pd.read_html(self.path,header=0,index_col=0)[5]
      self.Inhibition_matrix = pd.read_html(self.path,header=0,index_col=0)[7]
      self.marking =pd.read_html(self.path,header=0,index_col=0)[9].loc["Current"]
      self.process_time = {"P16":3,"P12":1,"P14":2,"P6":3,"P2":1,"P4":5}
             
      self.Places_names= self.Forwards_incidence.index.tolist()
      self.NPLACES=len( self.Places_names)
      self.Places_obj=[]
      self.Places_dict={}
        
      self.Transition_names= self.Forwards_incidence.columns.tolist()   
      self.NTRANSITIONS=len( self.Transition_names)
      self.Transition_obj=[]
      self.Transition_dict={}
        
      self.goal=5
      self.action_space = spaces.Discrete(self.NTRANSITIONS)
      self.observation_space = spaces.Box(np.array([0]*self.NPLACES), np.array([self.goal]*self.NPLACES),dtype=np.float32)
   

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
            token=self.marking[i]
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
            
      print ("Model Loaded from {}".format(self.path))
               
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
    
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human'):
    ...
  def close(self):
    ...