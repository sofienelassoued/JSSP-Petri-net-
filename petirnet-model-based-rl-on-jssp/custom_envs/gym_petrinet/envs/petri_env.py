#%% Libraries importation
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from graphviz import Digraph
from PIL import Image

#%% Main environement 
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

      self.viewer = None
      self.Terminal=False
      self.simulation_clock=0
      
      self.path = "D:\Sciebo\Semester 4 (Project Thesis)\Programming\Petrinet modelisation/petri1.html"
      self.Forwards_incidence = pd.read_html(self.path,header=0,index_col=0)[1]
      self.Backwards_incidence= pd.read_html(self.path,header=0,index_col=0)[3]
      self.Combined_incidence = pd.read_html(self.path,header=0,index_col=0)[5]
      self.Inhibition_matrix = pd.read_html(self.path,header=0,index_col=0)[7]
      self.initial_marking =pd.read_html(self.path,header=0,index_col=0)[9].loc["Current"]
      
      self.process_time = {"P16":3,"P12":1,"P14":2,"P6":3,"P2":1,"P4":5}
      #elf.process_time = {"P16":0,"P12":0,"P14":0,"P6":0,"P2":0,"P4":0}
             
      self.Places_names= self.Forwards_incidence.index.tolist()
      self.NPLACES=len( self.Places_names)
      self.Places_obj=[]
      self.Places_dict={}
        
      self.Transition_names= self.Forwards_incidence.columns.tolist()   
      self.NTRANSITIONS=len( self.Transition_names)
      self.Transition_obj=[]
      self.Transition_dict={}
        
      self.goal=5
      self.marking =self.initial_marking
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
              return(f"Place name {self.pname}  Tokens: {self.token} Process Time: {self.process_time} Input:{self.In_arcs}  Output{self.Out_arcs } , currently enabled:{self.enabled}, enabled time :{self.token_enabled_time}" )
                   
      class Transition:
          def __init__(self,name,time,In_arcs,Out_arcs):
        
              self.tname =name
              self.time=0
              self.In_arcs=In_arcs
              self.Out_arcs=Out_arcs                         
       
          def __str__(self):
              return(f"Tansition name {self.tname}  Timer: {self.time}  Input:{self.In_arcs}  Output{self.Out_arcs }" )
          
      for i in self.Places_names: 
          

          # outer loop for every place        
        
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
  

                   
      for i in self.Transition_names:  
          
          # outer loop for every transition        
      
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
            
      #print ("Model Loaded from {}".format(self.path))
                    
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
               
  def fire_transition (self,action):
      
      possible=False
      in_process=False
      in_process_place=""
    
        
      Transition=self.Transition_names[action]
      current_marking=np.array(tuple(self.marking)).astype(np.int64)
  
        
      firing_array =pd.DataFrame(np.zeros(self.NTRANSITIONS),index=self.Transition_names)
      firing_array.iloc[action]=1
   
      Next_marking_values=(firing_array.T.values.dot(self.Combined_incidence.T.values)+ current_marking)[0].astype(np.int64)
      Next_marking=pd.DataFrame(Next_marking_values,index=self.Places_names,columns=["Current"],dtype="int64")
       
      possible = all([Next_marking.loc[i].values >= 0 for i in Next_marking.index])  #test if firing is possible 
      
      for i in (self.Transition_dict[Transition][1]):  # test is an Upstream place still in process
          for p in self.Places_obj:      
              if p.pname==i and (p.token_enabled_time-p.process_time)<0 :  
                 in_process=True 
                 in_process_place=p.pname
     
 
      if  not possible  :

          print("firing Halted! ")
          return (self.marking,False) 
                  
      elif in_process:  
  
          print(f"Upstream {in_process_place} Still in process , firing halted ")
          return (self.marking,False)   
        
        #-------------if firing successful---------------
         
      else :

         for i in (self.Transition_dict[Transition][2]):   
             #Loop on downstream places
             for k in self.Places_obj:          
                 if k.pname==i:
                     k.token+=1        #Update the token number
                     k.enabled=True    # activate enabled status of place 
            
                       
                        
         for i in (self.Transition_dict[Transition][1]): 
             #Loop on upstream places and reset Clock
             for k in self.Places_obj:
                 if k.pname==i:
                     
                     k.token-=1              #update token number
                     k.enabled=False         #decativate enabled status 
                     k.token_enabled_time=0  #reset plave internal clock
  
         print(" firing successful! ")
         return (Next_marking["Current"],True)
     
         
  def Reward(self,Next_state,delivery): 
      
      reward=0
     
      if  int(self.marking["OB"])>self.goal:
          
          # Goal achieved  
          reward=+100
          print("Goal achieved !! ")  
          self.terminal=True
          
      elif self.terminal==True :
          # dead lock
          reward=-1000
          print ("Dead lock")

      elif delivery == False :
          # firing halted
          reward=-100
          #print("in process firing halted" )
            
      else :# firing sccessful                   
          pass# reward=-time*10p
            #print("in process firing successful" )
      
      return reward  
        
  def graph_generater(self):
      

      self.load_model()
      g = Digraph('output', format='png')
      
      for n in self.Places_obj:
          g.node(str(n.pname), color='black')

      for n in self.Transition_names:
        g.node(str(n),shape="box")

      for i in self.Places_obj:
          for j in i.In_arcs:  
              
              if j=="T1" :
                  g.edge(j,i.pname,color='red') 
              else: 
                  g.edge(j,i.pname,color='black')
              
          for k in i.Out_arcs :    
              g.edge(i.pname,k)
        
      img=g.render() 

      return img

  def step(self, action):
      
  
      reward=0
      done=False
      info = {}
      observation=[]
      Max_steps=1000 # maximum steps in episode before terminating the eipsode
      self.simulation_clock+=1
      
      print (f"*** Simulation Clock {self.simulation_clock}  **** ")

      for p in self.Places_obj: 

          self.Places_dict[p.pname]= [p.token,p.In_arcs, p.Out_arcs,p.process_time,p.features] #Synchronising dic and Obj Places

          if p.enabled==True:     
              p.token_enabled_time+=1 #update internal clock 
                                     
          for j in range (p.token):    # update the feature vector in places objects     
              p.features[j]=self.simulation_clock-p.token_enabled_time
      
          if (p.token_enabled_time-p.process_time) >0:  # not in process# initialise enabled status of place exept in propcess
              p.enabled=False 
          else:p.enabled=True
              
      
      for t in self.Transition_obj: #Synchronising dic and Obj Transition      
          self.Transition_dict[t.tname]= [t.time,t.In_arcs,t.Out_arcs]         
   
                      
      transition_summary=self.possible_firing()["Firing enabled"] 
      if all([transition_summary[i]==False for i in transition_summary.index]) : 
          print("no fireable transition available episode Terminated ")
          self.terminal=True  
          
      elif self.simulation_clock> Max_steps:
          print("No response episode Terminated")
          self.terminal=True 
          
                      
      observation,delivery=self.fire_transition (action)
      reward=self.Reward(observation,delivery)
      info.update({"Action": self.Transition_names[action]})
      done=self.terminal
      
      
      self.marking=observation
      
      return observation, reward, done, info
      
      
      
  def reset(self):

      self.load_model()
      self.terminal=False  
      self.marking=self.initial_marking
      self.episode_actions_history=[]
      self.episode_timing=0
      self.episode_reward=0
      self.simulation_clock=0
      
      return self.initial_marking
      
      
        
  def render(self, mode='human'):
        fname = self.graph_generater()
        im = Image.open(fname)
        im.show() 
      
  def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
      

