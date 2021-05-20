
#%% Libraries importation
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from graphviz import Digraph

import pygame 
import os
from graphviz import Digraph
from graphviz import render
from random import sample



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

      self.Terminal=False
      self.simulation_clock=0
      self.grafic_container=[]
      
   
      
      self.path = "D:\Sciebo\Semester 4 (Project Thesis)\Programming\Petrinet modelisation/petri4.html"
      self.Forwards_incidence = pd.read_html(self.path,header=0,index_col=0)[1]
      self.Backwards_incidence= pd.read_html(self.path,header=0,index_col=0)[3]
      self.Combined_incidence = pd.read_html(self.path,header=0,index_col=0)[5]
      self.Inhibition_matrix = pd.read_html(self.path,header=0,index_col=0)[7]
      self.initial_marking =pd.read_html(self.path,header=0,index_col=0)[9].loc["Current"]
      
      self.process_timing = {"P16":3,"P12":1,"P14":2,"P6":3,"P2":1,"P4":5}
      #elf.process_time = {"P16":0,"P12":0,"P14":0,"P6":0,"P2":0,"P4":0}
             
      self.Places_names= self.Forwards_incidence.index.tolist()
      self.NPLACES=len( self.Places_names)
      self.Places_obj=[]
      self.Places_dict={}
        
      self.Transition_names= self.Forwards_incidence.columns.tolist()   
      self.NTRANSITIONS=len( self.Transition_names)
      self.Transition_obj=[]
      self.Transition_dict={}
        
      self.goal=10
      self.feature_dimesion=2*max (self.initial_marking)
      self.marking =self.initial_marking
      self.action_space = spaces.Discrete(self.NTRANSITIONS)
      self.observation_space = spaces.Box(np.array([0]*self.NPLACES), np.array([self.goal]*self.NPLACES),dtype=np.float32)
   

      #------------------Load and reconstruct the Petrinet from HTML file 
 
      class Place:
          
          def __init__(self,name,token,In_arcs,Out_arcs,time,features,waiting_time):
            
              self.pname =name
              self.token=token  
              self.In_arcs=In_arcs
              self.Out_arcs=Out_arcs 
               
              self.waiting_time=waiting_time
              self.process_time=0
              self.features=features
 
                  

          def __str__(self):
              return(f"Place name {self.pname}  Tokens: {self.token} Input:{self.In_arcs}  Output{self.Out_arcs } , process time :{self.process_time} , time until activation {self.waiting_time}" )
                   
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
          feature=[-1]*self.feature_dimesion
          waiting_time=0

          for j in self.Forwards_incidence.columns.tolist() :   
              if self.Forwards_incidence.loc[i,j]==1:
                  In_arcs.append(j)
            
          for k in self.Backwards_incidence.columns.tolist() :   
              if self.Backwards_incidence.loc[i,k]==1:
                  Out_arcs.append(k)   
        
          if i in list(self.process_timing.keys()):
             waiting_time=self.process_timing[i]
        
          self.Places_obj.append(Place(name,token,In_arcs, Out_arcs,time,feature,waiting_time))
          self.Places_dict.update({name: [token,In_arcs, Out_arcs,time,feature,waiting_time]})
              
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
      
      
 
        
  def Create_Snapshot(self,action,fired,inprocess,reward,episode=1):

      def graph_generater(action,fired,inprocess):
                 
           g = Digraph('output', format='png' ) 
           
           for n in self.Places_obj:     
               place=str(str(n.pname)+" ("+str(n.token)+")")
               if n.pname in inprocess:
                   g.node(place, color='blue')
               else: g.node(place, color='black')
                      
           for n in self.Transition_names:    
                      
               if n==action :
                  g.node(str(n),shape="box",color='red')
               else:g.node(str(n),shape="box",color='black')
          
           
           for i in self.Places_obj:           
               place=str(str(i.pname)+" ("+str(i.token)+")")
                    
               for j in i.In_arcs:                 
                   if j==action and fired==True :
                      g.edge(j,place,color='red' )                 
                   else :g.edge(j,place,color='black')
                                      
               for k in i.Out_arcs :    
                   g.edge(place,k)                        
                   
           return g   
     

      black = (0, 0, 0)
      
      pygame.font.init()
      font = pygame.font.Font('freesansbold.ttf', 15)
      
      petri=graph_generater(action,fired,inprocess)  
      petri.render(str(self.simulation_clock),cleanup=True)
      
      image=pygame.image.load(str(self.simulation_clock)+".png") 
      Episode=font.render(str("Episode : "+str (episode)), True, black)   
      Step=font.render(str("Step : "+str (self.simulation_clock)), True, black)        
      Reward=font.render(str("Reward : "+str (reward)), True, black)
      
      display_width = image.get_width()
      display_height =image.get_height()
      screen_shot=pygame.Surface((display_width,display_height))  
      screen_shot.blits(blit_sequence=((image,(0,0)),(Episode,(0,0)),(Step,(0,20)),(Reward,(0,40))))
  
      self.grafic_container.append (screen_shot)
      os.remove(str(self.simulation_clock)+".png")
 
                    
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
      feature_array=[]
      inp_rocess_Places=[]
      
    
        
      Transition=self.Transition_names[action]
      current_marking=np.array(tuple(self.marking)).astype(np.int64)
  
        
      firing_array =pd.DataFrame(np.zeros(self.NTRANSITIONS),index=self.Transition_names)
      firing_array.iloc[action]=1
   
      Next_marking_values=(firing_array.T.values.dot(self.Combined_incidence.T.values)+ current_marking)[0].astype(np.int64)
      Next_marking=pd.DataFrame(Next_marking_values,index=self.Places_names,columns=["Current"],dtype="int64")
       
      possible = all([Next_marking.loc[i].values >= 0 for i in Next_marking.index])  #test if firing is possible 
      
      for i in (self.Transition_dict[Transition][1]):  # test is an Upstream place still in process
          for p in self.Places_obj:      
              if p.pname==i and p.waiting_time>0:  
                 in_process=True
                 inp_rocess_Places.append(p.pname)
                 

      #generate the feature matrix
      for i in self.Places_obj:
          feature_array.append(i.features)       
      feature_matrix =np.matrix(feature_array)
      FM = pd.DataFrame(data=feature_matrix, index=self.Places_names,columns=None)
      
      

      if  not possible  :

          #print("firing Halted! ")
          return (self.marking,FM,False,inp_rocess_Places) 
                  
      elif in_process:  
  
          #print(f"Upstream {in_process_place} Still in process , firing halted ")
          return (self.marking,FM,False,inp_rocess_Places)   
        
        #-------------if firing successful---------------
         
      else :

         for i in (self.Transition_dict[Transition][2]):   
             for k in self.Places_obj:          
                 if k.pname==i:   
                     #Loop on downstream places
                     if k.pname in list(self.process_timing.keys()):
                         k.process_time=self.process_timing[k.pname]
                         

                        
         for i in (self.Transition_dict[Transition][1]): 
             #Loop on upstream places and reset Clock
             for k in self.Places_obj:
                 if k.pname==i:           
                     pass              #change upstream properties
         
         #print(" firing successful! ")
         return (Next_marking["Current"],FM,True,inp_rocess_Places)
     
        
     
         
  def Reward(self,Next_state,delivery): 
      
      reward=0
     
      if  int(self.marking["OB"])>self.goal:
          
          # Goal achieved  
          reward=+100
          #print("Goal achieved !! ")  
          self.Terminal=True
          
      elif self.Terminal==True :
          # dead lock
          reward=-1000
          #print ("Dead lock")

      elif delivery == False :
          # firing halted
          reward=-100
          #print("in process firing halted" )
            
      else :
          # firing sccessful                   
          reward=-self.simulation_clock*10
            #print("in process firing successful" )
      
      return reward  
  
        
  

  def step(self, action,testing=False,episode=0):
      
      reward=0
      done=False
      info = {}
      observation=[]
      Max_steps=500 # maximum steps in episode before terminating the eipsode
      self.simulation_clock+=1
      
      print (f"****** Simulation Clock {self.simulation_clock}  ****** ")

      for p in self.Places_obj: 
          
          #Synchronising dic and Obj Places and marking
          p.token=self.marking[p.pname]
          self.Places_dict[p.pname]= [p.token,p.In_arcs, p.Out_arcs,p.process_time,p.features]


          if p.process_time>0:     
              p.waiting_time-=1 #update internal clock 
                           
          for j in range (p.token):    # update the feature vector in places objects     
              p.features[j]=p.waiting_time
 
      for t in self.Transition_obj: #Synchronising dic and Obj Transition      
          self.Transition_dict[t.tname]= [t.time,t.In_arcs,t.Out_arcs]         
   
      #test termination               
      transition_summary=self.possible_firing()["Firing enabled"] 
      if all([transition_summary[i]==False for i in transition_summary.index]) : 
          #print("no fireable transition available episode Terminated ")
          self.Terminal=True  
          
      elif self.simulation_clock> Max_steps:
          #print("No response episode Terminated")
          self.Terminal=True 
          
             
      Nxmarking,Timefeatures,fired,inprocess=self.fire_transition (action)
      
      observation=np.array(tuple(Nxmarking)).astype(np.int64)
      reward=self.Reward(Nxmarking,fired)
      info.update({"Action": self.Transition_names[action]})
      done=self.Terminal
      
      
      self.marking=Nxmarking  
      
      if testing==True: 
          self.Create_Snapshot(self.Transition_names[action],fired,inprocess,reward,episode)
          
      return observation, reward, done, info
      
      
      
  def reset(self):


      self.Terminal=False  
      self.marking=self.initial_marking
      self.episode_actions_history=[]
      self.grafic_container=[]
      self.episode_timing=0
      self.episode_reward=0
      self.simulation_clock=0
      
      return  np.array(tuple(self.initial_marking)).astype(np.int64)
      
      
        
  def render(self):
           
      
      clock = pygame.time.Clock()   
      try:
       display_width = self.grafic_container[0].get_width()
       display_height =self.grafic_container[0].get_height()
          
      except:
          display_width=300
          display_height=500
          
      pygame.init()
      pygame.display.init()   
      pygame.display.set_caption('Petrinet')
      Display = pygame.display.set_mode((display_width,display_height))
      
      
      clock.tick(1)
  
      for i in range (len(self.grafic_container)):

          pygame.time.wait(500)
          Display.blit(self.grafic_container[i],(0,0))
          pygame.display.update()

          for event in pygame.event.get() :
             if event.type == pygame.QUIT :
                pygame.quit()

      pygame.display.quit()
      
      
  def close(self):
      pygame.display.quit()
         

