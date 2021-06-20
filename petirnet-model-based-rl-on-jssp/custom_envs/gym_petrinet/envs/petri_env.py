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

from random import randrange

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
      self.max_steps=500 # maximum steps in episode before terminating the eipsode
      self.grafic_container=[]
      self.saved_render=[]
      
      self.path = os.getcwd()+"\modelisation/2.html"
      self.Forwards_incidence = pd.read_html(self.path,header=0,index_col=0)[1]
      self.Backwards_incidence= pd.read_html(self.path,header=0,index_col=0)[3]
      self.Combined_incidence = pd.read_html(self.path,header=0,index_col=0)[5]
      self.Inhibition_matrix = pd.read_html(self.path,header=0,index_col=0)[7]
      self.initial_marking =pd.read_html(self.path,header=0,index_col=0)[9].loc["Current"]
      
      self.process_timing = {"S11":1,"S21":2,"S31":3,"S12":3,"S32":2,"S22":1}
      #self.process_timing = {"S11":0,"S21":0,"S31":0,"S12":0,"S32":0,"S22":0}
             
      self.Places_names= self.Forwards_incidence.index.tolist()
      self.NPLACES=len( self.Places_names)
      self.Places_obj=[]
      self.Places_dict={}
        
      self.Transition_names= self.Forwards_incidence.columns.tolist()   
      self.NTRANSITIONS=len( self.Transition_names)
      self.Transition_obj=[]
      self.Transition_dict={}
        
      self.goal=19
      self.delivered=0
      self.marking =self.initial_marking
 
      self.action_space = spaces.Discrete(self.NTRANSITIONS)   
      
      self.observation_space = spaces.Box(low=-100, high=100, shape=(1,self.NTRANSITIONS),dtype=np.int32)
      #self.observation_space = spaces.Box(low=-100, high=100, shape=(6,6),dtype=np.int32)
      
      

   
      #------------------Load and reconstruct the Petrinet from HTML file----------------# 
 
      class Place:          
          def __init__(self,name,token,In_arcs,Out_arcs,time,features,waiting_time):
            
              self.pname =name
              self.token=token  
              self.In_arcs=In_arcs
              self.Out_arcs=Out_arcs 
               
              self.waiting_time=waiting_time
              self.process_time=0
              self.features=features
              self.features[0]=token 
 
                  

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
          feature=[-1]*len(self.marking)
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
            
      print ("Model Loaded from {}".format(self.path))
      
      
     # for i in (self.Places_obj):
     #     i.token=randrange(10)
    #  self.Places_obj[-4].token=10


        
  def Create_Snapshot(self,action,fired,inprocess,reward,firing,episode=0,):

      def graph_generater(action,fired,inprocess):
                 
           g = Digraph('output', format='jpg' ) 
           
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
     
      white= (255, 255, 255)
      black = (0, 0, 0)
      blue = (0, 0, 255)
      
      pygame.font.init()
      font = pygame.font.Font('freesansbold.ttf', 12)
      font2 = pygame.font.SysFont('arial', 11)
      
      petri=graph_generater(action,fired,inprocess)  
      petri.render(str(self.simulation_clock),cleanup=True)
      
      image=pygame.image.load(str(self.simulation_clock)+".jpg") 
      Episode=font.render(str("Episode : "+str (episode)), True, black)   
      Step=font.render(str("Step : "+str (self.simulation_clock)), True, black)        
      step_Reward=font.render(str("Step Reward : "+str (reward)), True, blue)
      ep_Reward=font.render(str("Episode Reward : "+str (self.episode_reward)), True, blue)
      firing=font2.render(str(firing), True, blue)
      
      display_width = image.get_width()
      display_height =image.get_height()
      
      if  display_width >700 or display_height :
          display_width=600
          display_height=500
          image = pygame.transform.scale(image, (display_width, display_height))
          
      screen_shot=pygame.Surface((display_width,display_height+100)) 
      screen_shot.fill(white)
      screen_shot.blits(blit_sequence=((Episode,(5,0)),(Step,(5,20)),(firing,(5,40)),(step_Reward,(5,60)),(ep_Reward,(5,80)),(image,(0,100))))
  
      self.grafic_container.append (screen_shot)  
      self.saved_render.append (screen_shot)
      os.remove(str(self.simulation_clock)+".jpg")
 
                    
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
      in_process_Places=[]
      
    
        
      Transition=self.Transition_names[action]
      current_marking=np.array(tuple(self.marking)).astype(np.int64)
  
        
      firing_array =pd.DataFrame(np.zeros(self.NTRANSITIONS),index=self.Transition_names)
      firing_array.iloc[int (action)]=1
   
      Next_marking_values=(firing_array.T.values.dot(self.Combined_incidence.T.values)+ current_marking)[0].astype(np.int64)
      Next_marking=pd.DataFrame(Next_marking_values,index=self.Places_names,columns=["Current"],dtype="int64")
       
      possible = all([Next_marking.loc[i].values >= 0 for i in Next_marking.index])  #test if firing is possible 
      
      for i in (self.Transition_dict[Transition][1]):  # test is an Upstream place still in process
          for p in self.Places_obj:      
              if p.pname==i and p.waiting_time>0:  
                 in_process=True
                 in_process_Places.append(p.pname)
                                        
      #generate the feature matrix Stacked 
      #FM=pd.concat([self.marking,self.Combined_incidence], axis=1).to_numpy()
      
      #generate the feature matrix multiplied   
      FM=self.Combined_incidence.T.values.dot(current_marking).reshape(1,self.NTRANSITIONS)
      

      if  not possible  :

          #print("firing Halted! ")
          return (self.marking,FM,False,in_process_Places) 
                  
      elif in_process:  
  
         # print("Upstream  Still in process , firing halted ")
          return (self.marking,FM,False,in_process_Places)   
        
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
         return (Next_marking["Current"],FM,True,in_process_Places)
     
    
  def Reward(self,Next_state,delivery,): 
      
      reward=0   
      if  int(self.marking["OB"])>self.goal:
          
          # Goal achieved  
          reward=+10
          firing_info="Goal achieved !! "  
          self.Terminal=True
          print("Goal achieved !! " )
          
      elif self.delivered<int(self.marking["OB"]):
          # a piece is delivered       
          reward=+7 
          firing_info="a piece is delivered  "
          
      elif self.Terminal==True :
          # dead lock
          reward=-100
          firing_info="Dead lock"

      elif delivery == False :
          # firing halted
          reward=-5
          firing_info="in process firing halted" 
            
      else :
          # firing sccessful  
          #reward=-self.simulation_clock                 
          reward=5
          firing_info="in process firing successful"
    
      self.delivered=int(self.marking["OB"])   
     # print(int(self.marking["OB"]))

      
      return reward ,firing_info
 

  def step(self, action,testing=False,episode=0):
      
      reward=0
      done=False
      info = {}
      observation=[]
      self.simulation_clock+=1
      
      #print (f"****** Simulation Clock {self.simulation_clock}  ****** ")

      for p in self.Places_obj: 
          
          #Synchronising dic and Obj Places and marking
          p.token=self.marking[p.pname]
          p.features[0]=p.token
          self.Places_dict[p.pname]= [p.token,p.In_arcs, p.Out_arcs,p.process_time,p.features]


          if p.process_time>0:     
              p.waiting_time-=1 #update internal clock                          
 
      for t in self.Transition_obj: #Synchronising dic and Obj Transition      
          self.Transition_dict[t.tname]= [t.time,t.In_arcs,t.Out_arcs]         
   
      #test termination               
      transition_summary=self.possible_firing()["Firing enabled"] 
      if all([transition_summary[i]==False for i in transition_summary.index]) : 
         # print("no fireable transition available episode Terminated ")
          self.Terminal=True  
          
      elif self.simulation_clock> self.max_steps:
         # print("No response episode Terminated")
          self.Terminal=True 
          
             
      Nxmarking,features,fired,inprocess=self.fire_transition (action)
      
      
      
      observation=features
      reward,firing_info=self.Reward(Nxmarking,fired)
      info.update({"Action": self.Transition_names[action]})
      done=self.Terminal

      self.marking=Nxmarking  
      self.episode_reward+=reward
      
      
      if testing==True: # take a snapshot in testing fase 
          self.Create_Snapshot(self.Transition_names[action],fired,inprocess,reward,firing_info,episode)
          print ("Screenshot for step {} created ".format(self.simulation_clock))
          
      return observation, reward, done, info
        
      
  def reset(self):


      self.Terminal=False  
      self.episode_timing=0
      self.episode_reward=0
      self.simulation_clock=0
      self.episode_actions_history=[]
      self.marking=self.initial_marking
      
      array=np.array(np.zeros((1,self.NTRANSITIONS)),dtype=np.int32)  
      matrix=np.array(np.zeros((6,6)),dtype=np.int32) 
          
      return    array 

   
        
  def render(self,replay=False,continues=True):  
      
      speed =100
      position=(0,0)
      white= (255, 255, 255)
      clock = pygame.time.Clock() 
      
      try:
       display_width = (self.grafic_container[0].get_width())
       display_height =(self.grafic_container[0].get_height())+100
          
      except:
          display_width=300
          display_height=500
          
      pygame.init()
      pygame.display.init()   
      pygame.display.set_caption('Petrinet')
      Display = pygame.display.set_mode((display_width,display_height))
      Display.fill(white)
       
         
      frame =0  
      clock.tick(1)
      restart=True
      paused=False
           
      if continues==True:
          while True :          
              if restart==True:
                  
                  for i in range (len(self.grafic_container)):
              
                      pygame.time.wait(speed)               
                      if replay==False:Display.blit(self.grafic_container[i],position)
                      else:Display.blit(self.saved_render[i],position)  
                      
                      pygame.display.update()
                      for event in pygame.event.get() :
                          if event.type == pygame.QUIT :  
                              pygame.quit()
                              break
                          #Pause the animation  use "p" to pause and "c" to continue 
                          elif event.type == pygame.KEYDOWN:
                              if event.key == pygame.K_p:
                                  paused=True
                                  while paused==True :
                                      Display.blit(self.saved_render[i],position)
                                      for event in pygame.event.get() :
                                          if event.type == pygame.KEYDOWN:
                                              if event.key == pygame.K_c:
                                                  paused=False
                                                  break
                  restart=False
              #replay the animation     
              for event in pygame.event.get() :
                   if event.type == pygame.QUIT :
                              pygame.quit()
                              break
                   elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        restart=True
                        
      else : #play the animation frame by frame use Left and right keys:
      
          while True: 
            
              if replay==False:Display.blit(self.grafic_container[frame],position)
              else:Display.blit(self.saved_render[frame],position) 
              pygame.display.update()
              
              for event in pygame.event.get() :     
                  if event.type == pygame.QUIT :
                      pygame.quit()
                      break
                 
                  elif event.type == pygame.KEYDOWN:                   
                      if event.key == pygame.K_RIGHT: 
                         if frame <len(self.grafic_container)-1:frame+=1           
                      elif event.key ==pygame.K_LEFT: 
                          if frame >0 :frame-=1
                         
                                  
  
  def close(self):
      pygame.display.quit()
         
