#!/usr/bin/env python
# coding: utf-8

# In[239]:


import numpy as np
import pandas as pd
import itertools
from itertools import product 
import random


# In[893]:


saved_model =[[],{},[],{}]


# In[1144]:


class Petri_net : 
    
    def __init__(self,saved_model):

        self.places_list=saved_model[0]
        self.places_dic=saved_model[1]
        self.transition_list=saved_model[2]
        self.transition_dict=saved_model[3]
         
    class Place:  
        
        def __init__(self,name,token,In_arcs,Out_arcs):
            
            self.pname =name
            self.pid=int( self.pname[1])-1
            self.token=token
            self.in_tran=In_arcs
            self.out_tran=Out_arcs                         
       
        def __str__(self):
            return(f"The defined place{self.pname} : number of initial tokens: {self.token}             input transitions {self.in_tran}, output transition: {self.out_tran}") 
        
        
    class Transition:
        
        def __init__(self,name,In_arcs,Out_arcs):
            
            self.tname=name
            self.tid=int( self.tname[1])-1
            self.in_places=In_arcs
            self.out_places=Out_arcs
        
        def __str__(self):
            return(f"The defined Transition {self.tname} :  input in_place {self.in_places}, output in_place: {self.out_places}")
                      
        
    def add_place (self):
        
        in_arcs=[]
        out_arcs=[]
        
        name= input(" Place name: ").upper()
        token= int( input(f"Initial number of token in the place {name}: "))    
        for i in range (0,int(input (f"Number of input transition arcs to {name}  : "))):
            in_arcs.append(input(f" input arc number {i+1} ").upper())        
        for i in range (0,int(input (f"Number of output transition arcs to {name}  : "))):
            out_arcs.append(input(f" output arc number {i+1} ").upper())
            
        print(name,token,in_arcs,out_arcs)
            
        self.places_list.append(self.Place(name,token,in_arcs,out_arcs))
        self.places_dic.update({name:self.places_list[-1]})
        
        
    
    def add_transition (self):
        
        in_arcs=[]
        out_arcs=[]
              
        name= input(" Transition name: ").upper()    
        for j in range (0,int(input (f"Number of input places arcs to {name}  : "))):
            in_arcs.append(input(f" input arc number {j+1} ").upper())
        for j in range (0,int(input (f"Number of output places arcs to {name}  : "))):
            out_arcs.append(input(f" output arc number {j+1} ").upper())
        
        self.transition_list.append(self.Transition(name,in_arcs,out_arcs)) 
        self.transition_dict.update({name:self.transition_list[-1]})
            
            
    def Petri_definition(self):
        

        n_places=int(input ("Total Number of Places in the Petri : "))
        for i in range (n_places):
            self.add_place()    
            
        n_transition=int(input ("Total Number of transition in the Petri : "))       
        for i in range (n_transition):
            self.add_transition()
            
    def model_save():
        
       
        saved_model[0]= self.places_list
        saved_model[1]= self.places_dic
        saved_model[2]= self.transition_list
        saved_model[3]= self.transition_dict
         
    def __str__(self):
        
        places=" " 
        transition=" " 
                  
        for a in range (len(self.places_list)) :
            places+=str(self.places_list[a])

        for b in range (len (self.transition_list)) :
            transition+=str(self.transition_list[b]) 

            return(f"The Petrinet summary \n  {places} \n {transition} ")
                
    def possible_firing(self) :
        
        situation=[True]*len(self.transition_list)   
        
        for i in range (len(self.transition_list)):
            
                for j in range (len (self.transition_list[i].in_places)):
                    
                            
                    if (self.places_dic [(self.transition_list[i].in_places[j])].token)-1<0:                        
                        situation[i]=False
        return situation
    
    def firing (self,ref):
        
        for i in range (len(self.transition_dict[ref].in_places)):                 
            if self.places_dic[self.transition_dict[ref].in_places[i]].token-1<0:
          
                print("firing Halted")
            
            
            
                break
                
            else:
                
                for i in range (len(self.transition_dict[ref].in_places)):                   
                    self.places_dic[self.transition_dict[ref].in_places[i]].token-=1
                
                for i in range (len(self.transition_dict[ref].out_places)):                   
                    self.places_dic[self.transition_dict[ref].out_places[i]].token+=1                
                print ("firing successful ")   
                break
    
    

class Q_learning(Petri_net) :   
      
    def __init__(self):
        super().__init__(saved_model) 
       
    

        self.delivered=0
        
        self.state=[0]*(len(self.places_list))
        
        for p in range (0,len(self.places_list)): 
            self.state[p]=self.places_list[p].token

        self.state_space=  list(itertools.product(range(0,(5)), repeat = len(self.places_list)))
        self.action_space= self.transition_list
        self.state_action_space=[0]*(len(self.state_space)*len(self.action_space))
        
        self.Q_tabel=pd.DataFrame(np.zeros((len(self.state_space), len(self.action_space))))
        #self.Q_tabel.insert(loc=0, column="State", value=self.state_space, allow_duplicates=False)
        self.TQ_tabel=self.Q_tabel.T
        
        self.states_coder={} 
        self.states_decoder={} 
        for i in range(len(self.state_space)):
            self.states_coder.update({i:self.state_space[i]})           
        for i in range(len(self.state_space)):
            self.states_decoder.update({self.state_space[i]:i})
 
    
    def reset(self):
        
        self.Q_tabel=pd.DataFrame(np.zeros((len(self.state_space), len(self.action_space))))
        #self.Q_tabel.insert(loc=0, column="State", value=self.state_space, allow_duplicates=False)    
        return(self.Q_tabel)
    

    
    def state_update(self):
        
        for p in range (0,len(self.places_list)): 
            self.state[p]=self.places_list[p].token 

        return (self.state)
        
    def Q_update(self):
 
        Updated_Qtable=self.Q_tabel.copy()     
        for i in range (len(self.state_space)):
            for j in range (0,len(self.action_space)):
                Updated_Qtable.iloc[i,j]+= -1 
                
        Updated_Qtable.iloc[0,1]= 10
        Updated_Qtable.iloc[41,1]= 30
        Updated_Qtable.iloc[41,0]= 20
        
        self.Q_tabel=Updated_Qtable
        self.TQ_tabel=self.Q_tabel.T
        return (Updated_Qtable)
    
        
    def step(self,action,):
        
        delivered=0
        
        next_state=[0]*len(self.places_list)
        
        self.firing(action.tname)
     
        for p in range (0,len(self.places_list)): 
            next_state[p]=self.places_list[p].token
        print(next_state)
        
        
        state_id=self.states_decoder[tuple (self.state)]
        next_state_id=self.states_decoder[tuple (next_state)]
        
        reward=self.TQ_tabel[state_id][action.tid]
        
        if self.places_list[3].token>=2:
                self.places_list[3].token-=1
                self.delivered+=1
  
        print(f" old state {state_id}:{self.state} the new State {next_state_id}:{next_state}   collected reward :{reward}, ")
        print(f" total delivery :{self.delivered} ")
    
        return (state_id,next_state_id,reward)
        
    
        
    
    
    def training (self):

        all_epochs = []
        all_penalties = []
        alpha = 0.1
        gamma = 0.6
        epsilon = 0.1
        epochs=10
            

        for i in range(1, epochs):

            current_state = self.states_decoder[tuple(self.state)]
            epochs,  reward, = 0, 0
            done = False
            self.delivered=0

            while  done==False:
                
                if random.uniform(0, 1) < epsilon:
                    action = random.sample(self.action_space, 1)[0] # Explore action space
                    print(f" random action for the state {self.states_coder[current_state]} is {action.tname}")
                else:
                    action = self.transition_list[np.argmax(self.TQ_tabel[current_state])] # Exploit learned values
                    print(f" best action   for the state {self.states_coder[current_state]} is {action.tname}")
                    
                    
                current_state_id,next_state_id,reward= self.step(action) 
                
                old_value = self.TQ_tabel[current_state_id][(action.tid)]
                next_max = np.max(self.TQ_tabel[next_state_id])
                
                new_value = (1 - alpha) * old_value + alpha * (reward -1 + gamma * next_max )
                
                self.TQ_tabel[current_state_id][(action.tid)] = new_value
                
                
                current_state =next_state_id 
                
                if self.delivered>=100  :   
                    done = True
                    
                
            print (f"Delivery goal Reached end episode  {i}")
                
   


# In[1145]:


petri_test=Petri_net(saved_model)
rl=Q_learning()


# In[993]:


#petri_test.Petri_definition()


# In[1143]:


petri_test.places_list[3].token=1
petri_test.places_list[2].token=2
petri_test.places_list[1].token=0
petri_test.places_list[0].token=1


# In[1140]:


print (len (petri_test.places_list),len (petri_test.transition_list))


# In[1141]:


print (petri_test.possible_firing())
#petri_test.firing("T2")


# In[1147]:


rl.training()


# In[1146]:


print (rl.state)
print( len(rl.state_space))
print (len( rl.state_action_space))


# In[1133]:


rl.reset()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




