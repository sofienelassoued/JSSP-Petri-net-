# -*- coding: utf-8 -*-

import pandas as pd


path = "D:\Sciebo\Semester 4 (Project Thesis)\Git Projects\My petrinet simulator/petri0.html"


class Petri_env:
    
    def __init__(self,path):
    

        self.Forwards_incidence = pd.read_html(path,header=0,index_col=0)[1]
        self.Backwards_incidence= pd.read_html(path,header=0,index_col=0)[3]
        self.Combined_incidence = pd.read_html(path,header=0,index_col=0)[5]
        self.Inhibition_matrix = pd.read_html(path,header=0,index_col=0)[7]
        self.Initial_Marking =pd.read_html(path,header=0,index_col=0)[9]
            
        self.Places_names= self.Forwards_incidence.index.tolist()
        self.NPLACES=len( self.Places_names)
        self.Places_obj=[]
        
        self.Transition_names= self.Forwards_incidence.columns.tolist()   
        self.NTRANSITIONS=len( self.Transition_names)
        self.Transition_obj=[]
        
           
        
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
            token=self.Initial_Marking.loc["Current"][i]
                
                
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
         
           

petri=Petri_env(path)
petri.load_model()

#%%
print(petri.Combined_incidence)
print(petri.Initial_Marking)

for i in range (petri.NPLACES) :
    print(petri.Places_obj[i])
       
for i in range (petri.NTRANSITIONS) :
    print(petri.Transition_obj[i])



