
import glob
import pygame 
import PIL 
import os
from graphviz import Digraph
from graphviz import render
from random import sample


#%%


clock = pygame.time.Clock()

def graph_generater(Places_obj,Transition_names):
      
      g = Digraph('output', format='png')
      
      for n in Places_obj:
          g.node(str(n.pname), color='black')

      for n in Transition_names:
        g.node(str(n),shape="box")

      for i in Places_obj:
          for j in i.In_arcs:   
                  g.edge(j,i.pname,color='black')
              
          for k in i.Out_arcs :    
              g.edge(i.pname,k)

      return g

def create_animated_images (g):
    
    pass
    

    
def load_images():
    
    images_list = []
    #create_animated_images ()


    for filename in glob.glob('D:\Sciebo\Semester 4 (Project Thesis)\Programming\petirnet-model-based-rl-on-jssp/*.png'): #assuming gif
        im=pygame .image.load(filename)
        images_list.append(im)
        #os.remove(filename)
   
    display_height=pygame.Surface.get_height(images_list[0])
    display_width=pygame.Surface.get_width (images_list[0])   
    
    return (display_height,display_width,images_list)


        
display_height,display_width,images_list=load_images() 

black = (0,0,0)
white = (255,255,255)
Display = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Petrinet')


def redraw(i):

    Display.blit(images_list[i], (0,0))
    pygame.display.update()

run = True
'''
while run:
    clock.tick(1)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    for i in range(len(images_list)):
        pygame.time.wait(100)
        redraw(i)
pygame.quit()'''