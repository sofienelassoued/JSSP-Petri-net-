# -*- coding: utf-8 -*-
"""
Created on Mon Oct  26 02:32:49 2020

@author: Akhil

Edit: 09-12-2020 15:00
@author: Sharafath

"""
import torch
import gym
from gym import spaces
import numpy as np
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Jssp_env(gym.Env):
    
    def __init__(self):
        
        super (Jssp_env, self).__init__()
        
        '''
        The nodes, edges and edge attributes are encoded in the form of one-hot encoding
        The first two elements of the node in each row represent the work piece counts
        present in that particular node and third and fourth elements are encoded to 
        differentiate between a buffer and machine node. The final three elements are 
        encoded to  differentiate each machine and node at any given point.
        
        The nodes are split into four different types input buffers, machines, 
        machine buffers, and output buffer for the sake of convinience. we have 
        workpiece processing times and setup times. the setup times are used only
        whenever there is a change in job. 
        
        In edges the first list contain source nodes and second list contains
        respective target nodes
        
        In edge attributes we have two types of edge attributes job1 and job2, 
        so two different notation
        
        we take job1 as one action and job2 as other use discrete action space 
        and box observation spaces from gym environment. Box space cretaes an 
        array of elements with numbers between higher limit and lower limit 
        and other counters are intialised as zeros and empty lists and the max 
        counter is facilitated to let the agent to explore a maximum number of 
        steps of 600  
        '''
        self.target1 = 10    # wp1 count
        self.target2 = 10    # wp2 count
        
        max_targets = max(self.target1, self.target2)
        
        # self.x to be created for every GNN pass by concatenating 
        # self.input_buffer, self.machine_buffer, self.machines and
        # self.output_buffer.
        
        self.input_buffer = torch.tensor([[self.target1,0,0,1,0,0,1],
                                          [0,self.target2,0,1,0,1,0]], \
                                         dtype = torch.float).to(device)
                               
        self.machine_buffers = torch.tensor([[0,0,0,1,1,1,0],
                                             [0,0,0,1,1,0,1],
                                             [0,0,0,1,0,1,1]],\
                                            dtype = torch.float).to(device)
                                
        self.machines = torch.tensor([[0,0,1,0,1,1,0],
                                      [0,0,1,0,1,0,1],
                                      [0,0,1,0,0,1,1]],\
                                     dtype = torch.float).to(device)
                                     
        self.output_buffer = torch.tensor([0,0,0,1,1,0,0], \
                                     dtype = torch.float).to(device)
            
        self.wp_pt = torch.tensor([[10, 8, 6],
                                    [8, 9, 15]]) # processing time for WP1, WP2 in corresponding machines
        self.wp_st = torch.tensor([[0, 10, 0],
                                   [10, 0, 0]])
        
        self.wp_seq = torch.tensor([[0, 1, 2, 3],
                                   [1, 2, 0, 3]]) # WP1, WP2  processing sequence
        
        self.action_space = spaces.MultiDiscrete((3,2))
        self.observation_space = spaces.Box(np.array([0]*7), np.array([max_targets]*7), \
                                            dtype = np.float32)
        
        self.edgeIndex = torch.tensor([[0,2,5,3,6,4,7,1,3,6,4,7,2,5], #source #edges
                                       [2,5,3,6,4,7,8,3,6,4,7,2,5,8]], #targets
                                      dtype = torch.float).to(device)
        
        
        #self.edgeIndex1 = torch.tensor([[0,2,5,3,6,4,7],
                            #            [2,5,3,6,4,7,8]], dtype = torch.float)
        #self.edgeIndex2 = torch.tensor([[1,3,6,4,7,2,5],
                             #           [2,5,3,6,4,7,8]], dtype = torch.float)
        
        self.edgeAttr = torch.tensor([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],   #job1
                                     [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],\
                                     dtype = torch.float).to(device)  #job2     #edge attributes
        
        self.machine_times = torch.tensor([0.]*3)
        self.current_machine = 0.
        self.ep_rew  = 0.
        self.output_list = []
        self.running_counter_list = []
        self.ep_rew_list = []
        
        self.running_counter = 0
        self.max_counter = max_targets*6*10
        
    def render():
        pass

    def init_buffers(self):
        
        '''
        wp1_first and wp2_first are the machines in which the inputs enter first
        and the wp1 and wp2 count from the input buffers 1 and 2 are directly
        transferd into the machine buffers 1 and 2 respectively. This is how the 
        buffers are initialised
        
        '''
        print('*******Buffers Initialize*******')
        wp1_first = self.wp_seq[0][0]
        wp2_first = self.wp_seq[1][0]
        
        self.machine_buffers[wp1_first][0] = self.input_buffer[0][0]
        self.machine_buffers[wp2_first][1] = self.input_buffer[1][1]


    def terminal(self):
        
        #print(self.running_counter, self.max_counter)
         '''
         This function tells us whether the episode is a success or a fail.
         if all the workpieces reach the output buffer then it is a success
         else it is a fail and also if the running counter value exceeds the 
         max counter then also it is a fail. 
         
         If the process is either success or a fail the workpieces in the output
         buffer are appended  and wp1 and wp2 counts are summed, running counter list
         tracks the number of steps taken to reach the terminal state, and calculates
         the reward for all the steps and episodic rewards 
         '''
         if (self.output_buffer[0] == self.target1 and \
            self.output_buffer[1] == self.target2):
            print('*******Success*******')
            self.output_list.append(sum(self.output_buffer[:2]))
            self.running_counter_list.append(self.running_counter)
            self.reward += 5
            self.ep_rew += self.reward
            self.ep_rew_list.append(self.ep_rew)
            
            return True
     
         elif self.running_counter>self.max_counter:
            print('*******Failed*******')
            self.output_list.append(sum(self.output_buffer[:2]))
            self.running_counter_list.append(self.running_counter)
            self.reward -= 5
            self.ep_rew += self.reward
            self.ep_rew_list.append(self.ep_rew)
            
            return True
        
         else:
            return False
        

    def reset(self):
        
        '''
        This reset helps to get back to initial position of the environment
        
        previous job is the machine which doesnot have a previous job which is m3
        The above mentioned situation is only during initialising the environment 
        later it changes any machine can be empty without any previous job
        
        '''
        self.input_buffer = torch.tensor([[self.target1,0,0,1,0,0,1],
                                          [0,self.target2,0,1,0,1,0]], \
                                         dtype = torch.float).to(device)
                               
        self.machine_buffers = torch.tensor([[0,0,0,1,1,1,0],
                                             [0,0,0,1,1,0,1],
                                             [0,0,0,1,0,1,1]],\
                                            dtype = torch.float).to(device)
                                
        self.machines = torch.tensor([[0,0,1,0,1,1,0],
                                      [0,0,1,0,1,0,1],
                                      [0,0,1,0,0,1,1]],\
                                     dtype = torch.float).to(device)
                                     
        self.output_buffer = torch.tensor([0,0,0,1,1,0,0], \
                                     dtype = torch.float).to(device)
        
        self.machine_times = torch.tensor([0]*3)
        self.prev_job = torch.tensor([2]*3)
        self.running_counter = 0
        self.current_machine = 0
        self.ep_rew  = 0
        
        self.init_buffers()
        
        state = self.machine_buffers[self.current_machine].cpu().numpy()
        
        return state
        

    def fasten_time(self):
        '''
        Returns
        -------
        Fastened Time
        It fastens the time steps by the  minimum numbe rof steps required to 
        make the processing time of one of the machines zero and proceeeds to 
        make the updated processing time zero.
        '''
        for i in range(len(self.machine_times)):
            _min = 0
            cloned_times = self.machine_times.clone()
            sortedcloned_times,_ = cloned_times.sort()
            for val in sortedcloned_times:
                if val == 0:
                    pass
                else:
                    _min = val
                    break
            self.machine_times = self.machine_times - _min
            self.machine_times[self.machine_times<0] = 0
            # print(self.machine_times)
        return +1
        
    def step(self, actions):
        
        '''
        The step function returns the state(observation), reward and boolean(done)
        and information ususally by taking only actions as inputs
        '''
        done = False
        self.reward = 0
        info = {}
        # print('\n\nBefore Machine Times', self.machine_times)
        # print('Before Buffers\n', self.machine_buffers[:,:2])
        # print('Before Machines\n', self.machines[:,:2])
        # print('Output', self.output_buffer[:2])
        # print('Action', action)
        
        # time.sleep(0.5)
        '''
        if the machine times of the current machine is zero and a workpiece is 
        present in the machine buffer then send a work piece to the current machine 
        and add the work piece count as 1 in the current machine and subtract 1 
        from the machine buffer
        
        At the same time now the workpiece times which are zero should be set to the 
        sum of workpiece processing time of the action in the current machine and 
        workpiece setup time of the action of previous job of current machine. 
        
        
    
        '''
        self.current_machine = actions[0]
        action = actions[1]
        
        if self.machine_times[self.current_machine]==0 and \
            self.machine_buffers[self.current_machine][action]>0:
                self.machines[self.current_machine][action] += 1
                self.machine_buffers[self.current_machine][action] -= 1
                
                self.machine_times[self.current_machine] = self.wp_pt[action][self.current_machine]\
                    + self.wp_st[action][self.prev_job[self.current_machine]]
                self.prev_job[self.current_machine] = torch.Tensor([action])
                self.reward -= self.machine_times[self.current_machine]* 0.1
        
        self.running_counter += self.fasten_time()

        cm_array1 = np.array(torch.where(self.machine_times.cpu()==0)[0])
        cm_array2 = np.array(torch.where(self.machine_buffers.sum(axis=1).cpu()!=0)[0])
        cm_array = np.intersect1d(cm_array1, cm_array2)
        
        #fast forward time time empty machine found
        while len(cm_array)==0 and int(self.machine_buffers.sum())!=0:
            self.running_counter += self.fasten_time()
            done = self.terminal()
            if done:
                break
            cm_array1 = np.array(torch.where(self.machine_times.cpu()==0)[0])
            cm_array2 = np.array(torch.where(self.machine_buffers.sum(axis=1).cpu()!=0)[0])
            cm_array = np.intersect1d(cm_array1, cm_array2)
        
        # if len(cm_array)>0:
        #     self.current_machine = int([cm_array if len(cm_array)==1 else cm_array[0]][0])
        #     i = 0
        #     while sum(self.machine_buffers[self.current_machine][:2])==0 and i<len(cm_array):
        #         self.current_machine = int(cm_array[i])
        #         i+=1
        
        for i in cm_array1:
            if self.prev_job[i]!=2 and sum(self.machines[i][:2]) == 1:
                job = self.prev_job[i]
                next_machine_index = int(torch.where(self.wp_seq[job]==i)[0]+1)
                if self.wp_seq[job][next_machine_index] < 3:
                    self.machine_buffers[int(self.wp_seq[job]\
                                                  [next_machine_index])][job] += 1
                elif self.wp_seq[job][next_machine_index] == 3:
                    self.output_buffer[job] += 1
                    # self.reward += 1
                    
                self.machines[i][:2] = 0
        
        done = self.terminal()
        self.ep_rew +=self.reward
        # print('self.reward', self.reward)
        # print('self.ep_rew', self.ep_rew)
        # print(self.output_buffer)
        # time.sleep(0.1)
        state = self.machine_buffers[self.current_machine].cpu().numpy()     
        return state, float(self.reward), done, info
#%% 

if __name__ == '__main__':   
    from stable_baselines3.common.env_checker import check_env
    env = Jssp_env()
    # It will check your custom environment and output additional warnings if needed
    check_env(env, warn=True)
