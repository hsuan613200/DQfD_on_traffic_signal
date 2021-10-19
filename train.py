from Vissim import Vissim
from DQNAgent import DQNAgent
import os
import numpy as np
import random
import time
import csv

#use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#open and initial environment
Filename = os.path.abspath(os.getcwd())
Filename += r"\vissim\training\sequential_four_phase.inpx"
env = Vissim(Filename)
env.initialize_env()

#build agent
agent = DQNAgent(env, double = True, dueling = True, with_per = True)
agent.load("fixedtime_regular/demo_agent.h5")
agent.exp_buffer = agent.load_demos('fixedtime_regular/fixedtime.p')

#set simulation time
simulation_time = 4500    #單位:s(熱機900秒+模擬3600秒)

#light parameter
detect_interval = 5
min_green_time = 10
yellow_time = 3
all_red = 2

#copy or train model
global_step = 0
copy_steps = 100 
steps_train = 4 
start_steps = 0

save_path = "fixedtime_regular/"

#record performance
#############################################################
with open(save_path + 'phase_process.csv', 'w', newline='') as csvfile:
    
    writer = csv.writer(csvfile)
    
    writer.writerow(['episodes', 'mean_length_0', 'mean_length_1', 'mean_length_2', 'mean_length_3',
                    'episode_ratio_0', 'episode_ratio_1', 'episode_ratio_2', 'episode_ratio_3', 'mean_cycle_time'])


with open(save_path + 'process.csv', 'w', newline='') as csvfile:
    
    writer = csv.writer(csvfile)
    
    writer.writerow(['episodes', 'reward', 'travel_time', 'stop_time(car)', 'stop_time(scooter)', 'loss', 'simulation_time'])
        
#############################################################

threshold = 0
episode = 1
#start train
while threshold < 50:
    #record time
    start = time.time()
    
    #set random seed
    env.randseed_set(episode)
    
    #initial environment
    env.initialize_env()
    
    #get performance
    episodic_waiting_time = {"car":0,"scooter":0}
    episodic_reward = 0
    
    #這是避免第一次還沒做動作就儲存經驗用的
    first = True
    
    #record loss
    episodic_loss = []
    
    #warm up
    env.warm_up(895)
    
    #get control and signal info
    env.take_phase_control()
    privious_phase = env.current_phase
    #change to next
    next_phase = (env.current_phase + 1)%len(env.phase_composition)
    env.change_to_next_phase(yellow_time, all_red, next_phase)

    
    while env.time < simulation_time:    
        #make sure over min green time
        if env.phase_time >= min_green_time:
            env.update_state()
            global_step += 1
            #make sure have <s,a,r,s'>
            if first == False:
                #calculate reward
                #########################################################################
                car, left_car, scooter = env.get_specific_throughtput(privious_phase)
                full_left_turn = env.left_turn_full_detect()
                reward = ((car + 3 * left_car + 0.3 * scooter)
                          -0.005*(1.78*stop_time_car + 1.19*stop_time_scooter)
                         -5*change)
                #########################################################################
                
                #save
                agent.remember(env.previous_state, env.previous_phase_state, action, reward, env.state, env.phase_state)
                
                #cumulate reward
                episodic_reward += reward 
            
            else:
                first = False
            
            #action choose
            action = agent.choose_action(env.state, env.phase_state, episode-1)

            
            #change to next
            if action == 1:
                #record phase info
                privious_phase = env.current_phase
                next_phase = (env.current_phase + 1)%len(env.phase_composition)
                change = 1
                #take change action
                env.change_to_next_phase(yellow_time, all_red, next_phase)
                #calculate performance
                stop_time_car, stop_time_scooter = env.get_stop_time()
                episodic_waiting_time["car"] += stop_time_car
                episodic_waiting_time["scooter"] += stop_time_scooter
            
            #extand current
            else:
                change = 0

            #check for training
            if global_step % steps_train == 0 and global_step >= start_steps:
                
                loss = agent.training()

                episodic_loss.append(loss)
                
            #check for coping
            if (global_step + 1) % copy_steps == 0 and global_step >= start_steps:
                agent.update_target_network()
        
        #simulation next interval
        env.next_time_step(detect_interval)
        
        #calculate performance
        stop_time_car, stop_time_scooter = env.get_stop_time()
        episodic_waiting_time["car"] += stop_time_car
        episodic_waiting_time["scooter"] += stop_time_scooter
        
        
    ###########sim end#############
    
    #episode loss
    total_episodic_loss = np.sum(episodic_loss)
    
    #save model
    if total_episodic_loss < min_loss and global_step >= start_steps:
        agent.save(save_path + "dqn_agent_{}.h5".format(episode))
        min_loss = total_episodic_loss
        
    if episodic_reward > 1600: # an episode reward by well trained DQN agent
        threshold += 1
        
    #save latest model
    agent.save(save_path + "lastest_dqn_agent.h5")
    
    #get performance
    travel_time = env.get_travel_time()
    
    mean_phase_time, mean_phase_ratio, mean_cycle_time = env.get_signal_result()
    
    #end time
    end = time.time()
          
    print('episodes', episode, 'Reward', episodic_reward, "travel_time", travel_time,
          "loss", total_episodic_loss, "simulation_time", end-start)
    with open(save_path + 'process.csv','a', newline='') as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([episode, episodic_reward, travel_time, episodic_waiting_time["car"], episodic_waiting_time["scooter"],
                        total_episodic_loss, end-start])

    with open(save_path + 'phase_process.csv','a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([episode, mean_phase_time['0'], mean_phase_time['1'], mean_phase_time['2'], mean_phase_time['3'],
                        mean_phase_ratio['0'], mean_phase_ratio['1'], mean_phase_ratio['2'], mean_phase_ratio['3'], mean_cycle_time])

    episode += 1