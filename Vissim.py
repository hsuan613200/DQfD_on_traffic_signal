import numpy as np
import copy
import win32com.client as com
from collections import Counter
import random

class Vissim():
    def __init__(self, Filename):
        #open Vissim window
        self.Vissim = com.Dispatch("Vissim.Vissim")
        self.Vissim.LoadNet(Filename)
        #using quick mode
        self.Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode",1)
        #environment time
        self.time = 0
        #phase passed time
        self.phase_time = 0
        #calculate veh
        self.phase_car = 0
        self.phase_scooter = 0
        self.arr = 0
        #delay second
        self.delay = 0
        self.delay_stop = 0
        #current phase
        self.current_phase = 0
        #signal control
        self.SC_number = 1
        #record state
        self.previous_state = None
        self.state = None
        self.previous_phase_state = None
        self.phase_state = None
        #connect to the road set in Vissim
        self.flow = [1,2,3,4,11,12,13,14]
        #connect to the phase set in Vissim
        self.phase_composition = [[1,3],[11,13],[2,4],[12,14]]
        #the road for every direction
        self.position = {"North":[1,11,21,31,41,51],
                         "West":[2,12,22,32,42,52],
                         "South":[3,13,23,33,43,53],
                         "East":[4,14,24,34,44,54]}
        self.lanes = [1,2,3,4]
        self.left_lanes = [11,12,13,14]
        self.two_step_left_turn = [31,32,33,34]
        self.turn_left_lanes = [51,52,53,54]
        #collect all throughput
        self.datacollection_key = 100
        #transfer the number
        self.no_to_name = {"1":"North", "2":"West", "3":"South", "4":"East", 
                           "11":"North_left", "12":"West_left", "13":"South_left", "14":"East_left"}  
        
        #for performance
        self.phase_time_record = {"0":0, "1":0, "2":0, "3":0}
        self.phase_time_temp_record = {"0":0, "1":0, "2":0, "3":0}
        self.phase_num_temp_record = {"0":0, "1":0, "2":0, "3":0}
        self.cycle = 0
    
    #initial environment    
    def initialize_env(self):
        self.time = 0
        self.phase_time = 0
        self.phase_car = 0
        self.phase_scooter = 0
        self.current_phase = 0
        self.update_state()
        self.phase_time_record = {"0":0, "1":0, "2":0, "3":0}
        self.phase_time_temp_record = {"0":0, "1":0, "2":0, "3":0}
        self.phase_num_temp_record = {"0":0, "1":0, "2":0, "3":0}
        self.cycle = 0
        
    #set for random seed  
    def randseed_set(self, seed):
        self.random_seed = seed
        self.Vissim.Simulation.SetAttValue('RandSeed', self.random_seed)
        
    #collect throughtput
    def get_throughtput(self):
        throughtput_collection = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(self.datacollection_key)
        car = throughtput_collection.AttValue("Vehs(Current,Last,10)")
        scooter = throughtput_collection.AttValue("Vehs(Current,Last,70)")
        return car,scooter
    
    #simlulate next time step      
    def next_time_step(self, time_step):
        self.time += time_step
        self.phase_time += time_step
        Sim_break_at = self.time
        self.Vissim.Simulation.SetAttValue('SimBreakAt', Sim_break_at)
        self.Vissim.Simulation.RunContinuous()
        car, scooter = self.get_throughtput()
        if car != None:
            self.phase_car += car
        if scooter != None:
            self.phase_scooter += scooter
        #get performance after warm up network
        arr_all, delay_all, delay_stop_all = self.get_performance()
        if self.time > 900 and arr_all != None:
            self.arr += arr_all
        if self.time > 900 and delay_all != None:
            self.delay += delay_all
        if self.time > 900 and delay_stop_all != None:
            self.delay_stop += delay_stop_all
            
    #network warm up
    def warm_up(self, WarmUpTime):
        self.Vissim.Simulation.SetAttValue('SimBreakAt', WarmUpTime)
        self.Vissim.Simulation.RunContinuous()
        self.time = WarmUpTime
    
    #break time
    def break_time(self, Time):
        self.Vissim.Simulation.SetAttValue('SimBreakAt', Time)
        self.Vissim.Simulation.RunContinuous()
        self.time = Time
        
    
    #update
    def update_state(self):
        self.previous_state = self.state
        self.previous_phase_state = self.phase_state
        self.state = self.veh_matrix()
        self.phase_state = self.get_phase_state()
    
    #get state
    def veh_matrix(self):
        state_matrix = {}
        for position in self.position.keys():
            desired_speed = 50
            car_pos = np.zeros((3,30))
            scooter_pos = np.zeros((3,30))
            car_speed = np.zeros((3,30))
            scooter_speed = np.zeros((3,30))
            for no in self.position[position]:
                vehs = self.Vissim.Net.Links.ItemByKey(no).Vehs.GetMultipleAttributes(("VehType","Lane","Pos","Speed"))
                if no < 10:
                    for veh in vehs:
                        lane = int(veh[1][-1])
                        pos = min(29,int(veh[2]//5))
                        if veh[0] == "100":
                            car_pos[-lane][pos] += 1 
                            car_speed[-lane][pos] += veh[3]
                        else:
                            scooter_pos[-lane][pos] += 1 
                            scooter_speed[-lane][pos] += veh[3]
                elif no < 30:
                    if no < 20:
                        for veh in vehs:
                            lane = 0
                            pos = int((veh[2]+110)//5)
                            car_pos[lane][pos] += 1
                            car_speed[lane][pos] += veh[3]
                    else:
                        for veh in vehs:
                            lane = 0
                            pos = int((veh[2]+90)//5)
                            car_pos[lane][pos] += 1
                            car_speed[lane][pos] += veh[3]
                elif no < 50:
                    for veh in vehs:
                        pos = 29
                        scooter_pos[2][pos] += 1
                        scooter_speed[2][pos] += veh[3]
                else:
                    for veh in vehs:
                        pos = 29
                        car_pos[0][pos] += 1
                        car_speed[0][pos] += veh[3]
                    
            
            for lane in range(len(car_pos)):
                for pos in range(len(car_pos[lane])):
                    if car_pos[lane][pos] == 0:
                        car_speed[lane][pos] = desired_speed
                    else:
                        car_speed[lane][pos] = car_speed[lane][pos]/car_pos[lane][pos]
                    if scooter_pos[lane][pos] == 0:
                        scooter_speed[lane][pos] = desired_speed
                    else:
                        scooter_speed[lane][pos] = scooter_speed[lane][pos]/scooter_pos[lane][pos]
                        
            state_matrix[position] = np.array([car_pos,scooter_pos,car_speed,scooter_speed]).transpose((1,2,0))
            
        return state_matrix
    

    #take the signal control 
    def take_phase_control(self):
        SignalController = self.Vissim.Net.SignalControllers.ItemByKey(self.SC_number)
        
        for sg in SignalController.SGs:
            sg.SetAttValue("ContrByCOM", True)
            
        #get current signal
        while True:
            if SignalController.SGs.ItemByKey(1).AttValue("SigState")=="GREEN":
                self.current_phase = 0
                break
            elif SignalController.SGs.ItemByKey(21).AttValue("SigState")=="GREEN":
                self.current_phase = 1
                break
            elif SignalController.SGs.ItemByKey(2).AttValue("SigState")=="GREEN":
                self.current_phase = 2
                break
            elif SignalController.SGs.ItemByKey(22).AttValue("SigState")=="GREEN":
                self.current_phase = 3
                break
            else:
                self.Vissim.Simulation.RunSingleStep()
    
    #simulate for peak factor
    def set_random_flow(self,episode):
        input_list = {"0":[200,200,500,700],"1":[200,200,550,650],"2":[200,200,600,600],"3":[200,250,450,700],"4":[200,250,500,650],
                    "5":[200,250,550,600],"6":[200,300,400,700],"7":[200,300,450,650],"8":[200,300,500,600],"9":[200,300,550,550],
                    "10":[200,350,350,700],"11":[200,350,400,650],"12":[200,350,450,600],"13":[200,350,500,550],"14":[200,400,400,600],
                    "15":[200,400,450,550],"16":[200,400,500,500],"17":[200,450,450,500],"18":[250,250,400,700],"19":[250,250,450,650],
                    "20":[250,250,500,600],"21":[250,250,550,550],"22":[250,300,350,700],"23":[250,300,400,650],"24":[250,300,450,600],
                    "25":[250,300,500,550],"26":[250,350,350,650],"27":[250,350,400,600],"28":[250,350,450,550],"29":[250,350,500,500],
                    "30":[250,400,400,550],"31":[250,400,450,500],"32":[250,450,450,450],"33":[300,300,300,700],"34":[300,300,350,650],
                    "35":[300,300,400,600],"36":[300,300,450,550],"37":[300,300,500,500],"38":[300,350,350,600],"39":[300,350,400,550],
                    "40":[300,350,450,500],"41":[300,400,400,500],"42":[300,400,450,450],"43":[350,350,350,550],"44":[350,350,400,500],
                    "45":[350,350,450,450],"46":[350,400,400,450],"47":[400,400,400,400]}
        
        for i in range(1,5):
            random.seed(episode*i)
            flow = input_list[str(random.randint(0,47))]
            random.shuffle(flow)
            self.Vissim.Net.VehicleInputs.ItemByKey(i).SetAttValue("Volume(2)", int(flow[0]*4))
            self.Vissim.Net.VehicleInputs.ItemByKey(i).SetAttValue("Volume(3)", int(flow[1]*4))
            self.Vissim.Net.VehicleInputs.ItemByKey(i).SetAttValue("Volume(4)", int(flow[2]*4))
            self.Vissim.Net.VehicleInputs.ItemByKey(i).SetAttValue("Volume(5)", int(flow[3]*4))
    
    
    #change to next phase
    def change_to_next_phase(self, yellow_time, all_red, next_phase):
        
        SignalController = self.Vissim.Net.SignalControllers.ItemByKey(self.SC_number)
        
        for i in self.phase_composition[self.current_phase]:
            SignalController.SGs.ItemByKey(i).SetAttValue("SigState", "AMBER")
            
        self.next_time_step(yellow_time)   
        
        for i in self.phase_composition[self.current_phase]:
            SignalController.SGs.ItemByKey(i).SetAttValue("SigState", "RED")
        
        self.next_time_step(all_red)
        
        self.phase_time_temp_record[str(self.current_phase)] += self.phase_time
        self.phase_num_temp_record[str(self.current_phase)] += 1
        
        cycle_finish = True
        for i in self.phase_num_temp_record.keys():
            if self.phase_num_temp_record[str(i)] == 0:
                cycle_finish = False
                break
        
        if cycle_finish:
            for i in self.phase_time_record.keys():
                self.phase_time_record[str(i)] += self.phase_time_temp_record[str(i)]
            
            self.phase_time_temp_record = {"0":0, "1":0, "2":0, "3":0}
            self.phase_num_temp_record = {"0":0, "1":0, "2":0, "3":0}
            self.cycle += 1
        
        if self.current_phase % 2 == 1:
            for i in self.phase_composition[self.current_phase]:
                SignalController.SGs.ItemByKey(i+10).SetAttValue("SigState", "RED")
            
        
        for i in self.phase_composition[next_phase]:
            SignalController.SGs.ItemByKey(i).SetAttValue("SigState", "GREEN")
            SignalController.SGs.ItemByKey(i+10).SetAttValue("SigState", "GREEN")
                
        self.current_phase = next_phase
        self.phase_time = 0
        self.phase_car = 0
        self.phase_scooter = 0
    
    #signal result      
    def get_signal_result(self):
        total_cycle_time = 0
        mean_phase_time = {}
        mean_phase_ratio = {}
        for i in self.phase_time_record.keys():
            if self.phase_time_record[str(i)] == 0:
                mean_phase_time[str(i)] = 0
            else:
                mean_phase_time[str(i)] = self.phase_time_record[str(i)]/self.cycle
                
            total_cycle_time += self.phase_time_record[str(i)]
            
        for i in self.phase_time_record.keys():
            if self.phase_time_record[str(i)] == 0:
                mean_phase_ratio[str(i)] = 0
            else:
                mean_phase_ratio[str(i)] = self.phase_time_record[str(i)]/total_cycle_time
        
        mean_cycle_time = total_cycle_time/self.cycle
        
        return mean_phase_time, mean_phase_ratio, mean_cycle_time
    
    #transfer to one hot
    def get_phase_state(self):
        phase_state = [0] * len(self.phase_composition)
        phase_state[self.current_phase] = 1
        return np.array(phase_state)                
    
    #get network performance
    
    #travel time
    def get_travel_time(self):
        conv = lambda i : i or 0
        car_travel_time = np.asarray(self.Vissim.Net.VehicleTravelTimeMeasurements.GetMultiAttValues('TravTm(Current,Last,10)'))[:,1]
        car_travel_time = np.asarray([conv(i) for i in car_travel_time])
        
        scooter_travel_time = np.asarray(self.Vissim.Net.VehicleTravelTimeMeasurements.GetMultiAttValues('TravTm(Current,Last,70)'))[:,1]
        scooter_travel_time = np.asarray([conv(i) for i in scooter_travel_time])
        
        car_num = np.asarray(self.Vissim.Net.VehicleTravelTimeMeasurements.GetMultiAttValues('Vehs(Current,Last,10)'))[:,1]
        car_num = np.asarray([conv(i) for i in car_num])
        
        scooter_num = np.asarray(self.Vissim.Net.VehicleTravelTimeMeasurements.GetMultiAttValues('Vehs(Current,Last,70)'))[:,1]
        scooter_num = np.asarray([conv(i) for i in scooter_num])
        
        travel_time = np.sum(((car_travel_time * car_num) + (scooter_travel_time * scooter_num)))  / np.sum((car_num + scooter_num))
        return travel_time


    #stop time
    def get_stop_time(self):
        performance = self.Vissim.Net.VehicleNetworkPerformanceMeasurement
        total_stop_time_car =  performance.AttValue('DelayStopTot(Current,Last,10)')
        total_stop_time_scooter =  performance.AttValue('DelayStopTot(Current,Last,70)')
        
        return total_stop_time_car, total_stop_time_scooter
    
    #delay time
    def get_performance(self):
        performance = self.Vissim.Net.VehicleNetworkPerformanceMeasurement
        #arr_car =  performance.AttValue('VehArr(Current,Last,10)')
        #arr_scooter =  performance.AttValue('VehArr(Current,Last,70)')
        #delay_time_car =  performance.AttValue('DelayTot(Current,Last,10)')
        #delay_time_scooter =  performance.AttValue('DelayTot(Current,Last,70)')
        #stop_car =  performance.AttValue('DelayStopTot(Current,Last,10)')
        #stop_scooter =  performance.AttValue('DelayStopTot(Current,Last,70)')
        arr_all = performance.AttValue('VehArr(Current,Last,All)') 
        delay_all = performance.AttValue('DelayTot(Current,Last,All)')
        delay_stop_all = performance.AttValue('DelayStopTot(Current,Last,70)')
        
        return arr_all, delay_all, delay_stop_all
    
    #total vehicle
    def get_vehact(self):
        performance = self.Vissim.Net.VehicleNetworkPerformanceMeasurement
        #act_car =  performance.AttValue('VehAct(Current,Last,10)')
        #act_scooter =  performance.AttValue('VehAct(Current,Last,70)')
        act_all =  performance.AttValue('VehAct(Current,Last,all)')

        return act_all

    
    ############################################################################
    #####func for record demonstration#####
    #############################################################################
            
    #run one second to control
    def first_second(self, sg):
        SignalController = self.Vissim.Net.SignalControllers.ItemByKey(self.SC_number)
        SG_to_SH = {1:[1,3], 2:[11,13], 3:[2,4], 4:[12,14]}
        
        for i in SG_to_SH[sg]:
            SignalController.SGs.ItemByKey(i).SetAttValue("SigState", 'green')
            SignalController.SGs.ItemByKey(i+10).SetAttValue("SigState", 'green')
            
        self.Vissim.Simulation.SetAttValue('SimBreakAt', 1)
        self.Vissim.Simulation.RunContinuous()
    
    #simulation signal table data from Vissim
    def set_signal(self, sc, image):
        SignalController = self.Vissim.Net.SignalControllers.ItemByKey(self.SC_number)
        SG_to_SH = {1:[1,3], 2:[11,13], 3:[2,4], 4:[12,14]}
        
        if sc%2==1:
            if image=='green':
                for i in SG_to_SH[sc]:
                    SignalController.SGs.ItemByKey(i).SetAttValue("SigState", image)
                    SignalController.SGs.ItemByKey(i+10).SetAttValue("SigState", image)
            else:
                for i in SG_to_SH[sc]:
                    SignalController.SGs.ItemByKey(i).SetAttValue("SigState", image)
        else:
            if image=='green':
                for i in SG_to_SH[sc]:
                    SignalController.SGs.ItemByKey(i+10).SetAttValue("SigState", image)
                    
            elif image=='amber':
                for i in SG_to_SH[sc]:
                    SignalController.SGs.ItemByKey(i).SetAttValue("SigState", image)
                    
            else:
                for i in SG_to_SH[sc]:
                    SignalController.SGs.ItemByKey(i).SetAttValue("SigState", image)
                    SignalController.SGs.ItemByKey(i+10).SetAttValue("SigState", image)

    #adjust the volume
    def set_demo_flow(self, num):
        flow = {1:[1400, 1800, 2000, 1200], 2:[1200, 1400, 1800, 2000],
                3:[2000, 1200, 1400, 1800], 4:[1800, 2000, 1200, 1400]}
        if num == 1:
            for i in range(1,5):
                for j in range(1,6):
                    volume = "Volume(" + str(j) + ")"
                    self.Vissim.Net.VehicleInputs.ItemByKey(i).SetAttValue(volume, 1600)

        elif num == 41:
            for i in range(1,5):
                for j in range(2,6):
                    volume = "Volume(" + str(j) + ")"
                    self.Vissim.Net.VehicleInputs.ItemByKey(i).SetAttValue(volume, flow[1][j-2])

        elif num == 71:
            for i in range(1,5):
                for j in range(2,6):
                    volume = "Volume(" + str(j) + ")"
                    self.Vissim.Net.VehicleInputs.ItemByKey(i).SetAttValue(volume, flow[i][j-2])
    #check the demonstration is correct
    def get_demo_specific_throughtput(self, phase, time):
        car = 0
        left_car = 0
        scooter = 0
        for i in self.phase_composition[phase]:
            throughtput_collection = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(i)
            if i < 10:
                for j in range(time-4, time+1):
                    car += throughtput_collection.AttValue("Vehs(Current,"+str(j)+",10)")
                    scooter += throughtput_collection.AttValue("Vehs(Current,"+str(j)+",70)")
            else:
                for j in range(time-4, time+1):
                    left_car += throughtput_collection.AttValue("Vehs(Current,"+str(j)+",10)")
  
        return car,left_car,scooter
                    
                    
    def get_demo_throughtput(self, time):
        throughtput_collection = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(self.datacollection_key)
        for i in range(time-4, time+1):
            car = throughtput_collection.AttValue("Vehs(Current,"+str(i)+",10)")
            scooter = throughtput_collection.AttValue("Vehs(Current,"+str(i)+",70)")
        return car,scooter
    
    def get_demo_stop_time(self, time):
        performance = self.Vissim.Net.VehicleNetworkPerformanceMeasurement
        for i in range(time-4, time+1):
            total_stop_time_car =  performance.AttValue("DelayStopTot(Current,"+str(i)+",10)")
            total_stop_time_scooter =  performance.AttValue("DelayStopTot(Current,"+str(i)+",70)")

        return total_stop_time_car, total_stop_time_scooter