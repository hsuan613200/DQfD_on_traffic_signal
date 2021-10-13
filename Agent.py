import numpy as np
import pickle
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
import keras.backend as K
from memory_buffer import MemoryBuffer

class DQNAgent():
    def __init__(self, env, dueling = False, double = False, with_per = False):
        #state size
        self.state_size = env.state["North"].shape
        #phase size
        self.phase_size = len(env.phase_state)
        #action space
        self.action_size = 2
        #experience buffer
        self.exp_buffer = MemoryBuffer(20000, with_per)
        self.demo_buffer = MemoryBuffer(20000, with_per)        
  
        self.dueling = dueling

        self.double = double
        
        #discount factor
        self.discount_factor = 0.99    # discount rate
        
        #set for greedy
        ######################################################
        self.epsilon = 0.5  
        self.eps_min = 0.1
        self.eps_max = 0.2
        self.eps_decay_steps = 50
        #####################################################
        
        #deep learning param
        self.batch_size = 64
        self.learning_rate = 0.0001
        
        self.main_model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()
 
    #network model
    def build_model(self):
        positions = ["North","West","South","East"]
        input_data = []
        pos_output = []
        for pos in positions:
            x_input = layers.Input(shape=self.state_size, dtype='float32', name=pos+"_input")
            x = layers.Conv2D(16, (2, 3),padding='valid', activation='relu')(x_input)
            x = layers.MaxPooling2D((1, 2))(x)
            x = layers.Conv2D(32, (2, 2),padding='valid', activation='relu')(x)
            x = layers.MaxPooling2D((1, 2))(x)
            x_output = layers.Flatten(name=pos+"_output")(x)
            input_data.append(x_input)
            pos_output.append(x_output)
        x = layers.concatenate(pos_output)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        
        phase_input = layers.Input(shape=(self.phase_size,), name='phase_input')
        phase_output = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(phase_input)
        input_data.append(phase_input)
        
        x = layers.concatenate([x,phase_output])
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        
        if (self.dueling):
#             x = layers.Dense(self.action_size + 1, activation='linear')(x)
            x = layers.Dense(self.action_size + 1, activation='linear', kernel_regularizer=regularizers.l2(0.01))(x)
            action_output = layers.Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True),
                                          output_shape=(self.action_size,), name='action_output')(x)
        
        else:
            action_output = layers.Dense(self.action_size, activation='linear',name='action_output')(x)
            
        model = models.Model(inputs=input_data, outputs=[action_output])
        model.compile(loss='mse', optimizer = optimizers.Adam(lr=self.learning_rate))
        
        return model
    
    #weight copy
    def update_target_network(self):
        weights = self.main_model.get_weights()
        self.target_model.set_weights(weights)
    
    #save the <s,a,r,s'> the buffer
    def remember(self, state, phase_state, action, reward, next_state, next_phase_state, error=None):
        self.exp_buffer.memorize(state, phase_state, action, reward, next_state, next_phase_state, error)
       
    
    #choose action with greedy policy
    def choose_action(self, state, phase_state, step):
        actions = self.main_model.predict({'North_input': state["North"][np.newaxis, :], 
                                           'West_input': state["West"][np.newaxis, :], 
                                           'South_input': state["South"][np.newaxis, :],
                                           'East_input': state["East"][np.newaxis, :],
                                           'phase_input': phase_state.reshape(1,-1)})
        action = np.argmax(actions, axis=-1)
        action = self.epsilon_greedy(action, step)
        return int(action)
    
    #choose action with greedy policy
    def choose_action_without_greedy(self, state, phase_state):
        actions = self.main_model.predict({'North_input': state["North"][np.newaxis, :], 
                                   'West_input': state["West"][np.newaxis, :], 
                                   'South_input': state["South"][np.newaxis, :],
                                   'East_input': state["East"][np.newaxis, :],
                                   'phase_input': phase_state.reshape(1,-1)})
        action = np.argmax(actions, axis=-1)
        return int(action)
        
    #greedy policy
    def epsilon_greedy(self, action, step):
        self.epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return action
    
    #agent(network) training
    def training(self):
        all_state, action, reward, all_next_state, idx = self.exp_buffer.sample_batch(self.batch_size)
        # states

        y = self.main_model.predict(all_state)

        next_Q = self.target_model.predict(all_next_state)

        if (self.double):
            next_act = self.main_model.predict(all_next_state)
            max_act = np.argmax(next_act, axis=1)

            discount_Q = [self.discount_factor * next_Q[index,max_act[index]] for index in range(len(all_state["North_input"]))]

            target_y = reward + discount_Q

        else:

            target_y = reward + self.discount_factor * np.max(next_Q, axis=-1)

        if(self.exp_buffer.with_per):
            for i in range(len(action)):
                error = np.abs(target_y[i] - y[i][action[i]])
                self.exp_buffer.update(idx[i], error)

        for index in range(len(action)):
            y[index][int(action[index])] = target_y[index]

        history = self.main_model.fit(all_state,{'action_output': y}, epochs=1, verbose=0)
        loss = history.history['loss']
        return loss

    #pretrain with demostration data
    def pre_training(self):
        all_state, action, reward, all_next_state, idx = self.exp_buffer.sample_batch(self.batch_size)
        # states

        y = self.main_model.predict(all_state)

        supervise = 0
        act = np.argmax(y, axis=1)
        for i in range(len(action)):
            if reward[i] > 0 and act[i] != action[i]:
                supervise += 1
            elif reward[i] < 0 and act[i] != action[i]:
                supervise -= 1

        next_Q = self.target_model.predict(all_next_state)

        target_y = reward + self.discount_factor * np.max(next_Q, axis=-1)

        if(self.exp_buffer.with_per):
            for i in range(len(action)):
                error = np.abs(target_y[i] - y[i][action[i]])
                self.exp_buffer.update(idx[i], error)
                
        for index in range(len(action)):
            y[index][int(action[index])] = target_y[index]

        history = self.main_model.fit(all_state,{'action_output': y}, epochs=1, verbose=0)
        loss = history.history['loss']
        return supervise, loss
        
    #load model
    def load(self, name):
        self.main_model.load_weights(name)
    #save model
    def save(self, name):
        self.main_model.save_weights(name)
    #load demostration memory buffer
    def load_demos(self, fname):
        with open(fname, 'rb') as demo_file:
            return pickle.load(demo_file)