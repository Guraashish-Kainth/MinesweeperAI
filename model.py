from asyncio.windows_events import NULL
from tracemalloc import start
from keras.models import Sequential
from keras.layers import Conv2D
import keras
from keras.layers import Activation
from keras.layers import BatchNormalization
import os
import tensorflow as tf
import numpy as np
from minesweeper import SIZE_X, SIZE_Y
import math
from keras import backend as K
import random

EPSILON = 1 * math.e**(-8)

class Neural_Net(Sequential):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.add(Conv2D(filters, kernel_size, padding = 'same'))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mean_absolute_error'], run_eagerly=True)        

    def save_model(self, file_name = 'model.pth'):
        model_path = './model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        file_name = os.path.join(model_path, file_name)
        self.save(file_name)
    
class Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model    
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1 = .99)
        self.loss = keras.losses.CategoricalCrossentropy(reduction='none')
        self.accuracy = tf.keras.metrics.CategoricalAccuracy('accuracy')
        self.metric = tf.keras.metrics.CategoricalCrossentropy('metric_categorical_crossentropy')

    def long_train_step (self, start_state, tile, reward, end_state, game_over):  
        tile = np.array(tile)
        pred = []
        states = np.zeros(shape = (tile.shape[0], tile.shape[1], tile.shape[2], 1))
        for s in start_state: 
            temp_state = np.empty((SIZE_X, SIZE_Y))               
            for x in range(SIZE_X):
                for y in range(SIZE_Y):
                    if s[x][y][0] == 0:
                        temp_state[x][y] = -1
                    else:
                        temp_state[x][y] = s[x][y][1]
            temp_state = temp_state.reshape(temp_state.shape[0], temp_state.shape[1], 1)
            temp_state = temp_state[np.newaxis]
            np.append(states, temp_state)
        for state in states:
            pred.append(self.model(state))
        target = np.array(pred)
        for idx in range(len(game_over)):
            q_new = reward[idx]
            if not game_over[idx]:
                next_state = np.empty((SIZE_X, SIZE_Y))
                for x in range(SIZE_X):
                    for y in range(SIZE_Y):
                        if end_state[idx][x][y][0] == 0:
                            next_state[x][y] = -1
                        else:
                            next_state[x][y] = end_state[idx][x][y][1]
                next_state = next_state.reshape(next_state.shape[0], next_state.shape[1], 1)
                q_new = reward[idx] + self.gamma * np.max(self.model(next_state))
            tile_index = np.unravel_index(np.argmax(tile[idx]), tile[idx].shape)
            target[idx][tile_index[0]][tile_index[1]] = q_new        

        for idx in range(len(game_over)):
            with tf.GradientTape() as tape: 
                pred = self.model(states[idx])
                true = target[idx, :, :, :]
                true = true[np.newaxis].reshape(pred.shape)
                loss_value = self.loss(true, pred)
                reduced = tf.reduce_mean(loss_value)

                grads = tape.gradient(reduced, self.model.trainable_variables)

                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables)) 

                self.accuracy.update_state(true, pred)
                self.metric.update_state(true, pred) 

        print(grads)


    def short_train_step(self, start_state, tile, reward, end_state, game_over):
        state = np.zeros((SIZE_X, SIZE_Y))
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if start_state[x][y][0] == 0:
                    state[x][y] = -1
                else:
                    state[x][y] = start_state[x][y][1] + EPSILON

        state = state.reshape(state.shape[0], state.shape[1], 1)
        state = state[np.newaxis]
        state = tf.keras.utils.normalize(state)
        pred = self.model(state)
        target = pred.numpy()
        q_new = reward
        if not game_over:
                next_state = end_state[:,:,1]
                next_state = next_state.reshape(next_state.shape[0], next_state.shape[1], 1)
                q_new = reward + self.gamma * np.max(self.model(next_state))
                
        target[0, np.unravel_index(np.argmax(tile), tile.shape), 0] = q_new

        with tf.GradientTape() as tape: 
            pred = self.model(state)
            loss_value = self.loss(target, pred)
            reduced = tf.reduce_mean(loss_value, keepdims = True)

        grads = tape.gradient(reduced, self.model.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights)) 

        self.accuracy.update_state(target, pred)
        self.metric.update_state(target, pred) 
        #print(grads)
    
    def train_step (self, start_state, tile, mines, reward, end_state, game_over, long_train):          
        if long_train:
            self.long_train_step(start_state, tile, reward, end_state, game_over)
        else:            
            self.short_train_step(start_state, tile, reward, end_state, game_over)      
        #print(self.model.trainable_variables)
        K.clear_session()

