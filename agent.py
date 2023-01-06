from cProfile import label
from logging import PlaceHolder
from os import times
import tensorflow as tf
from minesweeper import Minesweeper, window, SIZE_X, SIZE_Y
from collections import deque
from tkinter import *
import numpy as np
import random
from model import Neural_Net, Trainer
import matplotlib.pyplot as plt
import cProfile



MAX_MEMORY = 10000
BATCH_SIZE = 10
L_RATE = .001

class Agent:
        def __init__(self):
            self.game_count = 0
            self.epsilon = 0
            self.gamma = 0
            self.memory = deque(maxlen = MAX_MEMORY)
            self.model = Neural_Net(1, (5,5))
            self.trainer = Trainer(self.model, L_RATE, self.gamma)

        def get_board(self, game):
            tiles = game.tiles
            board = np.ndarray((SIZE_X, SIZE_Y, 2))
            for x in range(SIZE_X):
                for y in range(SIZE_Y):
                    board[x][y][0] = int(tiles[x][y]['state'])
                    board[x][y][1] = int(tiles[x][y]['state']) * int(tiles[x][y]['mines'])
            return board
        
        def remember(self, start_state, tile, mines, reward, end_state, game_over):
            self.memory.append((start_state, tile, mines, reward, end_state, game_over))

        def train_long_memory(self):
            if len(self.memory) > BATCH_SIZE:
                sample = random.sample(self.memory, BATCH_SIZE)
            else:
                sample = self.memory
            
            start_state, tile, mines, reward, end_state, game_over = zip(*sample)
            self.trainer.train_step(start_state, tile, mines, reward, end_state, game_over, True)

        def train_short_memory(self, start_state, tile, mines, reward, end_state, game_over):
            self.trainer.train_step(start_state, tile, mines, reward, end_state, game_over, False)

        def next_tile(self, start_state):
            self.epsilon = 50 - self.game_count
            if random.randint(0, 200) < self.epsilon:
                unclicked_list = np.argwhere(start_state[:,:,0] == 0)
                index = np.random.choice(len(unclicked_list), size = 1)
                x, y = unclicked_list[int(index)]

            else:
                state = np.empty((SIZE_X, SIZE_Y))
                for sx in range(SIZE_X):
                    for sy in range(SIZE_Y):
                        if start_state[sx][sy][0] == 0:
                            state[sx][sy] = -1
                        else:
                            state[sx][sy] = start_state[sx][sy][1]
                state = state.reshape(state.shape[0], state.shape[1], 1)
                state = state[np.newaxis]
                new_tile = self.model(state)
                new_tile = tf.reshape(new_tile, (SIZE_X, SIZE_Y))
                new_tile = new_tile.numpy()
                for sx in range(SIZE_X):
                    for sy in range(SIZE_Y):
                        if start_state[sx][sy][0] == 0:
                            new_tile[sx][sy] = -1
                x, y = np.unravel_index(np.argmax(new_tile), new_tile.shape)
                
                           
            tile = np.zeros((SIZE_X, SIZE_Y))

            tile[x][y] = 1

            return tile
def train():
    plot_times = []
    plot_boxes = []
    plot_mean_boxes = []
    record_score = 0
    record_time = 100000000000
    agent = Agent()
    window = Tk() 
    window.title("Minesweeper")
    minesweeper = Minesweeper(window)
    while agent.game_count < 1000:      
        start_state = agent.get_board(minesweeper)

        tile = agent.next_tile(start_state)
        tile_index = np.unravel_index(np.argmax(tile), tile.shape)
        reward, game_over, boxes, time = minesweeper.onClick(minesweeper.tiles[tile_index[0]][tile_index[1]])

        #reward, game_over, boxes, time = minesweeper.onClickWrapper(tile_index[0], tile_index[1])

        end_state = agent.get_board(minesweeper)

        agent.train_short_memory(start_state, tile, minesweeper.tiles[tile_index[0]][tile_index[1]]['mines'], reward, end_state, game_over)

        agent.remember(start_state, tile, minesweeper.tiles[tile_index[0]][tile_index[1]]['mines'], reward, end_state, game_over)

        if game_over:
            minesweeper.restart()
            agent.game_count += 1
            agent.train_long_memory()

            if boxes > record_score:
                record_time = time.total_seconds()
                agent.model.save_model()
            elif boxes == record_score:
                if time.total_seconds() < record_time:
                    record_time = time.total_seconds()
                    agent.model.save_model()
            if boxes > record_score:
                record_score = boxes
            print('Game', agent.game_count, 'Boxes', boxes, 'Time', time.total_seconds(), 'Record Time', record_time)
            if(minesweeper.won):
                scaler = 1
            else:
                scaler = -1

            time_edited = time.total_seconds()
            plot_times.append(time_edited)

            plot_boxes.append(boxes)
            
            mean_boxes = boxes / agent.game_count

            plot_mean_boxes.append(mean_boxes)

            #plt.clf()

            #plt.xlabel('Game Count')
            #plt.ylabel('Score')

            #plt.plot(range(agent.game_count), plot_times,  label = 'Time')     
            #plt.plot(range(agent.game_count), plot_boxes, label = 'Boxes')
            #plt.plot(range(agent.game_count), plot_mean_boxes, label = 'Mean Boxes')  
            #leg = plt.legend(loc='upper left')

            #plt.show()            




if __name__ == '__main__':
    plt.ion()    
    train()