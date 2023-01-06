from tensorflow import keras
import tensorflow as tf
from agent import Agent
import numpy as np
from tkinter import *
from minesweeper import Minesweeper, window, SIZE_X, SIZE_Y

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # fraction of memory
config.gpu_options.visible_device_list = "0"

keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

if __name__ == '__main__':
    path = './model/model.pth'
    model = keras.models.load_model(path)
    agent = Agent()
    window = Tk() 
    window.title("Minesweeper")
    minesweeper = Minesweeper(window)
    games_won = 0
    while agent.game_count < 200:        
        start_state = agent.get_board(minesweeper)

        tile = agent.next_tile(start_state)
        tile_index = np.unravel_index(np.argmax(tile), tile.shape)
        reward, game_over, boxes, time = minesweeper.onClick(minesweeper.tiles[tile_index[0]][tile_index[1]])

        end_state = agent.get_board(minesweeper)        

        if game_over:
            minesweeper.restart()
            agent.game_count += 1
            if minesweeper.won == True:
                games_won += 1    
    print("Games Played:", agent.game_count)
    print('Games Won:', games_won)

            

            
