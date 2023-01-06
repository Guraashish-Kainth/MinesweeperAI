from agent import train
import cProfile

if __name__ == '__main__':
    cProfile.run('train()', 'restats')
    import pstats
    p = pstats.Stats('restats')
    p.sort_stats('time').print_stats(5)
