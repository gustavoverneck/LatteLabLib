from LatteLab import *

def main():
    config = Config(total_timesteps=10000, window_dimensions=(1280,720), grid_size=(64,64,64)).get()
    lbm = LBM(config)
    lbm.run()

if __name__ == '__main__':   
    main()