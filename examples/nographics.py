from LatteLab import *
import matplotlib.pyplot as plt

def main():
    config = Config(total_timesteps=1000, window_dimensions=(1280,720), grid_size=(64,64,64), use_graphics=False).get()
    lbm = LBM(config)
    lbm.run()
    rho, u = lbm.getMacroscopicData()
#    lbm.exportMacroscopicData()
if __name__ == '__main__':   
    main()