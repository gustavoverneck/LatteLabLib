# test.py
from LatteLab import *
from LatteLab.sim_flags import FLAG_F, FLAG_S, FLAG_E

def uIC(x, y, z):
    if x < 10:
        return 1
    else:
        return 0
    
def rhoIC(x, y, z):
    if x < 10:
        return 1
    else:
        return 0

def flagsIC(x, y, z):
    return FLAG_F


def main():
    config = Config(total_timesteps=10000, window_dimensions=(1280,720), grid_size=(64,64,64)).get()
    lbm = LBM(config)
    lbm.setInitialConditions(uIC, target='u')
    lbm.setInitialConditions(rhoIC, target='rho')
    lbm.setInitialConditions(flagsIC, target='flags')
    lbm.run()

if __name__ == '__main__':   
    main()