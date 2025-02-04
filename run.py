from LatteLabLib import *

def main1():
    config = Config(total_timesteps=10000, window_dimensions=(1280,720), grid_size=(64,64,64)).get()
    lbm = LBM(config)
    lbm.run()

def main2():
    graphics = Graphics()
    graphics.setColorScheme("inferno")
    graphics.render()


if __name__ == '__main__':   
    main1()