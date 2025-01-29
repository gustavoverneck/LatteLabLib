# config.py


available_velocities_sets = ['D2Q9', 'D3Q19']
avilable_simtypes = ['fluid']                       # Add plasma in the future

class Config:
    """
    Config class for managing simulation configuration settings.
    Attributes:
        config (dict): A dictionary containing the configuration settings.
    Methods:
        __init__(velocities_set='D2Q9', use_temperature=True, simtype='fluid', grid_size=(100, 100, 0)):
            Initializes the Config object with default or provided settings.
        get():
            Retrieves the current configuration.
        checkConfig():
    """
    
    def __init__(self, velocities_set='D2Q9', use_temperature=True, simtype='fluid', grid_size=(100, 100, 0), viscosity=0.1):
        """
        Initialize the configuration for the simulation.
        Parameters:
        velocities_set (str): The set of velocities to use for the simulation. Default is 'D2Q9'.
        use_temperature (bool): Flag to indicate whether to use temperature in the simulation. Default is True.
        simtype (str): The type of simulation to run. Default is 'fluid'.
        grid_size (tuple): The size of the simulation grid. Default is (100, 100, 0).
        Attributes:
        config (dict): A dictionary containing the configuration parameters.
        """
        
        self.config = {
            'velocities_set': velocities_set,
            'simtype': simtype,
            'use_temperature': True,
            'grid_size': grid_size,
            'viscosity': viscosity,
        }
        self.checkConfig()

    def get(self):
        """
        Retrieve the current configuration.
        Returns:
            dict: The current configuration settings.
        """
        return self.config

    def checkConfig(self):       
        """
        Validates the configuration dictionary stored in `self.config`.
        Raises:
            ValueError: If any required key is missing from the configuration.
            ValueError: If the value for 'velocities_set' is not in `available_velocities_sets`.
            ValueError: If the value for 'simtype' is not in `avilable_simtypes`.
            ValueError: If the value for 'use_temperature' is not a boolean.
            ValueError: If the value for 'grid_size' is not a tuple.
            ValueError: If the 'grid_size' tuple does not have exactly 3 elements.
            ValueError: If the value for 'viscosity' is not a float.
            ValueError: If the value for 'viscosity' is not positive.
        The required keys in the configuration are:
            - 'velocities_set': Must be present in `available_velocities_sets`.
            - 'simtype': Must be present in `avilable_simtypes`.
            - 'use_temperature': Must be a boolean.
            - 'grid_size': Must be a tuple with exactly 3 elements.
            - 'viscosity': Must be a positive float.
        """
        
        global available_velocities_sets, avilable_simtypes

        # Check if all required keys are present
        required_keys = ['velocities_set', 'simtype', 'use_temperature', 'grid_size', 'viscosity']  # Add your required keys here
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config argument: {key}")
        
        # Check if the values are valid
        if self.config['velocities_set'] not in available_velocities_sets:
            raise ValueError("Invalid velocities set")
        
        # Check if the simulation type is valid
        if self.config['simtype'] not in avilable_simtypes:       
            raise ValueError("Invalid simulation type")
        
        # Check if the use_temperature value is a boolean
        if not isinstance(self.config['use_temperature'], bool):
            raise ValueError("Invalid use_temperature value")
        
        # Check if the grid size is a tuple
        if not isinstance(self.config['grid_size'], tuple):
            raise ValueError("Invalid grid size value")
        
        # Check if the grid size tuple has 3 elements
        if len(self.config['grid_size']) != 3:
            raise ValueError("Invalid grid size value")
        
        # Check if the viscosity value is a float
        if not isinstance(self.config['viscosity'], float):
            raise ValueError("Invalid viscosity value")
        
        # Check if the viscosity value is positive
        if self.config['viscosity'] < 0:
            raise ValueError("Viscosity value must be positive")
        
        