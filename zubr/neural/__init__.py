import os


def start_dynet(config,params):
    """General function for setting up dynet memory and random seed 

    :param config: the global configuration with settings 
    :param params: the dynet params instance 
    :type params: cynet._dynet.DynetParams
    """
    params.set_mem(config.mem)
    params.set_random_seed(config.seed)
    params.init()
