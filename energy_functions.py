import numpy as np

# total process energy
def process_energy_lin(history):
    """ below baseline = less energy used """
    return np.sum(history)

def process_energy_abs(history):
    """ below baseline = more energy used """
    return np.sum(np.abs(history))

def process_energy_relu(history):
    """ below baseline = same as baseline """
    return np.sum(history[history > 0])

# state energy
def state_energy_bio_lin(X):
    """ below baseline = less energy used """
    return np.sum(X)

def state_energy_bio_abs(X):
    """ below baseline = more energy used """
    return np.sum(np.abs(X))
    
def state_energy_bio_relu(X):
    """ below baseline = same as baseline """
    return np.sum(X[X > 0])

process_energy_dictionary = {
    "pe_lin":process_energy_lin,
    "pe_abs":process_energy_abs,
    "pe_relu":process_energy_relu,
}

state_energy_dictionary =  {
    "se_lin":state_energy_bio_lin,
    "se_abs":state_energy_bio_abs,
    "se_relu":state_energy_bio_relu,
}

# hopfield network energy
def states_energy_hop(history):
    pass


# # Energy
# def process_energy_sync(self):
#     """ This assumes that deviating from baseline (0) uses more energy """
#     return np.sum(self.inference_history)

# def process_energy_sync_abs(self):
#     """ This assumes that deviating from baseline (0) uses more energy """
#     return np.sum(np.abs(self.inference_history))

# def energy_async(self):
#     """ Energy is calculated as the sum of all states """
#     return np.sum([self.inference_history >= 0])

# def energy_async_abs(self):
#     """sum of values >= 0 over the inference history"""
#     return np.sum([self.inference_history >= 0])
