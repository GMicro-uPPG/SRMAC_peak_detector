#!python3
# Author: Victor O. Costa 
# Performs random search on the crossover's alphas using regularized confusion matrix-based cost function 

import numpy as np
import pickle as pkl
from ppg_peak_detection import crossover_detector
from read_ppg_mimic import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz
from log_manager import time, log
#from plot import *

# --- Handle error
def err_finish(e):
    log.printWriteLog(e)

    timestamp_ = '\nLast timestamp: ' + str(time.getTimestamp())
    log.printWriteLog(timestamp_)

    log.closeLogFile() # close log file
#end-def


try:
    # Record example
    # name = records[0].name              # Record name: string
    # ppg = records[0].ppg                # Record ppg: [x_ppg, ppg]
    # hrv = records[0].hrv                # Record hrv: [x_hrv, hrv]
    # plotPPG(name, ppg, hrv)             # Plot ppg signal and peak points

    # Use 30 records to train model
    train_records = records[0:30]
    log.printWriteLog('\nrecords[0:30]')

    # Random search of alphas, using regularized confusion matrix-based cost
    peak_detector = crossover_detector()
    # Parameters
    C = 3                                          # Regularization hyperparameter
    log.printWriteLog('\nC = ', C)

    num_iterations = 5000 # 10000                         # Number of random search iterations
    log.printWriteLog('\nnum_iterations = ', num_iterations)

    # Optimization
    #solution_archive = np.zeros((num_iterations,3))
    best_solution = []
    for iteration in range(num_iterations):
        s_iter = '\n[Search iteration ' + str(iteration) + ']'
        log.printWriteLog(s_iter)
        # Randomize alphas, with fast alpha depending on slow alpha, thus guaranteeing fast alpha > slow alpha
        alpha_fast = np.random.uniform(0, 1)
        alpha_slow = np.random.uniform(alpha_fast, 1)   
        peak_detector.set_parameters(alpha_fast, alpha_slow)
        cost = peak_detector.total_regularized_cost(train_records, C)
        # Keep solutions in a matrix
        if iteration == 0:
            best_solution = [alpha_fast, alpha_slow, cost]
        elif cost < best_solution[-1]:
            best_solution = [alpha_fast, alpha_slow, cost]

        log.printWriteLog('\nbest_solution = alpha_fast, alpha_slow, cost') 
        log.printWriteLog('best_solution = ' + str(best_solution)) 
        #solution_archive[iteration, :] = [alpha_fast, alpha_slow, cost]

    # Sort solutions according to the costs
    #solution_archive = solution_archive[solution_archive[:,-1].argsort()]
    #best_solution = solution_archive[0]
    #log.printWriteLog(solution_archive)
    #pkl.dump(solution_archive, open("solution_archive.data","wb"))


    timestamp_ = '\nLast timestamp: ' + str(time.getTimestamp())
    log.printWriteLog(timestamp_)

    log.closeLogFile() # close log file


# --- Exceptions
except IOError:
    e = '\nError: An error occurred trying to read the file.\n'
    err_finish(e)

except ValueError:
    e = '\nError: Non-numeric data found in the file.\n'
    err_finish(e)

except ImportError:
    e = '\nError: No module found.\n'
    err_finish(e)

except EOFError:
    e = '\nError: Why did you do an EOF on me?\n'
    err_finish(e)

except KeyboardInterrupt:
    e = '\nError: You cancelled the operation.\n'
    err_finish(e)

except:
    e = 'An error occurred.'
    err_finish(e)