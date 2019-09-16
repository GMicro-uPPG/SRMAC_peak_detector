#!python3
# Author: Victor O. Costa 
# Performs random search on the crossover's alphas using rmse of parameters by minute as error function 

from ppg_peak_detection import crossover_detector
from read_ppg_mimic import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz


# TODO: load subjects ppg and reference PP intervals

# To print one record name
print(records[0].name) # <- String
# To print that record ppg
print(records[0].ppg) # <- list: [x_ppg, ppg]
# To print that record hrv
print(records[0].hrv) # <- list: [x_hrv, hrv]


'''
peak_detector = ppg_peak_detector()
# TODO
def features_extractor(heart_rate_signal):
    # calculate each parameter (feature)
    return feature1, feature2,..
    
# random search of estimated hrv parameters (features) by minute on each subject using given hrv references
solution_archive = []
for iteration in range(num_iterations):
    # Randomize alphas, with fast alpha depending on slow alpha, thus guaranteeing fast alpha > slow alpha
    alpha_slow = np.random.uniform(0, 1)
    alpha_fast = np.random.uniform(alpha_slow, 1)   
    peak_detector.set_parameters(alpha_fast, alpha_slow)
    error = peak_detector.features_rmse(ppg_signal= ,hr_references= , fs, features_extractor)
    # Keep solutions in a matrix
    solution_archive.append([alpha_fast, alpha_slow, error])

# sort solutions according to the errors
solution_archive = solution_archive[solution_archive[:,-1].argsort()]
best_solution = solution_archive[0]
'''