import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("netflix_incomplete.txt")


"""
Ks = [1,2,3,4]
seeds = [0,1,2,3,4]

for k in Ks:
    for seed in seeds:
        mixture, post, costs = common.init(X, k, seed)
        title = "K = {0}, Seed = {1}, Cost = {2}".format(k, seed, costs)
        print(title)
        common.plot(X, mixture, post, title)
"""        



# Example of how to run the EM algorithm with different seeds
seeds = [0, 1, 2, 3, 4]
K_values = [1, 12]
best_bic_score = -9000000000
best_K_value = None

for K in K_values:
    for seed in seeds:
        print(f"Running EM algorithm with K = {K}, seed={seed}")
    
        # Initialize the mixture model
        mixture, post = naive_em.initialize_mixture(X, K, seed=seed)
    
        # Run the EM algorithm
        mixture, post, log_likelihood = naive_em.run(X, mixture, post)
        bic_score = common.bic(X, mixture, log_likelihood)
        
        if bic_score is None or bic_score > best_bic_score:
            best_bic_score = bic_score
            best_K_value = K
    
        # Output results
        print(f"Final log-likelihood: {log_likelihood} \n bic-score = {bic_score}")
        #print(f"Means:\n{mixture.mu}\n")
        #print(f"Variances:\n{mixture.var}\n")
        #print(f"Mixing Proportions:\n{mixture.p}\n")
        #print("=" * 50)
        
print(f"\n Best BIC Score = {best_bic_score} \n Best K = {best_K_value}")