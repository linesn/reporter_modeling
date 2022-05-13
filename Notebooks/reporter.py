"""
Functions and objects to aid in Reporter modeling.

We will use the Google standard for docstring formatting.
"""

import scipy.stats as st
from tqdm.notebook import tqdm
import numpy as np


def bin_not(n):
    '''Just switch 1 and 0.'''
    return 1 - 1*n


def proportionate_threshold_agreement(x,y,t=0.5):
    """Determine how similar two n-long arrays are in a threshold sense.
    
    This function first labels which entries of x are less than t and which 
    entries of y are less than t, then sums up how many disagreements there
    are between these two label vectors, and divides this by the length n.
    That gives us the proportionate disagreement; we subtract from 1 to get 
    the proportionate agreement.
    """
    return (1-((x<t)^(y<t)).sum()/x.shape[0])


def generate_data(n_i, n_t, r_p=0.5):
    """Generate data for one topic using Reporter Model A.
    
    Args:
        n_i (int): The number of observers.
        n_t (int): The number of time steps.
        r_p (float): Noise parameter.
        
    Returns:
        o (array): The observed deterministic observations.
        delta (float): The (hidden) true delta (topic bias to 1).
        f (array): The (hidden) true f values (binary facts).
        beta (array): The (hidden) observer awareness biases.
        alpha (array): The (hidden) observer awareness values.
        epsilon (array): The (hidden) observer communication biases.        
        tau (array): The (hidden) observer communication choice values.
        r (array): The (hidden) random noise.
    """
    delta = st.uniform.rvs(0,1)
    epsilon = st.uniform.rvs(0,1,n_i)
    beta = st.uniform.rvs(0,1,n_i)
    f = st.bernoulli(delta).rvs(n_t)
    tau = np.array([st.bernoulli(epsilon_ij).rvs(n_t) for epsilon_ij in epsilon])
    alpha = np.array([st.bernoulli(beta_ij).rvs(n_t) for beta_ij in beta])
    r = np.array([st.bernoulli(r_p).rvs(n_t) for i in range(n_i)])
    o = np.array([[tau[i][t] * (alpha[i][t] * f[t] + bin_not(alpha[i][t])*r[i][t]) + 2 * \
                bin_not(tau[i][t]) for i in range(n_i)] for t in range(n_t)])
    return((o, delta, f, beta, alpha, epsilon,tau,r))


def evaluate_estimators_full(data, agreement_thresh=0.5, printing=True):
    """Evaluate Reporter Model A estimators
    
    Args:
        data (tuple): The output from generate_data, a tuple of
            (o, delta, f, beta, alpha, epsilon,tau,r).
        agreement_thresh (float): The agreement threshold for 
            proportionate threshold agreement.
        printing (bool): Indicates whether to print info.
    
    Returns:
    """
    (o, delta, f, beta, alpha, epsilon,tau,r) = data
    n_t, n_i = o.shape
    # MLE for epsilons 
    epsilon_hat = [1-np.argwhere(o[:,i]==2).shape[0]/(n_t) for i in range(n_i)]
    epsilon_norm = np.linalg.norm(epsilon_hat - epsilon)
    # We need to drop any timestep where we got absolutely no information.
    drop_rows = np.argwhere(np.all(o == 2, axis=1)).flatten()
    o = np.delete(o,drop_rows,0)
    n_t, n_i = o.shape
    # Concensus estimator for f
    b = np.array([st.mode([oit for oit in o[t] if oit!=2])[0][0] for t in range(n_t)])
    f = np.delete(f,drop_rows)
    # Judge the quality of the f estimate
    concensus_f_accuracies = 1-(abs(b-f).sum()/b.shape[0])
    # Estimate the probability of observing 1 empirically
    p_o_i_bar_1 = [(o[:,i]==1).sum()/((o[:,i]!=2).sum()) for i in range(n_i)] 
    # Estimate delta
    delta_hat = np.mean(b)
    # Estimate beta
    beta_hat = np.array([(2 * p_i - 1) / (2*delta_hat - 1) for p_i in p_o_i_bar_1])
    # Judge the quality of the beta estimate
    beta_norm = np.linalg.norm(beta_hat - beta)
    beta_agreements = proportionate_threshold_agreement(beta_hat, beta, agreement_thresh)
    if printing:
        print(f"Estimators are delta_hat={delta_hat}, b, beta_hat.")
        print(f"The norm distance between epsilon_hat and epsilon is {epsilon_norm}.")
        print(f"The approximation b for f is {100*concensus_f_accuracies}% accurate.")
        print(f"The approximation delta_hat for delta={delta} has relative error {(delta_hat-delta)/delta}.")
        print(f"The norm distance between beta_hat and beta is {beta_norm}.")
        print(f"The beta_hat estimator recognizes betas less than {agreement_thresh}")
        print(f"{beta_agreements*100}% of the time.")
    return(b, delta_hat, beta_hat)