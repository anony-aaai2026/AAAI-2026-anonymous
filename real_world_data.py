# Code for AAAI 2026: real world data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import pinv
from numpy.random import default_rng


np.random.seed(42)


# Load real dataset
protein_df = pd.read_csv("dataset/CASP.csv")
X = protein_df.drop(columns=["RMSD"]).values  # Features: F1-F9
y = protein_df["RMSD"].values  # Target: RMSD


# Initial pool setup
n_labeled = 20000  # Initial labeled size
X_initial, X_pool, y_initial, y_pool = train_test_split(X, y, train_size=n_labeled, random_state=42)
D_L_x = X_initial
D_L_y = y_initial.reshape(-1, 1)
D_U = X_pool
D_ML = y_pool

# Simulated model parameters for online update
lr = LinearRegression(fit_intercept=True)
lr.fit(D_L_x, D_L_y.ravel())

beta_prev = np.concatenate(([lr.intercept_], lr.coef_)).reshape(-1, 1)
lambda_factor = 0.998
label_budget =10000
alpha = 0.1
phi = 10



log_eta_mu, log_eta_var = -1,0.25
eta_t = np.random.lognormal(mean=log_eta_mu,sigma=log_eta_var, size=D_U.shape[0])

labeled_cost = 0
label_count = 0
total_utility = 0

D_L_aug = np.hstack((np.ones((D_L_x.shape[0], 1)), D_L_x))
H_inv = np.linalg.pinv(D_L_aug.T @ D_L_aug)  # Use pseudo-inverse for stability

# Compute UPVs: xᵀ H⁻¹ x
upvs = np.array([x @ H_inv @ x.T for x in D_L_aug])  
# Estimate mean and variance
mu_ovbal = np.mean(upvs)
var_ovbal = np.var(upvs)


def update_tau_online(mu_ovbal, var_ovbal, score_t, lambda_factor, alpha):
    mu = lambda_factor * mu_ovbal + (1 - lambda_factor) * score_t
    var = lambda_factor * var_ovbal + (1 - lambda_factor) * (score_t - mu) ** 2
    std = np.sqrt(var)
    z = norm.ppf(1 - alpha)
    tau = mu + z * std
    return tau, mu, var



def online_active_learning(D_L_x, D_L_y, D_U, D_ML, lambda_factor, mu_ovbal, var_ovbal, beta_prev, 
                           label_budget, alpha, cost_aware, model,
                           phi, eta_t, scheme):
    beta_t_history, H_t_inverse_history, smooth_mse_history, cost_history, forecast_error_history = [], [], [], [],[]
    tau_ovbal_history = []
    mu_ovbal_history=[]
    var_ovbal_history=[]
    score_history=[]
    estimated_utility_list = []
    true_utility_list = []
    mse_total=0
    avg_mse_history=[]
    l_j_history=[]
    sigma_y_sq_history=[]

# OVBAL Step 0: Initialization
# -----------------------------------
# Initialize H_0, beta_0, tau_0, sigma_y^2, c, etc.
# Compute initial UPV statistics from D_L    
    D_L_aug = np.hstack((np.ones((D_L_x.shape[0], 1)), D_L_x))
    H_prev = D_L_aug.T @ D_L_aug
    H_prev_inv = np.linalg.pinv(H_prev + 1e-6 * np.eye(H_prev.shape[0]))
    H_t_inverse_history=[np.trace(H_prev_inv)]
    sigma_y_sq_prev = np.var(D_L_y)
    sigma_y_sq_history=[sigma_y_sq_prev]
    var_beta_prev=sigma_y_sq_prev*np.trace(H_prev_inv)
    var_beta_history=[var_beta_prev]
    upvs = np.array([x @ H_inv @ x.T for x in D_L_aug])
    mu_ovbal = np.mean(upvs)
    var_ovbal = np.var(upvs)
    std = np.sqrt(var_ovbal)
    z = norm.ppf(1 - alpha)
    tau_ovbal = mu_ovbal + z * std
    labeled_cost = 0
    label_count = 0  
    total_utility=0
 


# === Online Active Learning Loop ===
# Step 1: Online Loop
    for t in range(len(D_U)):
        x_t = D_U[t].reshape(-1, 1)
        x_t_aug = np.vstack(([[1]], x_t))
        y_true_t = D_ML[t]

        possibility = 0.1

        # === Step 2: Determine whether to query ===
        if model == "ovbal":
            uncertainty_score = float(x_t_aug.T @ H_prev_inv @ x_t_aug)
            score_history.append(uncertainty_score)
            query = uncertainty_score >= tau_ovbal
            H_virtual = lambda_factor * H_prev + x_t_aug @ x_t_aug.T
            H_virtual_inv = np.linalg.pinv(H_virtual)
            delta_H_inv = H_prev_inv - H_virtual_inv
            l_j = sigma_y_sq_prev * np.trace(delta_H_inv)
            query_lj = l_j >= eta_t[t] / phi
            if cost_aware:
                query = query and query_lj    
            l_j_history.append(l_j)

       

        elif model == "Bia":
            query = True
            H_virtual = lambda_factor * H_prev + x_t_aug @ x_t_aug.T
            H_virtual_inv = np.linalg.pinv(H_virtual)
            delta_H_inv = H_prev_inv - H_virtual_inv
            l_j = sigma_y_sq_prev * np.trace(delta_H_inv)

        elif model == "RS":
            query = (np.random.rand() < possibility)
            H_virtual = lambda_factor * H_prev + x_t_aug @ x_t_aug.T
            H_virtual_inv = np.linalg.pinv(H_virtual)
            delta_H_inv = H_prev_inv - H_virtual_inv
            l_j = sigma_y_sq_prev * np.trace(delta_H_inv)

     

        # === Step 2.5: Estimate price BEFORE purchasing ===
        if query:
            if model in ["ovbal", "Bia", "RS"]:
                l_hat = max(0, l_j)
                estimated_utility_list.append(l_hat)

            if scheme == "BC":
                p_j = phi * l_hat
            elif scheme == "SC":
                p_j = eta_t[t]
                
            else:
                raise ValueError(f"Unknown scheme: {scheme}")

        # === Step 3a: Perform model update if queried ===
        if query:
            y_forecast_t = x_t_aug.T @ beta_prev
            epsilon_t = y_true_t - y_forecast_t
            forecast_error = (epsilon_t ** 2).item()
            forecast_error_history.append(forecast_error)
            mse_total+=forecast_error
            avg_mse_t = mse_total / (t + 1)
            avg_mse_history.append(avg_mse_t)
            sigma_y_sq = lambda_factor * sigma_y_sq_prev + (1 - lambda_factor) * forecast_error
            
            H_t = lambda_factor * H_prev + x_t_aug @ x_t_aug.T
            H_t_inv = np.linalg.pinv(H_t)
            beta_t = beta_prev + (H_t_inv @ x_t_aug) * epsilon_t

            if model == "ovbal":
                uncertainty_score = float(x_t_aug.T @ H_t_inv @ x_t_aug)
                tau_ovbal, mu_ovbal, var_ovbal = update_tau_online(
                    mu_ovbal, var_ovbal, uncertainty_score, lambda_factor, alpha)
                mu_ovbal_history.append(mu_ovbal)
                var_ovbal_history.append(var_ovbal)
                tau_ovbal_history.append(tau_ovbal)
            
            var_beta = sigma_y_sq * np.trace(H_t_inv)
            true_l_j = var_beta_prev-var_beta
            true_utility_list.append(true_l_j)
            total_utility += true_l_j


            # Update cost and labels
            labeled_cost += p_j
            label_count += 1
            D_L_x = np.vstack((D_L_x, x_t.T))
            D_L_y = np.vstack((D_L_y, y_true_t.reshape(1, 1)))
            sigma_y_sq_prev = sigma_y_sq
            beta_prev = beta_t
            H_prev = H_t
            H_prev_inv = H_t_inv
            var_beta_prev=var_beta
            var_beta_history.append(var_beta)
            beta_t_history.append(beta_t)
            H_t_inverse_history.append(np.trace(H_t_inv))
            sigma_y_sq_history.append(sigma_y_sq)
            cost_history.append(labeled_cost)


        else:
            # Step 4a: no purchase - model decay
            # K_t = H_t
            H_t = lambda_factor * H_prev
            # H_prev = K_t
            H_t_inv = np.linalg.pinv(H_t)
            beta_t = beta_prev
            beta_prev = beta_t
            H_prev = H_t
            H_prev_inv = H_t_inv
        
            beta_t_history.append(beta_t)
            H_t_inverse_history.append(np.trace(H_t_inv))

            # === Forecast for evaluation (even without label)
            y_forecast_t = x_t_aug.T @ beta_t
            epsilon_t = y_true_t - y_forecast_t
            forecast_error = (epsilon_t ** 2).item()
            forecast_error_history.append(forecast_error)
            mse_total+=forecast_error
            avg_mse_t = mse_total / (t + 1)
            avg_mse_history.append(avg_mse_t)

            sigma_y_sq = lambda_factor * sigma_y_sq_prev +  (1 - lambda_factor) * forecast_error
            sigma_y_sq_history.append(sigma_y_sq)
            var_beta=sigma_y_sq*np.trace(H_t_inv)
            var_beta_history.append(var_beta)
            var_beta_prev=var_beta
            sigma_y_sq_prev=sigma_y_sq





        
            if model=="ovbal":
                # Step4b: update threshold still using score
                uncertainty_score = float(x_t_aug.T @ H_t_inv @ x_t_aug)
                tau_ovbal, mu_ovbal, var_ovbal = update_tau_online(
                mu_ovbal, var_ovbal, uncertainty_score, lambda_factor, alpha)
                mu_ovbal_history.append(mu_ovbal)
                var_ovbal_history.append(var_ovbal)
                tau_ovbal_history.append(tau_ovbal)

            l_t = 0
            total_utility += l_t
            cost_history.append(labeled_cost)

        # Step 5: budget check
        if labeled_cost >= label_budget:
            print("Budget exhausted.")
            break


    if model == "ovbal":
        return beta_t_history, H_t_inverse_history, smooth_mse_history, cost_history,  avg_mse_history, tau_ovbal_history, label_count, total_utility, estimated_utility_list, true_utility_list, mu_ovbal_history, var_ovbal_history, score_history, l_j_history, var_beta_history, sigma_y_sq_history
    else:
        return beta_t_history, H_t_inverse_history, smooth_mse_history, cost_history,  avg_mse_history, None, label_count, total_utility, estimated_utility_list, true_utility_list, None, None, None, None, var_beta_history, sigma_y_sq_history
    



# Run both ovbal and oqbcal
results_ovbal_sc = online_active_learning(D_L_x, D_L_y, D_U, D_ML,lambda_factor=lambda_factor, mu_ovbal=mu_ovbal, var_ovbal=var_ovbal, beta_prev=beta_prev, 
                           label_budget=label_budget, alpha=alpha, cost_aware=True,  model= "ovbal",
                           phi=phi, eta_t= eta_t,  scheme="SC")
results_bia_sc = online_active_learning(D_L_x, D_L_y, D_U, D_ML,lambda_factor=lambda_factor, mu_ovbal=mu_ovbal, var_ovbal=var_ovbal,beta_prev=beta_prev, 
                           label_budget=label_budget, alpha=alpha, cost_aware=True,  model= "Bia",
                           phi=phi, eta_t= eta_t,   scheme="SC")
results_rs_sc = online_active_learning(D_L_x, D_L_y, D_U, D_ML,lambda_factor=lambda_factor,mu_ovbal=mu_ovbal, var_ovbal=var_ovbal, beta_prev=beta_prev,
                           label_budget=label_budget, alpha=alpha, cost_aware=True, model= "RS",
                           phi=phi, eta_t= eta_t,   scheme="SC")

# Convert histories
beta_ovbal = np.array(results_ovbal_sc[0]).squeeze()
beta_bia = np.array(results_bia_sc[0]).squeeze()
beta_rs = np.array(results_rs_sc[0]).squeeze()
h_inv_ovbal=results_ovbal_sc[1]
h_inv_bia=results_bia_sc[1]
h_inv_rs=results_rs_sc[1]
smooth_mse_ovbal = results_ovbal_sc[2]
smooth_mse_bia = results_bia_sc[2]
smooth_mse_rs = results_rs_sc[2]
cost_ovbal = results_ovbal_sc[3]
cost_bia= results_bia_sc[3]
cost_rs = results_rs_sc[3]
avg_mse_ovbal = results_ovbal_sc[4]
avg_mse_bia = results_bia_sc[4]
avg_mse_rs = results_rs_sc[4]
tau_ovbal_history = results_ovbal_sc[5]
label_count_ovbal = results_ovbal_sc[6]
label_count_bia = results_bia_sc[6]
label_count_rs = results_rs_sc[6]
utility_ovbal_sc = results_ovbal_sc[7]
utility_bia_sc = results_bia_sc[7]
utility_rs_sc= results_rs_sc[7]
est_util_ovbal = results_ovbal_sc[8]
true_util_ovbal = results_ovbal_sc[9]
est_util_bia = results_bia_sc[8]
true_util_bia = results_bia_sc[9]
est_util_rs = results_rs_sc[8]
true_util_rs = results_rs_sc[9]

var_beta_ovbal=results_ovbal_sc[14]
var_beta_bia=results_bia_sc[14]
var_beta_rs=results_rs_sc[14]
sigma_y_ovbal=results_ovbal_sc[15]
sigma_y_bia=results_bia_sc[15]
sigma_y_rs=results_rs_sc[15]


results_ovbal_bc = online_active_learning(D_L_x, D_L_y, D_U, D_ML,lambda_factor=lambda_factor, mu_ovbal=mu_ovbal, var_ovbal=var_ovbal, beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha, cost_aware=True,  model= "ovbal",
                           phi=phi, eta_t= eta_t,  scheme="BC")
results_bia_bc = online_active_learning(D_L_x, D_L_y, D_U, D_ML,lambda_factor=lambda_factor, mu_ovbal=mu_ovbal, var_ovbal=var_ovbal, beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha, cost_aware=True,  model= "Bia",
                           phi=phi, eta_t= eta_t, scheme="BC")
results_rs_bc = online_active_learning(D_L_x, D_L_y, D_U, D_ML,lambda_factor=lambda_factor, mu_ovbal=mu_ovbal, var_ovbal=var_ovbal, beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha, cost_aware=True, model= "RS",
                           phi=2, eta_t= eta_t,  scheme="BC")

cost_ovbal_bc = results_ovbal_bc[3]
cost_bia_bc= results_bia_bc[3]
cost_rs_bc = results_rs_bc[3]

utility_ovbal_bc = results_ovbal_bc[7]
utility_bia_bc = results_bia_bc[7]
utility_rs_bc = results_rs_bc[7]

results_ovbal_sc_no = online_active_learning(D_L_x, D_L_y, D_U,D_ML, lambda_factor=lambda_factor,mu_ovbal=mu_ovbal, var_ovbal=var_ovbal, beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha,  cost_aware=False, model= "ovbal",
                           phi=phi, eta_t= eta_t, scheme="SC")
results_ovbal_bc_no = online_active_learning(D_L_x, D_L_y, D_U,D_ML,lambda_factor=lambda_factor, mu_ovbal=mu_ovbal, var_ovbal=var_ovbal,beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha, cost_aware=False,  model= "ovbal",
                           phi=phi, eta_t= eta_t, scheme="BC")

label_count_ovbal_no=results_ovbal_sc_no[6]
cost_ovbal_no=results_ovbal_sc_no[3]
utility_ovbal_sc_no=results_ovbal_sc_no[7]
tau_ovbal_history_no=results_ovbal_sc_no[5]


avg_mse_ovbal_bc_no=results_ovbal_bc_no[4]
label_count_ovbal_bc_no=results_ovbal_bc_no[6]
cost_ovbal_bc_no=results_ovbal_bc_no[3]
utility_ovbal_bc_no=results_ovbal_bc_no[7]

plt.rcParams.update({
    "font.family": "serif",         
    "font.size": 20,
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 3,
    "figure.dpi": 300,
    "savefig.format": "pdf"
})


n_runs = 100
all_traces = {"ovbal": [], "bia": [], "rs": []}

for i in range(n_runs):
    X_initial, X_pool, y_initial, y_pool = train_test_split(X, y, train_size=n_labeled, random_state=i)

    for model in ["ovbal", "Bia", "RS"]:
        results = online_active_learning(
            X_initial, y_initial.reshape(-1, 1), X_pool, y_pool,
            lambda_factor=lambda_factor,
            mu_ovbal=mu_ovbal, var_ovbal=var_ovbal,
            beta_prev=beta_prev.copy(),
            label_budget=label_budget,
            alpha=alpha, cost_aware=True, model=model,
            phi=phi, eta_t=eta_t, scheme="SC")

        var_beta_trace = results[14]
        all_traces[model.lower()].append(var_beta_trace)

# Align by truncating to common minimum length
min_len = min(min(len(tr) for tr in group) for group in all_traces.values())
def truncate_traces(traces):
    return np.array([tr[:min_len] for tr in traces])

var_beta_ovbal = truncate_traces(all_traces["ovbal"])
var_beta_bia = truncate_traces(all_traces["bia"])
var_beta_rs = truncate_traces(all_traces["rs"])

x = np.arange(min_len)
plt.figure(figsize=(10, 5))

# Helper to plot shaded mean ± quantile
def plot_mean_quantile(data, label, color, linestyle):
    mean = np.mean(data, axis=0)
    q25 = np.quantile(data, 0.25, axis=0)
    q75 = np.quantile(data, 0.75, axis=0)
    plt.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=3)
    plt.fill_between(x, q25, q75, color=color, alpha=0.2)

plot_mean_quantile(var_beta_ovbal, "OVBAL", "blue", "-")
plot_mean_quantile(var_beta_bia, "BIA", "orange", "--")
plot_mean_quantile(var_beta_rs, "RS", "green", ":")

plt.xlabel("Time step")
plt.ylabel("Trace of Var($\\beta$)")
plt.legend(loc="upper right", fontsize=16)
plt.grid(False)
plt.tight_layout()
plt.savefig("trace_var_beta_protein.pdf")
plt.show()




fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)

# SC Pricing
axs[0].plot(cost_ovbal, label="OVBAL", linestyle="-")
axs[0].plot(cost_bia, label="BIA", linestyle="-.")
axs[0].plot(cost_rs, label="RS", linestyle="--")
axs[0].axhline(label_budget, color='red', linestyle='--', linewidth=3)
axs[0].text(len(cost_ovbal)*0.6, label_budget * 0.9, 'Budget Limit', color='red', fontsize=18)
axs[0].set_xlabel("Time step",fontsize=25)
axs[0].set_ylabel("Cumulative cost",fontsize=25)
axs[0].tick_params(axis='x',labelsize=20)
axs[0].tick_params(axis='y',labelsize=20)
axs[0].legend()

# BC Pricing
axs[1].plot(cost_ovbal_bc, label="OVBAL", linestyle="-")
axs[1].plot(cost_bia_bc, label="BIA", linestyle="-.")
axs[1].plot(cost_rs_bc, label="RS", linestyle="--")
axs[1].axhline(label_budget, color='red', linestyle='--', linewidth=3)
axs[1].text(len(cost_ovbal_bc)*0.6, label_budget * 0.9, 'Budget Limit', color='red',fontsize=18)
axs[1].set_xlabel("Time step", fontsize=25)
axs[1].tick_params(axis='x',labelsize=20)
axs[1].tick_params(axis='y',labelsize=20)
axs[1].legend()
plt.tight_layout()
plt.savefig("cumulative_cost_with_budget_protein.pdf")
plt.show()


# # --- Follwed by further analysis which is not included in the main paper, to see those results by uncommenting the code ---
 
# plt.figure(figsize=(10, 4))
# plt.plot(h_inv_ovbal, label="OVBAL", linewidth=2)
# plt.plot(h_inv_bia, label="BIA", linestyle="--", linewidth=2)
# plt.plot(h_inv_rs, label="RS", linestyle=":", linewidth=2)
# plt.xlabel("Time step")
# plt.ylabel("$H_{t}^{-1}$")
# plt.legend()
# plt.grid(False)
# plt.tight_layout()
# plt.savefig("ht-1_all_models_protein.pdf")
# plt.show()


# plt.figure(figsize=(10, 4))
# plt.plot(sigma_y_ovbal, label="OVBAL", linewidth=2)
# plt.plot(sigma_y_bia, label="BIA", linestyle="--", linewidth=2)
# plt.plot(sigma_y_rs,label="RS", linestyle=":", linewidth=2)
# plt.xlabel("Time step")
# plt.ylabel("$σ_{y}^{2}$")
# plt.legend()
# plt.grid(False)
# plt.tight_layout()
# plt.savefig("sigma_y_all_models.pdf")
# plt.show()


# plt.figure(figsize=(10, 4))
# plt.plot(var_beta_ovbal, label="OVBAL", linewidth=2)
# plt.plot(var_beta_bia, label="BIA", linestyle="--", linewidth=2)
# plt.plot(var_beta_rs, label="RS", linestyle=":", linewidth=2)
# plt.xlabel("Time step")
# plt.ylabel("Trace of Var($\\beta$)")
# plt.legend()
# plt.grid(False)
# plt.tight_layout()
# plt.savefig("trace_var_beta_all_models.pdf")
# plt.show()




# fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
# methods = ["ovbal",  "Bia", "RS"]
# true_utils = [true_util_ovbal,  true_util_bia, true_util_rs]
# est_utils = [est_util_ovbal,  est_util_bia, est_util_rs]
# for i, ax in enumerate(axs.flatten()):
#     ax.plot(true_utils[i], label="True Utility", linestyle='-')
#     ax.plot(est_utils[i], label="Estimated Utility", linestyle='--')
#     ax.set_xlabel("Query Step")
#     ax.set_ylabel("Utility")
#     ax.legend()
#     ax.grid(False)
# plt.tight_layout()
# plt.savefig("utility_over_time_subplots_protein.pdf")
# plt.show()



# fig, axs = plt.subplots(1, 3, figsize=(20, 4.5), sharey=True)
# methods = ["ovbal", "Bia", "RS"]
# betas = [beta_ovbal,beta_bia, beta_rs]
# linestyles = ["--", ":"]
# n_features = beta_ovbal.shape[1]  # Fix here
# for i, ax in enumerate(axs):
#     for j in range(n_features):
#         ax.plot(betas[i][:, j], linestyle=linestyles[1], label=f"Estimated $β_{j}$" if i == 0 else "")
#     ax.set_xlabel("Time step")
#     if i == 0:
#         ax.set_ylabel("β value")
#     ax.grid(False)

# # Add legend only once
# axs[0].legend(loc="upper left", ncol=1)
# plt.tight_layout()
# plt.savefig("beta_evolution_all_methods_protein.pdf")
# plt.show()


# mu_ovbal_history =results_ovbal_sc[10]
# var_ovbal_history=results_ovbal_sc[11]
# score_history=results_ovbal_sc[12]

# # --- Plot: Uncertainty score vs tau ---
# plt.figure(figsize=(7, 4))
# plt.plot(score_history, label=r"$x^\top H^{-1} x$", linewidth=1.6)
# plt.plot(tau_ovbal_history, label=r"$\tau$", linestyle="--", linewidth=1.6)
# plt.xlabel("Time step")
# plt.ylabel("Uncertainty score")
# plt.legend()
# plt.tight_layout()
# plt.savefig("uncertainty_vs_tau_protein.pdf")
# plt.show()

# # --- Plot: Mean and Variance of UPV over time ---
# fig, ax1 = plt.subplots(figsize=(7, 4))
# ax1.plot(mu_ovbal_history, label="OVBAL", color="blue", linewidth=1.6)
# ax1.set_ylabel("μ", color="blue")
# ax1.tick_params(axis='y', labelcolor="blue")

# ax2 = ax1.twinx()
# ax2.plot(var_ovbal_history, label="OVBAL", color="red", linestyle="--", linewidth=1.6)
# ax2.set_ylabel("σ² ", color="red")
# ax2.tick_params(axis='y', labelcolor="red")
# fig.tight_layout()
# plt.savefig("ovbal_mu_var_evolution_protein.pdf")
# plt.show()

# l_j_history=results_ovbal_sc[13]
# plt.figure(figsize=(10, 4))
# plt.plot(l_j_history, label="OVBAL")
# plt.axhline(y=np.mean(l_j_history), color='r', linestyle='--', label="Mean $l_j$")
# plt.xlabel("Time step")
# plt.ylabel("Estimated Utility $l_j$_protein")
# plt.legend()
# plt.grid(False)
# plt.tight_layout()
# plt.show()

# threshold_list = eta_t[:len(l_j_history)] / phi
# plt.figure(figsize=(8, 4.2))
# plt.plot(l_j_history, label=r"$\hat{l}_j$ (Estimated Utility)", linewidth=2)
# plt.plot(eta_t[:len(l_j_history)] / phi, label=r"$\eta_j / \phi$ (Query Threshold)", linestyle="--", linewidth=2)
# plt.xlabel("Time step")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(False)
# plt.tight_layout()
# plt.savefig("utility_vs_threshold_protein.pdf")
# plt.show()




methods = ["OVBAL", "BIA", "RS", "OVBAL(no cost)"]
schemes = ["SC", "BC"]

# SC results
labels_sc = [label_count_ovbal, label_count_bia, label_count_rs,label_count_ovbal_no]
costs_sc = [cost_ovbal[-1], cost_bia[-1], cost_rs[-1],cost_ovbal_no[-1]]
utils_sc = [utility_ovbal_sc, utility_bia_sc, utility_rs_sc, utility_ovbal_sc_no]
eff_sc = np.array(utils_sc) / (np.array(costs_sc) + 1e-8)

# BC results
labels_bc = [results_ovbal_bc[6], results_bia_bc[6], results_rs_bc[6],results_ovbal_bc_no[6]]
costs_bc = [results_ovbal_bc[3][-1], results_bia_bc[3][-1], results_rs_bc[3][-1],results_ovbal_bc_no[3][-1]]
utils_bc = [results_ovbal_bc[7], results_bia_bc[7], results_rs_bc[7],results_ovbal_bc_no[7]]
eff_bc = np.array(utils_bc) / (np.array(costs_bc) + 1e-8)


print("=== Table: Total Utility (SC vs BC) ===")
print(f"{'Method':<8} | {'Scheme':<4} | {'Utility':<10} | {'#Labels':<8} | {'Cost':<10} | {'Util/Cost':<10}")
print("-" * 70)
for i in range(len(methods)):
    print(f"{methods[i]:<8} | {'SC':<4} | {utils_bc[i]:<10.2f} | {labels_sc[i]:<8} | {costs_sc[i]:<10.1f} | {utils_bc[i]/costs_sc[i]:<10.2f}")
for i in range(len(methods)):
    print(f"{methods[i]:<8} | {'BC':<4} | {utils_sc[i]:<10.2f} | {labels_bc[i]:<8} | {costs_bc[i]:<10.1f} | {utils_sc[i]/costs_bc[i]:<10.2f}")





