# AAAI 2026 code: toy example
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from numpy.linalg import pinv


np.random.seed(42)

# Synthetic data generation
n_samples = 10000
n_features = 3
X = np.random.randn(n_samples, n_features)
epsilon = np.random.randn(n_samples, 1)*0.3

lambda_factor=0.98
label_budget=2700
alpha=0.1
phi=5

n_labeled = int(0.3 * n_samples)


# Time-varying true beta
t_full = np.arange(n_samples)  
true_beta_full = np.array([
    1 + 0.0002 * t_full,         
    np.sin(0.005 * t_full),      
    0.5 + 0.001 * t_full,        
    0.0003 * t_full             
]).T


# Generate Y_true using time-varying beta
Y_true = np.sum(X * true_beta_full[:, 1:], axis=1, keepdims=True) + true_beta_full[:, 0:1] + epsilon

# Split data
D_L_x = X[:n_labeled]
D_L_y = Y_true[:n_labeled]
D_U   = X[n_labeled:]
D_ML  = Y_true[n_labeled:]
true_beta = true_beta_full[n_labeled:]
beta_0 = true_beta[:, 0].reshape(-1, 1)
beta_features = true_beta[:, 1:]

D_L_aug = np.hstack((np.ones((D_L_x.shape[0], 1)), D_L_x))
H_inv = np.linalg.pinv(D_L_aug.T @ D_L_aug)  # Use pseudo-inverse for stability
beta_prev = H_inv @ D_L_aug.T @ D_L_y
print(beta_prev)

D_L_aug = np.hstack((np.ones((D_L_x.shape[0], 1)), D_L_x))
H_inv = np.linalg.pinv(D_L_aug.T @ D_L_aug)  # Use pseudo-inverse for stability


log_eta_mu, log_eta_var = -1,0.25
eta_t = np.random.lognormal(mean=log_eta_mu,sigma=log_eta_var, size=D_U.shape[0])

labeled_cost = 0
label_count = 0
total_utility = 0



def compute_per_feature_tracking_error(beta_history, true_beta):
    beta_arr = np.array(beta_history).squeeze()
    errors = (beta_arr - true_beta[:beta_arr.shape[0]]) ** 2  
    return np.mean(errors, axis=0) 

def update_tau_online(mu_ovbal, var_ovbal, score_t, lambda_factor, alpha):
    mu = lambda_factor * mu_ovbal + (1 - lambda_factor) * score_t
    var = lambda_factor * var_ovbal + (1 - lambda_factor) * (score_t - mu) ** 2
    std = np.sqrt(var)
    z = norm.ppf(1 - alpha)
    tau = mu + z * std
    return tau, mu, var



def online_active_learning(D_L_x, D_L_y, D_U,  lambda_factor, beta_prev, 
                           label_budget, alpha, cost_aware, model,
                           phi, eta_t, verbose, scheme):
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
        true_b0 = true_beta[t, 0]
        true_b = true_beta[t, 1:]
        x_t = D_U[t].reshape(-1, 1)
        x_t_aug = np.vstack(([[1]], x_t))
        y_true_t = D_ML[t]

        possibility = 0.05

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
    


print(beta_prev.copy())

# Run both ovbal and oqbcal in cost_aware
results_ovbal_sc = online_active_learning(D_L_x, D_L_y, D_U, lambda_factor=lambda_factor, beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha,  cost_aware=True, model= "ovbal",
                           phi=phi, eta_t= eta_t, verbose=True, scheme="SC")
results_bia_sc = online_active_learning(D_L_x, D_L_y, D_U, lambda_factor=lambda_factor,beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha,  cost_aware=True, model= "Bia",
                           phi=phi, eta_t= eta_t, verbose=False, scheme="SC")
results_rs_sc = online_active_learning(D_L_x, D_L_y, D_U, lambda_factor=lambda_factor,beta_prev=beta_prev.copy(),
                           label_budget=label_budget, alpha=alpha,  cost_aware=True,model= "RS",
                           phi=phi, eta_t= eta_t, verbose=False,  scheme="SC")

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

print(beta_prev.copy())

results_ovbal_bc = online_active_learning(D_L_x, D_L_y, D_U,lambda_factor=lambda_factor, beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha, cost_aware=True,  model= "ovbal",
                           phi=phi, eta_t= eta_t, verbose=False, scheme="BC")
results_bia_bc = online_active_learning(D_L_x, D_L_y, D_U, lambda_factor=lambda_factor,  beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha,  cost_aware=True, model= "Bia",
                           phi=phi, eta_t= eta_t, verbose=False, scheme="BC")
results_rs_bc = online_active_learning(D_L_x, D_L_y, D_U,lambda_factor=lambda_factor,  beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha,  cost_aware=True, model= "RS",
                           phi=2, eta_t= eta_t, verbose=False, scheme="BC")

cost_ovbal_bc = results_ovbal_bc[3]
cost_bia_bc = results_bia_bc[3]
cost_rs_bc = results_rs_bc[3]
utility_ovbal_bc = results_ovbal_bc[7]
utility_bia_bc = results_bia_bc[7]
utility_rs_bc = results_rs_bc[7]
# Run both ovbal and oqbcal in no_cost_aware
results_ovbal_sc_no = online_active_learning(D_L_x, D_L_y, D_U, lambda_factor=lambda_factor, beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha,  cost_aware=False, model= "ovbal",
                           phi=phi, eta_t= eta_t, verbose=True, scheme="SC")
results_ovbal_bc_no = online_active_learning(D_L_x, D_L_y, D_U,lambda_factor=lambda_factor, beta_prev=beta_prev.copy(), 
                           label_budget=label_budget, alpha=alpha, cost_aware=False,  model= "ovbal",
                           phi=phi, eta_t= eta_t, verbose=False, scheme="BC")

avg_mse_ovbal_no=results_ovbal_sc_no[4]
label_count_ovbal_no=results_ovbal_sc_no[6]
cost_ovbal_no=results_ovbal_sc_no[3]
utility_ovbal_sc_no=results_ovbal_sc_no[7]
tau_ovbal_history_no=results_ovbal_sc_no[5]


avg_mse_ovbal_bc_no=results_ovbal_bc_no[4]
label_count_ovbal_bc_no=results_ovbal_bc_no[6]
cost_ovbal_bc_no=results_ovbal_bc_no[3]
utility_ovbal_bc_no=results_ovbal_bc_no[7]




# Per-feature tracking error
per_feature_err_ovbal = compute_per_feature_tracking_error(beta_ovbal, true_beta)
per_feature_err_sl = compute_per_feature_tracking_error(beta_bia, true_beta)
per_feature_err_rs = compute_per_feature_tracking_error(beta_rs, true_beta)

plt.rcParams.update({
    "font.family": "serif",         # AAAI prefers serif fonts
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

def to_subscript(n):
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(subscript_map)

plt.figure(figsize=(6.5, 4.2))
features = [f"β{to_subscript(i)}" for i in range(true_beta.shape[1])]
x = np.arange(len(features))
width = 0.2
plt.bar(x - 0.5 * width, per_feature_err_ovbal, width, label="OVBAL")
plt.bar(x + 0.5 * width, per_feature_err_sl, width, label="BIA")
plt.bar(x + 1.5 * width, per_feature_err_rs, width, label="RS")
plt.xticks(x, features)
plt.ylabel("$MSE_{β_k}$")
plt.legend()
plt.tight_layout()
plt.savefig("per_feature_beta_tracking_error.pdf")
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(20, 4.5), sharey=True)
methods = ["ovbal",  "Bia", "RS"]
betas = [beta_ovbal, beta_bia, beta_rs]
linestyles = ["--", ":"]

for i, ax in enumerate(axs):
    for j in range(true_beta.shape[1]):
        ax.plot(true_beta[:, j], linestyle=linestyles[0], label=f"True $β_{j}$" if i == 0 else "")
        ax.plot(betas[i][:, j], linestyle=linestyles[1], label=f"Estimated $β_{j}$" if i == 0 else "")
    ax.set_xlabel("Time step")
    if i == 0:
        ax.set_ylabel("β value")
    ax.grid(False)
# Add legend only once
axs[0].legend(loc="upper left", ncol=1)
plt.tight_layout()
plt.savefig("beta_evolution_all_methods.pdf")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(h_inv_ovbal, label="OVBAL", linewidth=2)
plt.plot(h_inv_bia, label="BIA", linestyle="--", linewidth=2)
plt.plot(h_inv_rs, label="RS", linestyle=":", linewidth=2)
plt.xlabel("Time step")
plt.ylabel("$H_{t}^{-1}$")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("ht-1_all_models.pdf")
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(sigma_y_ovbal, label="OVBAL", linewidth=2)
plt.plot(sigma_y_bia, label="BIA", linestyle="--", linewidth=2)
plt.plot(sigma_y_rs,label="RS", linestyle=":", linewidth=2)
plt.xlabel("Time step")
plt.ylabel("$σ_{y}^{2}$")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("sigma_y_all_models.pdf")
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(var_beta_ovbal, label="OVBAL", linewidth=2)
plt.plot(var_beta_bia, label="BIA", linestyle="--", linewidth=2)
plt.plot(var_beta_rs, label="RS", linestyle=":", linewidth=2)
plt.xlabel("Time step")
plt.ylabel("Trace of Var($\\beta$)")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("trace_var_beta_all_models.pdf")
plt.show()




fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
methods = ["ovbal",  "Bia", "RS"]
true_utils = [true_util_ovbal,  true_util_bia, true_util_rs]
est_utils = [est_util_ovbal,  est_util_bia, est_util_rs]
for i, ax in enumerate(axs.flatten()):
    ax.plot(true_utils[i], label="True Utility", linestyle='-')
    ax.plot(est_utils[i], label="Estimated Utility", linestyle='--')
    ax.set_xlabel("Query Step")
    ax.set_ylabel("Utility")
    ax.legend()
    ax.grid(False)
plt.tight_layout()
plt.savefig("utility_over_time_subplots.pdf")
plt.show()




fig, axs = plt.subplots(1, 3, figsize=(20, 4.5), sharey=True)
methods = ["ovbal", "Bia", "RS"]
betas = [beta_ovbal,beta_bia, beta_rs]
linestyles = ["--", ":"]
n_features = beta_ovbal.shape[1]  # Fix here
for i, ax in enumerate(axs):
    for j in range(n_features):
        ax.plot(betas[i][:, j], linestyle=linestyles[1], label=f"Estimated $β_{j}$" if i == 0 else "")
    ax.set_xlabel("Time step")
    if i == 0:
        ax.set_ylabel("β value")
    ax.grid(False)

# Add legend only once
axs[0].legend(loc="upper left", ncol=1)
plt.tight_layout()
plt.savefig("beta_evolution_all_methods.pdf")
plt.show()


# Align learner curves to baseline final MSE at t=0
# === Baseline: No Update (before any model is trained on D_U) ===
beta_baseline = beta_prev.copy()
mse_no_update = []
mse=0


for t in range(len(D_U)):
    x_t = D_U[t].reshape(-1, 1)
    x_aug = np.vstack(([1], x_t))
    
    true_b = true_beta[t].reshape(-1, 1)
    y_true = float(true_b.T @ x_aug) + float(epsilon[n_labeled + t])
    
    y_pred = float(beta_baseline.T @ x_aug)
    mse_t = (y_true - y_pred) ** 2
    mse+=mse_t
    mse_no_update.append(mse)

initial_mse = mse_no_update[-1] / len(D_U)


baseline_curve = [initial_mse] * len(avg_mse_ovbal)  # constant line for baseline
plt.figure(figsize=(7.2, 4.5))
plt.plot(avg_mse_ovbal, label="OVBAL", linestyle="-")
plt.plot(avg_mse_bia, label="BIA", linestyle="-.")
plt.plot(avg_mse_rs, label="RS", linestyle="--")
plt.xlabel("Time step", fontsize=13)
plt.ylabel("Forecasting MSE", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(loc="upper right", fontsize=11)
plt.grid(False)
plt.tight_layout()
plt.savefig("mse_vs_time_aligned.pdf")
plt.show()


def extract_query_mse(mse_list, cost_list):
    queried_idx = [i for i in range(1, len(cost_list)) if cost_list[i] != cost_list[i - 1]]
    mse_at_query = [mse_list[i] for i in queried_idx if i < len(mse_list)]
    return queried_idx, mse_at_query

methods = ["OVBAL", "BIA", "RS"]
avg_mse_all = [avg_mse_ovbal, avg_mse_bia, avg_mse_rs]
costs_all = [cost_ovbal, cost_bia, cost_rs]
colors = ["blue", "orange", "green"]


query_curves = []
for mse, cost in zip(avg_mse_all, costs_all):
    x, y = extract_query_mse(mse, cost)
    x = list(range(1, len(y) + 1))
    query_curves.append((x, y))

zoom_x_max = min(len(query_curves[0][0]), len(query_curves[2][0])) 

# === Plot ===
fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)

# Full Curve
for (x, y), label, color in zip(query_curves, methods, colors):
    axs[0].plot(x, y, label=label, linewidth=3)
axs[0].set_xlabel("Queried Labels", fontsize=25)
axs[0].set_ylabel("Forecasting MSE", fontsize=25)
axs[0].tick_params(axis='x', labelsize=20)
axs[0].tick_params(axis='y', labelsize=20)
axs[0].legend(fontsize=18)
axs[0].grid(False)

# Zoom-in
for (x, y), label, color in zip(query_curves, methods, colors):
    axs[1].plot(x, y, label=label, linewidth=3)
axs[1].set_xlim(0, zoom_x_max)
axs[1].set_xlabel("Queried Labels", fontsize=25)
axs[1].tick_params(axis='x', labelsize=20)
axs[1].tick_params(axis='y', labelsize=20)
axs[1].grid(False)
plt.tight_layout()
plt.savefig("mse_vs_queried_labels_zoom_compare.pdf")
plt.show()


def extract_query_mse(mse_list, cost_list):
    queried_idx = [i for i in range(1, len(cost_list)) if cost_list[i] != cost_list[i - 1]]
    mse_at_query = [mse_list[i] for i in queried_idx if i < len(mse_list)]
    return queried_idx, mse_at_query
methods = ["OVBAL",  "BIA", "RS"]
avg_mse_all = [avg_mse_ovbal,  avg_mse_bia, avg_mse_rs]
costs_all = [cost_ovbal, cost_bia, cost_rs]
colors = ["blue", "green", "red"]

plt.figure(figsize=(7.5, 5))
for method, mse_hist, cost_hist, color in zip(methods, avg_mse_all, costs_all, colors):
    x, y = extract_query_mse(mse_hist, cost_hist)
    plt.plot(range(1, len(y) + 1), y, label=method, linestyle="-", linewidth=2)
plt.xlabel("Queried Labels")
plt.ylabel("Forecasting MSE")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("mse_vs_queried_labels.pdf")
plt.show()


def extract_query_cost_mse_threshold(mse_hist, cost_hist, threshold=np.inf):
    query_costs = []
    query_mse = []
    last_cost = 0
    for i in range(len(cost_hist)):
        if i >= len(mse_hist):
            break
        if cost_hist[i] > last_cost:
            if cost_hist[i] > threshold:
                break
            query_costs.append(cost_hist[i])
            query_mse.append(mse_hist[i])
            last_cost = cost_hist[i]
    return query_costs, query_mse

methods = ["OVBAL", "BIA", "RS"]
colors = ["blue", "orange", "green"]
mse_all = [avg_mse_ovbal, avg_mse_bia, avg_mse_rs]
cost_all = [cost_ovbal, cost_bia, cost_rs]

query_curves = []
for mse, cost in zip(mse_all, cost_all):
    x, y = extract_query_cost_mse_threshold(mse, cost)
    query_curves.append((x, y))

zoom_cost_max = 121

# === Plot ===
fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)

# Full Curve
for (x, y), label, color in zip(query_curves, methods, colors):
    axs[0].plot(x, y, label=label, linewidth=3)
axs[0].set_xlabel("Cost Spent", fontsize=25)
axs[0].set_ylabel("Forecasting MSE", fontsize=25)
axs[0].tick_params(axis='x', labelsize=20)
axs[0].tick_params(axis='y', labelsize=20)
axs[0].legend(fontsize=18)
axs[0].grid(False)

# Zoom-in
for (x, y), label, color in zip(query_curves, methods, colors):
    axs[1].plot(x, y, label=label, linewidth=3)
axs[1].set_xlim(0, zoom_cost_max)
axs[1].set_xlabel("Cost Spent", fontsize=25)
axs[1].tick_params(axis='x', labelsize=20)
axs[1].tick_params(axis='y', labelsize=20)
axs[1].grid(False)

plt.tight_layout()
plt.savefig("mse_vs_cost_spent_zoom_compare.pdf")
plt.show()


def extract_query_cost_mse_threshold(mse_hist, cost_hist, threshold=10000):
    query_costs = []
    query_mse = []
    last_cost = 0
    for i in range(len(cost_hist)):
        if i >= len(mse_hist):
            break
        if cost_hist[i] > last_cost:
            if cost_hist[i] > threshold:
                break
            query_costs.append(cost_hist[i])
            query_mse.append(mse_hist[i])
            last_cost = cost_hist[i]
    return query_costs, query_mse
methods = ["OVBAL", "BIA", "RS"]
mse_all = [avg_mse_ovbal,  avg_mse_bia, avg_mse_rs]
cost_all = [cost_ovbal, cost_bia, cost_rs]
plt.figure(figsize=(7, 4.2))
for m, mse, cost in zip(methods, mse_all, cost_all):
    q_costs, q_mses = extract_query_cost_mse_threshold(mse, cost, threshold=10000)
    plt.plot(q_costs, q_mses, label=m)
plt.xlabel("Cost Spent")
plt.ylabel("Forecasting MSE")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()




fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)

# SC Pricing
axs[0].plot(cost_ovbal, label="OVBAL", linestyle="-")
axs[0].plot(cost_bia, label="BIA", linestyle="-.")
axs[0].plot(cost_rs, label="RS", linestyle="--")
axs[0].axhline(label_budget, color='red', linestyle='--', linewidth=3)
axs[0].text(len(cost_ovbal)*0.55, label_budget*0.9, 'Budget Limit', color='red', fontsize=18)
axs[0].set_xlabel("Time step", fontsize=25)
axs[0].set_ylabel("Cumulative cost", fontsize=25)
axs[0].tick_params(axis='x', labelsize=20)
axs[0].tick_params(axis='y',labelsize=20)
axs[0].legend(fontsize=18)

# BC Pricing
axs[1].plot(cost_ovbal_bc, label="OVBAL", linestyle="-")
axs[1].plot(cost_bia_bc, label="BIA ", linestyle="-.")
axs[1].plot(cost_rs_bc, label="RS ", linestyle="--")
axs[1].axhline(label_budget, color='red', linestyle='--', linewidth=3)
axs[1].text(len(cost_ovbal_bc)*0.55, label_budget*0.9, 'Budget Limit', color='red',fontsize=18)
axs[1].set_xlabel("Time step",fontsize=20)
axs[1].tick_params(axis='x', labelsize=20)
axs[1].tick_params(axis='y', labelsize=20)
axs[1].legend(fontsize=18)
fig.tight_layout()
plt.savefig("cumulative_cost_toy.pdf")
plt.show()

methods = ["ovbal", "Bia", "RS"]
final_utilities = [utility_ovbal_sc,  utility_bia_sc, utility_rs_sc]
final_costs = [cost_ovbal[-1],  cost_bia[-1], cost_rs[-1]]
efficiency = np.array(final_utilities) / np.array(final_costs)
plt.figure(figsize=(6.5, 4.2))
x = np.arange(len(methods))
plt.bar(x, efficiency, tick_label=methods)
plt.ylabel("Utility / Cost (Efficiency)")
plt.tight_layout()
plt.savefig("efficiency_sc.pdf")
plt.show()



mu_ovbal_history =results_ovbal_sc[10]
var_ovbal_history=results_ovbal_sc[11]
score_history=results_ovbal_sc[12]

# --- Plot: Uncertainty score vs tau ---
plt.figure(figsize=(7, 4))
plt.plot(score_history, label=r"$x^\top H^{-1} x$", linewidth=1.6)
plt.plot(tau_ovbal_history, label=r"$\tau$", linestyle="--", linewidth=1.6)
plt.xlabel("Time step")
plt.ylabel("Uncertainty score")
plt.legend()
plt.tight_layout()
plt.savefig("uncertainty_vs_tau.pdf")
plt.show()

# --- Plot: Mean and Variance of UPV over time ---
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(mu_ovbal_history, label="OVBAL", color="blue", linewidth=1.6)
ax1.set_ylabel("μ", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(var_ovbal_history, label="OVBAL", color="red", linestyle="--", linewidth=1.6)
ax2.set_ylabel("σ² ", color="red")
ax2.tick_params(axis='y', labelcolor="red")
fig.tight_layout()
plt.savefig("ovbal_mu_var_evolution.pdf")
plt.show()

l_j_history=results_ovbal_sc[13]

threshold_list = eta_t[:len(l_j_history)] / phi
plt.figure(figsize=(8, 4.2))
plt.plot(l_j_history, label=r"$\hat{l}_j$ (Estimated Utility)", linewidth=2)
plt.plot(eta_t[:len(l_j_history)] / phi, label=r"$\eta_j / \phi$ (Query Threshold)", linestyle="--", linewidth=2)
plt.xlabel("Time step")
plt.ylabel("Value")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("utility_vs_threshold.pdf")
plt.show()


# without_cost_aware

mu_ovbal_history_no =results_ovbal_sc_no[10]
var_ovbal_history_no=results_ovbal_sc_no[11]
score_history_no=results_ovbal_sc_no[12]

# --- Plot: Uncertainty score vs tau ---
plt.figure(figsize=(7, 4))
plt.plot(score_history_no, label=r"$x^\top H^{-1} x$", linewidth=1.6)
plt.plot(tau_ovbal_history_no, label=r"$\tau$", linestyle="--", linewidth=1.6)
plt.xlabel("Time step")
plt.ylabel("Uncertainty score")
plt.legend()
plt.tight_layout()
plt.savefig("uncertainty_vs_tau.pdf")
plt.show()

# --- Plot: Mean and Variance of UPV over time ---
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(mu_ovbal_history_no, label="OVBAL", color="blue", linewidth=1.6)
ax1.set_ylabel("μ", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(var_ovbal_history_no, label="OVBAL", color="red", linestyle="--", linewidth=1.6)
ax2.set_ylabel("σ² ", color="red")
ax2.tick_params(axis='y', labelcolor="red")
fig.tight_layout()
plt.savefig("ovbal_mu_var_evolution.pdf")
plt.show()

l_j_history_no=results_ovbal_sc_no[13]

threshold_list = eta_t[:len(l_j_history)] / phi
plt.figure(figsize=(8, 4.2))
plt.plot(l_j_history_no, label=r"$\hat{l}_j$ (Estimated Utility)", linewidth=2)
plt.plot(eta_t[:len(l_j_history_no)] / phi, label=r"$\eta_j / \phi$ (Query Threshold)", linestyle="--", linewidth=2)
plt.xlabel("Time step")
plt.ylabel("Value")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("utility_vs_threshold.pdf")
plt.show()


