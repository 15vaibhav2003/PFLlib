import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "/kaggle/working/PFLlib"
logs = base + "/system/logs"
out = base + "/plots"
os.makedirs(out, exist_ok=True)

runs = {
    "fedavg_seed0": "FedAvg (noisy client 4)",
    "fedprox_mu0.01_seed0": "FedProx mu=0.01",
    "fedprox_mu0.1_seed0": "FedProx mu=0.1",
    "fedavg_clean_seed0": "FedAvg (no noise)",
}
ev = {}
st = {}
for r in runs:
    ev[r] = pd.read_csv(f"{logs}/{r}_eval.csv")
    st[r] = pd.read_csv(f"{logs}/{r}_stats.csv")
cl = [0, 1, 2, 3, 4]

# 1) global test accuracy, all runs
plt.figure(figsize=(7, 4.5))
for r, lab in runs.items():
    plt.plot(ev[r]["round"], ev[r]["global/test_acc"], label=lab)
plt.xlabel("Round")
plt.ylabel("Global test accuracy")
plt.title("PathMNIST, 5 clients, Dirichlet alpha=0.3")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{out}/global_acc.png", dpi=150)
plt.close()

# 2) per-client test accuracy for FedAvg and FedProx mu=0.01
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, r in zip(axs, ["fedavg_seed0", "fedprox_mu0.01_seed0"]):
    for c in cl:
        lab = f"client {c}" + (" (40% noisy)" if c == 4 else "")
        lw = 2.5 if c == 4 else 1.2
        ax.plot(ev[r]["round"], ev[r][f"client_{c}/test_acc"], label=lab, linewidth=lw)
    ax.set_title(runs[r])
    ax.set_xlabel("Round")
    ax.grid(alpha=0.3)
axs[0].set_ylabel("Client test accuracy")
axs[1].legend()
plt.tight_layout()
plt.savefig(f"{out}/client_acc.png", dpi=150)
plt.close()

# 3) final per-client accuracy (mean of last 5 rounds), grouped bars
fig, ax = plt.subplots(figsize=(9, 4.5))
w = 0.2
for i, (r, lab) in enumerate(runs.items()):
    vals = []
    for c in cl:
        vals.append(ev[r][f"client_{c}/test_acc"].tail(5).mean())
    xs = []
    for c in cl:
        xs.append(c + (i - 1.5) * w)
    ax.bar(xs, vals, width=w, label=lab)
ax.set_xticks(cl)
ax.set_xticklabels([f"client {c}" for c in cl])
ax.set_ylabel("Test accuracy (mean of last 5 rounds)")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{out}/client_final_acc.png", dpi=150)
plt.close()

# 4) update norm and cosine per client (FedAvg with noisy client 4)
r = "fedavg_seed0"
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for c in cl:
    lab = f"client {c}" + (" (noisy)" if c == 4 else "")
    lw = 2.5 if c == 4 else 1.2
    axs[0].plot(st[r]["round"], st[r][f"client_{c}/update_norm"], label=lab, linewidth=lw)
    axs[1].plot(st[r]["round"], st[r][f"client_{c}/cosine"], label=lab, linewidth=lw)
    axs[2].plot(st[r]["round"], st[r][f"client_{c}/cosine_loo"], label=lab, linewidth=lw)
axs[0].set_title("Update norm ||w_k - w_global||")
axs[1].set_title("Cosine(delta_k, aggregated update)")
axs[2].set_title("Cosine with the other clients (leave-one-out)")
for ax in axs:
    ax.set_xlabel("Round")
    ax.grid(alpha=0.3)
axs[0].legend()
plt.suptitle(runs[r])
plt.tight_layout()
plt.savefig(f"{out}/norm_cosine.png", dpi=150)
plt.close()

# 5) results table (printed and saved as CSV)
rows = []
for r, lab in runs.items():
    e = ev[r]
    s = st[r]
    row = {
        "run": r,
        "final_global_acc (last 5 mean)": round(e["global/test_acc"].tail(5).mean(), 4),
        "best_global_acc": round(e["global/test_acc"].max(), 4),
        "client_acc_std (last 5 mean)": round(e["global/test_acc_std"].tail(5).mean(), 4),
    }
    for c in cl:
        row[f"client_{c}_acc"] = round(e[f"client_{c}/test_acc"].tail(5).mean(), 4)
    row["client_4_norm / others"] = round(
        s["client_4/update_norm"].mean() / s[[f"client_{c}/update_norm" for c in [0, 1, 2, 3]]].mean(axis=1).mean(), 3)
    row["client_4_cos_loo"] = round(s["client_4/cosine_loo"].mean(), 3)
    row["others_cos_loo"] = round(s[[f"client_{c}/cosine_loo" for c in [0, 1, 2, 3]]].mean(axis=1).mean(), 3)
    rows.append(row)
tab = pd.DataFrame(rows)
tab.to_csv(f"{out}/results_table.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(tab.T.to_string())
print("\nSaved plots:", sorted(os.listdir(out)))
