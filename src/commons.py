import pandas as pd
import seaborn as sns

def score( y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    # roc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

def outlier_plot(
        data: pd.DataFrame, outlier_method_name: str, x_var: str, y_var: str, xaxis_limits=[0, 1], yaxis_limits=[0, 1]
):
    print(f"Outlier method: {outlier_method_name}")
    method = f"{outlier_method_name}_anomaly"
    
    print(f"Number of anomalous values: {len(data[data['is_anomaly'] == -1])}")
    print(f"Number of non anomalous values: {len(data[data["is_anomaly"] == 1])}")

    g = sns.FacetGrid(data, col="is_anomaly", height=4, hue="is_anomaly", hue_order=[1, -1])
    g.map(sns.scatterplot, x_var, y_var)
    g.fig.suptitle(f"Outlier method: {outlier_method_name}", y=1.10, fontweight="bold")
    g.set(xlim=xaxis_limits, ylim=yaxis_limits)
    axes = g.axes.flatten()
    axes[0].set_title(f"Outliers\n{len(data[data["is_anomaly"] == -1])} points")
    axes[1].set_title(f"Inliers\n{len(data[data["is_anomaly"] == 1])} points")
    return g