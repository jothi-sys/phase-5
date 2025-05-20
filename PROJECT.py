import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from hashlib import sha256
from datetime import datetime
import randompip install pandas matplotlib scikit-learndef generate_energy_data(days=30):
    timestamps = pd.date_range(start="2025-01-01", periods=days, freq='D')
    base_usage = np.random.normal(loc=30, scale=5, size=days)  # kWh
    temperature = np.random.normal(loc=22, scale=5, size=days)  # Celsius
    energy_data = pd.DataFrame({
        'Date': timestamps,
        'Energy_Usage_kWh': base_usage,
        'Temperature_C': temperature
    })

    # Add a few anomalies
    energy_data.loc[random.randint(5, 25), 'Energy_Usage_kWh'] += random.randint(15, 30)
    return energy_datadef detect_anomalies(df):
    model = IsolationForest(contamination=0.1)
    df['Anomaly'] = model.fit_predict(df[['Energy_Usage_kWh']])
    return dfdef generate_recommendations(df):
    mean_usage = df['Energy_Usage_kWh'].mean()
    recommendations = []

    if mean_usage > 32:
        recommendations.append("Consider using smart thermostats to reduce average consumption.")
    if (df['Anomaly'] == -1).sum() > 0:
        recommendations.append("Anomalies detected. Investigate sudden spikes in energy use.")
    if df['Temperature_C'].mean() > 25:
        recommendations.append("High temperatures detected — optimize cooling systems.")

    return recommendationsdef generate_hash(record):
    return sha256(str(record).encode()).hexdigest()

def simulate_blockchain_log(df):
    df['Hash'] = df.apply(lambda row: generate_hash(row.to_dict()), axis=1)
    return dfdef multilingual_message(language='en'):
    messages = {
        'en': "Welcome to your AI-powered energy dashboard.",
        'es': "Bienvenido a su panel de energía impulsado por IA.",
        'fr': "Bienvenue sur votre tableau de bord énergétique IA.",
        'de': "Willkommen bei Ihrem KI-Energie-Dashboard."
    }def plot_energy_usage(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Energy_Usage_kWh', data=df, label='Energy Usage')
    plt.scatter(
        df[df['Anomaly'] == -1]['Date'],
        df[df['Anomaly'] == -1]['Energy_Usage_kWh'],
        color='red', label='Anomaly', zorder=5
    )
    plt.title('Energy Usage Over Time')
    plt.xlabel('Date')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_usage_bar(df):
    plt.figure(figsize=(10, 5))
    df.groupby(df['Date'].dt.day)['Energy_Usage_kWh'].mean().plot(kind='bar', color='skyblue')
    plt.title('Average Daily Energy Usage')
    plt.xlabel('Day of Month')
    plt.ylabel('kWh')
    plt.tight_layout()
    plt.show()
    return messages.get(language, messages['en'])# Greet the user
print(multilingual_message('en'))

# Step-by-step processing
data = generate_energy_data(30)
data = detect_anomalies(data)
data = simulate_blockchain_log(data)

# Show sample
print("\nSample Energy Data with Anomaly and Hash:\n", data.head())

# AI-based Recommendations
recs = generate_recommendations(data)
print("\nAI Recommendations:")
for rec in recs:
    print("✔", rec)

# Graphs
plot_energy_usage(data)
plot_usage_bar(data)
