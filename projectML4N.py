import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
df=pd.read_csv('/Users/gabrielemoschettini/Desktop/machine learning for networking/https_training.csv')

print("First dataset info")

#check for the number of columns and rows
print('\n')
print(f"Number of columns : " , len(df.columns))
print(f"Number of rows : " , len(df)) 

#part 1.1
target_col = 'label'
ip_col = 'c_ip'
numeric_cols = [c for c in df.columns if c.startswith('_')]
print("\FLOW LEVEL(disribution of flows)")

# Visualizziamo i dati grezzi dove ogni riga è una singola connessione TCP[cite: 36, 49].


features_to_plot = ['_c_bytes_all', '_s_rtt_avg', '_c_pkts_all']

plt.figure(figsize=(15, 5))
for i, col in enumerate(features_to_plot):
    plt.subplot(1, 3, i+1)

    log_scale = True if 'bytes' in col else False
    
    sns.histplot(df[col], kde=True, log_scale=log_scale)
    plt.title(f'Flow Level: {col}')
plt.tight_layout()
plt.show()


# 2 Ip level per client
# Raggruppiamo i dati per indirizzo IP (`c_ip`) calcolando la media delle feature.
# Questo crea un profilo per ogni utente, permettendoci di vedere se ci sono client con comportamenti anomali.
print("\n IP LEVEL (Aggregation per client")


df_ip = df.groupby(ip_col)[numeric_cols].mean()

print(f"Number of clients: {len(df_ip)}")

plt.figure(figsize=(15, 5))
for i, col in enumerate(features_to_plot):
    plt.subplot(1, 3, i+1)
    log_scale = True if 'bytes' in col else False
    
    # Qui plottiamo la distribuzione delle MEDIE per utente
    sns.histplot(df_ip[col], kde=True, log_scale=log_scale, color='orange')
    plt.title(f'IP Level (Mean): {col}')
plt.tight_layout()
plt.show()


#4. Analisi Livello 3: DOMAIN LEVEL (Per Sito Web)
print("\n--- DOMAIN LEVEL ---")
#Raggruppiamo per etichetta (`label`) per ottenere l'impronta digitale media di ogni servizio (es. Google vs Amazon)[cite: 11, 48].
# Fondamentale per capire se i servizi sono distinguibili in base al traffico (utile per la classificazione).
# Raggruppiamo per 'label' (Sito web) e calcoliamo la MEDIA
df_domain = df.groupby(target_col)[numeric_cols].mean()

print(f"Number of Domains:  {len(df_domain)}")

plt.figure(figsize=(15, 5))
for i, col in enumerate(features_to_plot):
    plt.subplot(1, 3, i+1)
    log_scale = True if 'bytes' in col else False
    
    sns.histplot(df_domain[col], kde=True, log_scale=log_scale, color='green')
    plt.title(f'Domain Level (Mean): {col}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.ecdfplot(data=df_domain, x='_c_bytes_all')
plt.title('ECDF dei Bytes medi per Dominio')
plt.xlabel('Media Bytes')
plt.grid(True)
plt.show()
