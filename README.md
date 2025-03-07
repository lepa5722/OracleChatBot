# OracleChatBot
**NetNinja** : Oracle Chatbot for Cybersecurity Attack Prediction

NetNinja is a cybersecurity-focused chatbot that protects against and predicts cyber attacks using time-series forecasting. As the first step in this project, we have considered five datasets for analysis.

**Datasets Considered**

**1. CyberThreatDetection**

- This dataset is designed for cyber threat detection using machine learning techniques.
- It includes various network traffic features such as packet size, duration, flow statistics, and attack labels (DDoS, Brute Force, Ransomware, etc.).
- Ideal for training intrusion detection systems (IDS) and anomaly detection models in a cybersecurity context.

**2. INDDOS24Dataset**

- A dataset focused on Distributed Denial-of-Service (DDoS) attacks, one of the most prevalent cyber threats today.
- It contains network traffic logs, attack patterns, and timestamps, allowing for the development of DDoS mitigation models.
- Helps improve real-time threat detection by analyzing variations in network traffic behavior.

**3. MicrosoftSecurityIncidentPrediction**

- A large-scale dataset provided by Microsoft Security AI Research for predicting security incidents.
- It includes over 1 million security alerts, 13 million evidence pieces, and 26,000 remediation actions from real-world enterprise environments.
- Used for threat forecasting, guided response systems, and automated cybersecurity incident triage.
- A valuable resource for predicting and mitigating security incidents before they escalate.

**4. NetworkTrafficData**

- Captures network traffic metadata to analyze normal and malicious activities.
- Includes features such as IP addresses, protocols, bytes transferred, and traffic flow rates.
- Useful for identifying anomalies, traffic monitoring, and understanding malicious communication patterns.
- Helps in time-series forecasting of cyber threats based on traffic variations.

**5. TokyoDroneCommunicationAndSecurityDataset**

- A unique dataset focusing on security threats in drone communication networks.
- Includes logs of drone-to-drone and drone-to-ground station communication with security event labels.
- Critical for securing UAV (Unmanned Aerial Vehicle) networks, preventing GPS spoofing, and detecting unauthorized drone activity.
- Helps expand cybersecurity research into aerial and IoT-based communication systems.

**Dataset Storage Structure**

All datasets are organized within the data folder, which contains the following subfolders:

1. **raw/** → Contains the original versions of the five datasets

2. **preprocessed/** → Contains the preprocessed versions of each dataset

**Data Preprocessing**

The preprocessing of the raw datasets is handled by the **PreProcessing.py** script located in the **scripts/** folder. This script cleans and transforms the datasets to ensure they are ready for training the prediction model

**Model Training**

The preprocessed datasets are then used for training the prediction model

**Dataset Access**

The datasets are available at the following Google Drive link:

https://drive.google.com/drive/folders/1hUmka8MZSmpYvkBCLqX5-Ri_3y4i1fBC?usp=sharing


