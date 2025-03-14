# NetNinja : Network Traffic Analysis Chatbot

## Overview
This project focuses on analyzing two types of network traffic:
1. **Regular (Normal) Traffic** – Enterprise-level and 5G network traffic.
2. **Attack Traffic** – Malicious traffic data including DDoS and security incidents.

By leveraging machine learning techniques, particularly **time-series forecasting**, the project aims to model, analyze, and predict network behaviors using a diverse set of datasets.

## Datasets Considered
The following five datasets were selected for analysis:

### **1. 5GTrafficDataset**
- Captures real-world 5G traffic collected from mobile devices using **PCAPdroid**.
- Includes timestamps, packet-level details, and traffic from applications like Netflix, YouTube Live, and Zoom.
- Useful for understanding high-throughput and low-latency network behaviors.

### **2. CSE-CIC-IDS2018 Normal Traffic**
- Provides enterprise-level network traffic logs for intrusion detection research.
- Includes packet flow statistics, source/destination details, and session durations.
- Used to benchmark normal network traffic in security studies.

### **3. INDDOS24Dataset**
- A dataset focused on **Distributed Denial-of-Service (DDoS) attacks**.
- Contains timestamps, attack types, and varying network load conditions.
- Ideal for studying traffic anomalies and DDoS mitigation strategies.

### **4. MicrosoftSecurityIncidentPrediction**
- Large-scale dataset for security incident forecasting.
- Includes security alerts, remediation actions, and enterprise attack logs.
- Helps in modeling long-term network security trends.

### **5. TokyoDroneCommunicationAndSecurityDataset**
- Captures network communication between drones and ground stations.
- Includes normal and malicious traffic patterns in **IoT-based systems**.
- Useful for analyzing **wireless communication security** in drone networks.

You can access the datasets via the following Google Drive link: 
https://drive.google.com/drive/folders/1hUmka8MZSmpYvkBCLqX5-Ri_3y4i1fBC?usp=drive_link

## Data Preprocessing Approach
A **custom preprocessing pipeline** was implemented using the **Temporal Fusion Transformer (TFT) framework**. This approach enables:
- **Time-series modeling** by extracting timestamp-based features.
- **Normalization and scaling** using **MinMaxScaler** for real-valued columns.
- **Categorical encoding** using **LabelEncoder** for network protocol data.
- **Handling missing values** by replacing NaNs/Infs with median values.
- **Feature engineering** to create new time-based inputs (e.g., Hour, Minute, Second).

### **Preprocessing Code Overview**
1. **GenericDataFormatter (Abstract Class)** – Defines the general structure.
2. **TrafficDataFormatter (Custom Implementation)** – Implements dataset-specific transformations.
3. **Scaling & Encoding** – MinMaxScaler for numerical features, LabelEncoder for categorical ones.
4. **Time-based Features** – Extracts temporal attributes for improved forecasting.
5. **Splitting Strategy** – 70% training, 15% validation, 15% testing.

```python
# Example transformation (simplified)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S")
df["Hour"] = df["Timestamp"].dt.hour
df["Minute"] = df["Timestamp"].dt.minute
df["Second"] = df["Timestamp"].dt.second
df.drop(columns=["Timestamp"], inplace=True)
```

## Folder Structure
```
/data
  ├── raw/  # Contains original datasets
  ├── preprocessed/
      ├── 5GTrafficDataset/
      │   ├── train.csv
      │   ├── val.csv
      │   ├── test.csv
      ├── CSE-CIC-IDS2018NormalTraffic/
      ├── INDDOS24Dataset/
      ├── MicrosoftSecurityIncidentPrediction/
      ├── TokyoDroneCommunicationAndSecurityDataset/
```

### **Generated Output Files**
- **train.csv** – Training dataset (70% split)
- **val.csv** – Validation dataset (15% split)
- **test.csv** – Testing dataset (15% split)

## Conclusion
This study provides a **unified approach** to analyzing normal and malicious network traffic using **TFT-based preprocessing**. The structured datasets facilitate **time-series forecasting** and **attack detection**, paving the way for advanced network monitoring solutions.

