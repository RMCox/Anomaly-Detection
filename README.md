# Anomaly-Detection

[Anomaly Detection Report PDF](https://github.com/RMCox/Anomaly-Detection/blob/main/Anomaly%20Detection%20in%20Cybersecurity%20Data.pdf)

Built a Streamlit dashboard for Anomaly Detection in DNS logs. Identified 2 kinds of anomaly; Time intervals with anomalous values for key features and Items which exhibit symptoms of DNS exploits

Implemented Autoencoders for outlier detection (and Median Absolute Deviation for static features).
Assessed the strengths and weaknesses of various outlier detection methods, such as robustness.
Engineered features that would capture the symptoms of certain DNS exploits (Botnet Heatbeats, DNS Tunnelling, DDoS Attacks, Domain Flux).
Implemented a risk-scoring system for these features, similar to providers such as Splunk.
Allowed for custom anomaly detection checks within the dashboard, specifying the desired method and time-interval
Allowed for retraining of neural networks within the dashboard
Allowed for the exporting of anomalies within the dashboard as a csv
