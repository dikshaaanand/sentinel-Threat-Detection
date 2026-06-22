# sentinel-Threat-Detection

Real-time cybersecurity threat detection with explainable AI. Classifies network attacks using Random Forest + SHAP explanations.

1. Install & Setup
bashpip install -r requirements.txt

2. Download Dataset
Get KDDTrain+.txt from KDD Cup 99 and place in project root.

3. Train Models
bashpython model.py
python explain.py

4. Run API
bashpython api.py

5. Open Dashboard
Visit http://localhost:5000 in your browser


What It Does--
Detects threats in real-time via WebSocket stream
Classifies attacks into 4 categories: DDoS, PortScan, Credential, Exploit
Explains predictions with SHAP feature importance
Risk scores 0-100 for each threat
Beautiful dashboard with live metrics, threat feed, and XAI analysis
