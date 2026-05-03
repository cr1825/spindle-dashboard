# Spindle Stiffness ML Dashboard — Azure Deployment

A production Flask + XGBoost dashboard for spindle stiffness classification using 3-sensor fusion (accelerometer, microphone, current) across 5 temperature classes (27°–45°C).

## Project Structure

```
spindle_dashboard/
├── app.py                          # Flask app + ML pipeline
├── startup.py                      # Gunicorn entry point (trains model on boot)
├── requirements.txt
├── Procfile
├── web.config                      # Azure App Service config
├── templates/
│   └── index.html                  # Full dashboard UI
├── *.csv                           # All 15 sensor data files (3 sensors × 5 temps)
└── README.md
```

---

## Deploy to Azure App Service (Recommended)

### Prerequisites
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed
- Azure subscription
- Python 3.11

### Step 1 — Login & create resources

```bash
az login

# Create resource group
az group create --name spindle-rg --location eastus

# Create App Service plan (B2 recommended for ML startup time)
az appservice plan create \
  --name spindle-plan \
  --resource-group spindle-rg \
  --sku B2 \
  --is-linux

# Create web app
az webapp create \
  --name spindle-ml-dashboard \
  --resource-group spindle-rg \
  --plan spindle-plan \
  --runtime "PYTHON:3.11"
```

### Step 2 — Configure startup command

```bash
az webapp config set \
  --name spindle-ml-dashboard \
  --resource-group spindle-rg \
  --startup-file "gunicorn --bind=0.0.0.0:8000 --timeout=300 --workers=1 startup:app"
```

### Step 3 — Deploy via ZIP

```bash
# From inside spindle_dashboard/ directory
zip -r ../dashboard.zip . -x "*.pyc" -x "__pycache__/*" -x "venv/*"

az webapp deploy \
  --name spindle-ml-dashboard \
  --resource-group spindle-rg \
  --src-path ../dashboard.zip \
  --type zip
```

### Step 4 — Open dashboard

```bash
az webapp browse --name spindle-ml-dashboard --resource-group spindle-rg
```

Your dashboard will be live at: `https://spindle-ml-dashboard.azurewebsites.net`

---

## Alternative: Deploy via GitHub Actions (CI/CD)

1. Push this folder to a GitHub repo
2. In Azure Portal → your App Service → **Deployment Center**
3. Select **GitHub** → authorize → pick your repo/branch
4. Azure auto-generates `.github/workflows/main_spindle.yml`
5. Every push to `main` redeploys automatically

---

## Local Development

```bash
cd spindle_dashboard
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python startup.py               # Trains model then starts Flask on :8000
```

Open `http://localhost:8000`

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/api/results` | GET | All ML metrics (accuracy, CV, confusion matrix, features) |
| `/api/predict` | POST | Real-time prediction from sensor features |
| `/api/health` | GET | Health check |

### Predict endpoint example

```bash
curl -X POST https://spindle-ml-dashboard.azurewebsites.net/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"acc_std": 85.4, "cur_rms": 0.19, "mic_std": 0.18}}'
```

Response:
```json
{
  "predicted_class": "35°",
  "probabilities": {
    "27°": 0.04, "30°": 0.11, "35°": 0.58, "40°": 0.19, "45°": 0.08
  }
}
```

---

## Azure Pricing Notes

| SKU | vCPU | RAM | ~Cost/month | Notes |
|---|---|---|---|---|
| B1 | 1 | 1.75 GB | ~$13 | May timeout on model training |
| **B2** | 2 | 3.5 GB | ~$26 | **Recommended** |
| B3 | 4 | 7 GB | ~$52 | For production traffic |

> Model trains on every cold start (~30–60s). For faster cold starts, consider saving the trained model with `joblib` to Azure Blob Storage and loading it on startup instead of retraining.

---

## Model Summary

- **Algorithm**: XGBoost Classifier (500 trees, max_depth=4, lr=0.05)
- **Sensors**: Accelerometer + Microphone + Current (22 features each = 66 total)
- **Feature selection**: Top 30 by XGBoost importance
- **Test accuracy**: ~76.5%
- **Cross-val accuracy**: ~76.7% ± 3.1% (5-fold stratified)
- **Best class**: 27°C — F1 = 87.2%
