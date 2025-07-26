# forecasting

.
├── ITSM_data.csv         # Input CSV data file
├── app/
│   ├── main.py           # FastAPI application
│   ├── forecaster.py     # Core forecasting logic and model training
│   └── models/           # Directory to store trained models (created by the app)
├── Dockerfile            # Docker configuration for containerization
├── requirements.txt      # Python dependencies
├── .dockerignore         # Files to ignore when building Docker image
├── prometheus.yml        # Prometheus configuration
├── .github/
│   └── workflows/
│       └── main.yml      # GitHub Actions CI/CD pipeline
└── README.md             # This file
