# MSL Python Backend

A FastAPI-based backend service for device fingerprinting and location matching using cellular, WiFi, and GPS data.

## Overview

MSL (Machine Sensing Location) is a system that collects and analyzes wireless signals (cellular and WiFi) along with GPS coordinates to create device fingerprints and perform location-based matching and learning. The backend provides API endpoints for managing devices, scans, locations, and fingerprints.

## Project Structure

### Core Modules

- **`app.py`** - FastAPI application initialization and router setup
- **`api.py`** - API route definitions and endpoint handlers
- **`db.py`** - Database configuration, models, and ORM setup using SQLAlchemy

### Feature Modules

- **`fingerprint.py`** - Device fingerprinting logic and feature extraction
- **`matching.py`** - Device matching algorithms (includes cosine similarity for signal comparison)
- **`learning.py`** - Machine learning model training and evaluation
- **`evaluation.py`** - Evaluation metrics and performance analysis

## Database Schema

The system uses PostgreSQL with the following main tables:

- **`devices`** - Stores unique device identifiers and metadata
- **`raw_scans`** - Raw cellular, WiFi, and GPS data from devices
- **`locations`** - Geographic locations identified by latitude/longitude centroids
- **`fingerprints`** - ML features extracted from location data

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- pip or conda

### Setup

1. Clone the repository:

```bash
git clone https://github.com/gitgatere/MSL_python_backend.git
cd MSL_python_backend
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variables:

```bash
# PostgreSQL connection string
export DATABASE_URL="postgresql://user:password@localhost:5432/msl_db"

# FastAPI settings (optional)
export ENVIRONMENT="development"
```

## Running the Application

Start the FastAPI server:

```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### View API Documentation

- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## API Endpoints

The API provides endpoints for:

- **Device Management** - Register and retrieve devices
- **Scan Data** - Submit and query cellular/WiFi/GPS scans
- **Locations** - Manage location data and centroids
- **Fingerprints** - Create and retrieve device fingerprints
- **Matching** - Match devices to locations using fingerprints
- **Learning** - Train and evaluate ML models

## Key Algorithms

### Device Matching

Uses cosine similarity to compare signal fingerprints:

- Cellular signal strength (RSSI) vectors
- WiFi signal strength vectors
- GPS coordinates for ground truth validation

### Feature Extraction

Extracts JSONB features from raw scan data for ML model training.

## Development

### Code Style

Follow PEP 8 conventions. Format code using:

```bash
black *.py
```

### Testing

Run tests with:

```bash
pytest
```

## Dependencies

See `requirements.txt` for the full dependency list, which includes:

- FastAPI - Web framework
- SQLAlchemy - ORM
- PostgreSQL driver (psycopg2)
- NumPy - Numerical computing
- Machine learning libraries (scikit-learn, etc.)

## Contributing

1. Create a feature branch
2. Make your changes
3. Commit with clear messages
4. Push to GitHub
5. Submit a pull request

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
