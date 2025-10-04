# Materials Discovery

An AI-driven pipeline for discovering new nanoscale materials with tunable band gaps and stable formation energies. This project leverages advanced machine learning techniques, including graph neural networks (GNNs) and active learning, to accelerate materials discovery.

## Overview

The Materials Discovery project is designed to predict material properties and generate new material candidates. It includes the following key components:

- **Data Processing**: Fetches data from the Materials Project API and converts crystal structures into graph representations.
- **Model Training**: Trains a Crystal Graph Convolutional Neural Network (CGCNN) to predict material properties.
- **Generative Modeling**: Uses a Variational Autoencoder (VAE) to generate new material candidates.
- **Uncertainty Estimation**: Employs ensemble methods to estimate prediction uncertainties.
- **Active Learning**: Implements an active learning loop to iteratively improve the model.
- **API Deployment**: Provides a FastAPI-based inference API for real-time predictions.

## Features

- Fetches and preprocesses data from the Materials Project API.
- Converts crystal structures to graph representations.
- Trains a CGCNN baseline GNN for property prediction.
- Implements ensemble uncertainty estimation.
- Includes a VAE-based generative scaffold for material generation.
- Supports an active learning loop for iterative model improvement.
- Deploys a FastAPI inference API for real-time predictions.

## Setup

1. **Install Requirements**:
    ```sh
    pip install -r requirements.txt
    ```
2. **Insert API Key**:
    Add your Materials Project API key in `src/data_loader.py`.
3. **Download Data**:
    ```sh
    python src/data_loader.py
    ```
4. **Train the Model**:
    ```sh
    python src/train.py --epochs 50 --lr 1e-3 --batch_size 32 --cutoff 5.0 --dataset_path data/materials.json
    ```
5. **Run the API Server**:
    ```sh
    uvicorn src.serve:app --host 0.0.0.0 --port 8000
    ```

## Docker

Build and run the Docker container:
```sh
docker build -t materials-discovery .
docker run -p 8000:8000 materials-discovery
```

## Tests

Run the test suite to ensure everything is working correctly:
```sh
pytest tests/
```

## CLI Usage Examples

### Train with Only Volume
```sh
python src/train.py --dataset_path data/materials.json --epochs 50 --lr 1e-3 --batch_size 32 --cutoff 5.0
```

### Multi-Task Training with Masking
```sh
python src/train.py --dataset_path data/materials.json --epochs 50 --lr 1e-3 --batch_size 32 --cutoff 5.0
```

### Check NaN Handling
```sh
pytest tests/test_training.py
```

## Project Structure

- `src/`: Contains the source code for data loading, model training, and API deployment.
- `data/`: Stores the dataset used for training and evaluation.
- `tests/`: Includes unit tests for validating the functionality of the pipeline.
- `Dockerfile`: Defines the Docker image for the project.
- `requirements.txt`: Lists the Python dependencies.

## Future Work

- Enhance the generative model to improve the diversity of generated materials.
- Integrate additional property prediction tasks.
- Optimize the active learning loop for faster convergence.
- Expand the API to include more endpoints for advanced queries.
