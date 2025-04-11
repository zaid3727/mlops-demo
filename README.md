# MLOps Demo: Train, Serve, and Test an ML Model

![CI](https://github.com/zaid3727/mlops-demo/actions/workflows/main.yml/badge.svg)

This project demonstrates a complete MLOps workflow using:

- Scikit-learn for model training (Iris dataset)
- MLflow for experiment tracking
- FastAPI for serving the model via a REST API
- Pytest for API testing
- GitHub Actions for CI pipeline

---

## How to Use

### Step 1: Train the model

```bash
python train.py
```

This will:
- Train a Random Forest classifier
- Save the model to `iris_model.pkl`
- Log metrics and artifacts to MLflow

---

### Step 2: Start the API

```bash
uvicorn serve:app --reload
```

Now visit: `http://127.0.0.1:8000/docs` to test the `/predict` endpoint.

Example request:

```json
POST /predict
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

---

### Step 3: Run the tests

```bash
pytest test_api.py
```

---

## CI/CD

- On every push to `master`, GitHub Actions runs the test suite automatically.

---

## Author

Built by [@zaid3727](https://github.com/zaid3727)
