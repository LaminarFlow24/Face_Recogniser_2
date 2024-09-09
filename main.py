from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import pickle
import numpy as np
from celery import Celery
from fastapi.responses import JSONResponse

app = FastAPI()

# Initialize Celery
celery = Celery(
    "tasks",
    broker="redis://localhost:6379/0",  # Redis as broker
    backend="redis://localhost:6379/0"  # Redis as backend
)

# Define model for receiving image data
class ImageData(BaseModel):
    images: list

@app.post("/train")
async def train_model(image_data: ImageData, background_tasks: BackgroundTasks):
    # Start the training task in the background
    job = train_model_task.delay(image_data.images)
    return {"job_id": job.id}

@app.get("/status/{job_id}")
async def check_status(job_id: str):
    job = celery.AsyncResult(job_id)
    
    if job.state == 'SUCCESS':
        # If training is complete, provide download link
        return {"status": "COMPLETED", "download_link": f"/download/{job_id}"}
    else:
        # Return the current job status
        return {"status": job.state}

@app.get("/download/{job_id}")
async def download_model(job_id: str):
    job = celery.AsyncResult(job_id)
    if job.state == 'SUCCESS':
        with open(f'{job_id}.pkl', 'rb') as model_file:
            return JSONResponse(content={"model": pickle.load(model_file)})
    else:
        return {"error": "Model not found or not completed yet."}

# Celery task to handle model training
@celery.task
def train_model_task(image_list):
    images = np.array([np.array(img) for img in image_list])
    
    # Dummy training process (could be replaced with actual ML code)
    labels = np.random.randint(0, 2, len(images))
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    X = np.array([img.flatten() for img in images])  # Flatten images
    y = labels
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple SVM classifier
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    
    # Save the trained model
    model_filename = f'{train_model_task.request.id}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(clf, f)
    
    return "Training completed"
