from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import asyncio
from inference.predictor import preprocessing, prediction

app = FastAPI()

# Batching config
BATCH_WAIT_TIME = 0.1  # seconds to wait before processing batch
MAX_BATCH_SIZE = 8

# Queue to hold incoming requests
request_queue = asyncio.Queue()


async def batch_processor():
    """Background task that collects and processes batches."""
    while True:
        batch = []
        futures = []

        # Wait for the first request
        frames, future = await request_queue.get()
        batch.append(frames)
        futures.append(future)

        # Wait BATCH_WAIT_TIME to collect more requests
        deadline = asyncio.get_event_loop().time() + BATCH_WAIT_TIME
        while len(batch) < MAX_BATCH_SIZE:
            timeout = deadline - asyncio.get_event_loop().time()
            if timeout <= 0:
                break
            try:
                frames, future = await asyncio.wait_for(request_queue.get(), timeout=timeout)
                batch.append(frames)
                futures.append(future)
            except asyncio.TimeoutError:
                break

        # Process the batch
        try:
            results = prediction(batch)  # pass list of frames
            for future, result in zip(futures, results):
                future.set_result(result)
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_processor())


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        frames = preprocessing(temp_path)

        # Create a future and push to queue
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await request_queue.put((frames, future))

        # Wait for batch processor to return result
        result = await future
        return JSONResponse(content=result)
    finally:
        os.remove(temp_path)
