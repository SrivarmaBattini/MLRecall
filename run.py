import os
import uvicorn

# local use only: enables the simulated demo account
os.environ.setdefault("MLRECALL_DEV", "1")      

if __name__ == "__main__":
    print("MLRecall is starting at http://127.0.0.1:8000  (Ctrl+C to stop)")
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)