import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import threading

from stream_utils import Streaming

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

stream_thread = None
streaming = Streaming()

@app.get("/")
def serve_ui():
    return FileResponse("./static/index.html")

@app.get("/start")
def start_stream(
    source: str = Query("0"),
    fps: int = Query(15),
    blur: int = Query(21),
    background: str = Query("none"),
    preview: bool = Query(False),
    ):
    
    global stream_thread
    
    streaming.update_streaming_config(in_source=source, out_source=None, fps=fps, blur=blur, background=background)
    
    if streaming.running:
        return JSONResponse(content={"message": "Stream already running"}, status_code=400)

    stream_thread = threading.Thread(
        target=streaming.stream_video, args=()
    )
    stream_thread.daemon = True  # Make thread terminate when main program exits
    stream_thread.start()

    return {"message": f"Streaming started from source: {source}, {fps} FPS and blur {blur}."}

@app.get("/stop")
def stop_stream():
    streaming.update_running_status(False)
    return {"message": "Stream stopped successfully"}

@app.get("/devices")
def devices():
    print("Listing available devices")
    return streaming.list_available_devices()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)