"""
One-command launcher for the Rotational Dual-Image 3D Disc Studio.
Starts the FastAPI/Uvicorn server and prints the local URL.
"""

import uvicorn


if __name__ == "__main__":
    print("=" * 62)
    print("  Rotational Dual-Image 3D Disc Studio")
    print("  Open in your browser: http://127.0.0.1:8700")
    print("=" * 62)
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8700, reload=False)
