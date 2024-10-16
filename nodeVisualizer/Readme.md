# Node Visualizer

## Usage
1. Preprocess the video data using `data_process.py`
```
python data_process.py <data_path> <video_id>
```
**NOTE**: This script will copy the frames to a local directory `./img` and create a `graph_data.json` for visualization purpose. They will be flushed every time the script is called.

2. Start a local server by running `start.bat` in Windows, or `start.sh` in Unix-like system. Or you could start any local server in this folder you like.

3. Open http://localhost:8000 if you use the `start` script and enjoy!