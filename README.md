# Motion Analysis in Neuromorphic Cameras

The code implementations of the system discussed in our project report, covering pose estimation, action classifiaction and privacy preservation in neuromorphic cameras.

![event_gif](./event_animation.gif)

## Modules

Each module mentioned in the report is contained within a standalone folder:
- action_classifier: Code and models related to pose estimation, action classifiactio and skeletal structure visualization.
- app: Web interface that simplifies the usage of select modules (e.g. V2E for custom event-stream video creation and face-detection and blurring).
- face_detection: Code used to conduct the face detection and shuffling experiment (Split from the Facial_Detection_and_Blurring_Events.ipynb file into logical subsections in .py format).  



## Requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod +x ./install_st-gcn.sh
chmod +x ./install_v2e.sh
./install_st-gcn.sh
./install_v2e.sh
```


## Usage

For the web interface:
```bash
python app/app.py
```

For action classification:
```bash
classify_action.py
```

For face detection you can load the python notebook file into [google colab](https://colab.research.google.com/) and use the T4 GPU to run each cell.

