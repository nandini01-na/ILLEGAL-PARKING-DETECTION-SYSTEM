# AI-Based Illegal Parking Detection System — Prototype

A practical Engineering Clinics demo based on the submitted project methodology:
Camera/Video → Image Processing → Vehicle Detection → Restricted-Zone Analysis → Violation Detection → Alert/Record.

## 1. Install
Use Python 3.10 or 3.11 if possible.

```bash
python -m venv venv
venv\\Scripts\\activate        # Windows
pip install -r requirements.txt
```

The first run may download the small `yolov8n.pt` model from Ultralytics. Internet is required only for that first model download if the weights are not already present.

## 2. Run
```bash
python app.py
```
Open the address shown in the terminal (normally http://127.0.0.1:5000).

## 3. Demo
- Click **Start Webcam**, or upload a road/parking video.
- The red rectangle is the predefined NO-PARKING ZONE.
- YOLO detects cars, motorcycles, buses and trucks.
- When a tracked vehicle remains inside the zone for 3 seconds, the box becomes red and the incident is logged in `logs/violations.db`.
- The dashboard shows current vehicles and violation records.

## Important prototype limitation
This MVP uses zone occupancy + persistence time as a practical proxy for parking. It does not identify a license plate or issue fines. Those are outside the initial project scope.

## MySQL
The college synopsis lists MySQL as the target database. This prototype uses SQLite so the team can run the inspection demo without configuring a database server. The database layer can be replaced with MySQL after the core demo is accepted.
