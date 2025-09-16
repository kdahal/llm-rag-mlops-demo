#!/bin/bash
pip install -r ../requirements.txt
mlflow server --host 0.0.0.0 --port 5000 &
python mlops_automation.py