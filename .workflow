name: CI
 
on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main
 
env:
  TRAIN_PATH: "MLproject/NVDA_Stock_Preprocessing/train_set"
  TEST_PATH: "MLproject/NVDA_Stock_Preprocessing/test_set"
 
jobs:
  build:
    runs-on: ubuntu-latest
 
    steps:
      # Checks-out your repository under $GITHUB_WORKSPACE
      - uses: actions/checkout@v3
 
      # Setup Python 3.12.3
      - name: Set up Python 3.12.3
        uses: actions/setup-python@v4
        with:
          python-version: "3.12.3"
      
      # Check Env Variables
      - name: Check Env
        run: |
          echo $TRAIN_PATH
          echo $TEST_PATH
 
      # Install mlflow
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install mlflow
          pip install tensorflow
      
      # Run as a mlflow project
      - name: Run mlflow project
        run: |
          mlflow run MLproject --env-manager=local 
 
      # Save models to GitHub Repository
      - name: Save mlruns to repo
        run: |
          git config --global user.name ${{ secrets.username }}
          git config --global user.email ${{ secrets.email }}
          git add mlruns/
          git commit -m "Save mlruns from CI run"
          git push origin main

      - name: Upload mlruns
        uses: actions/upload-artifact@v4
        with:
          name: mlruns
          path: mlruns/