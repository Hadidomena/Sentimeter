@echo off
call venv\Scripts\activate.bat
start "" py backend\sentiment.py
start "" go run backend\main.go
