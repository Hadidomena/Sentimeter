@echo off
call venv\Scripts\activate.bat
start "" py backend\sentiment.py
go run backend\main.go
