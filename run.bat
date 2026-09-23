@echo off
REM Activate the environment
cd .\anpr\Scripts\
.\activate

REM Navigate back to the project directory
cd ../../

REM Run main.py
python main.py

REM Run add_missing_data.py
python add_missing_data.py

REM Run visualize.py
python visualize.py
