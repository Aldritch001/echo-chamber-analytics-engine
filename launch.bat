@echo off
cd /d "C:\Users\User\Desktop\echo_chamber_project"
"C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run app.py --browser.gatherUsageStats=False
pause