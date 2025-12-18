*********** Files Included ****************

Script.py – Main pipeline script

input_config.json – Configuration for era paths and time windows

anomalies.json – Detected photometric deviation events

report.pdf – Methodology, results, and visual explanations

****************** How to Run *********************

python Script.py

************* Important Note on Data Paths *********************

The data_path fields in input_config.json have been masked for submission.
To reproduce results, replace each data_path with the local path to the corresponding era folder containing the CSV files.

Example : 

"data_path": "/path/to/era1_csv_folder"

************* Environment *************

Python 3.8+

pandas

numpy

matplotlib

