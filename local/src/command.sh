mprof run -o mprofile.dat HAR.py
python3 get_results.py
rm mprofile.dat
python3 HAR_cpu_read.py