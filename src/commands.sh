mprof run -o mprofile.dat HAR_containerized.py
python get_results_containerized.py
rm mprofile.dat
python HAR_containerized_cpu_read.py
