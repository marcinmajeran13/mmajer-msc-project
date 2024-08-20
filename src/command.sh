mprof run -o mprofile.dat HAR_cloud.py
python get_results.py
rm mprofile.dat
python HAR_cloud_cpu_read.py
