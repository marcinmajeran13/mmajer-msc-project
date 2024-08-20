mprof run -o mprofile.dat HAR_vertex.py
python get_results.py
rm mprofile.dat
python HAR_vertex_cpu_read.py
