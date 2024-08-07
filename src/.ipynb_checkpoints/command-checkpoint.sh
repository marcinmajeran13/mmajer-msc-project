mprof run -o mprofile.dat mmajer_test.py
python get_results.py
rm mprofile.dat
python mmajer_test_cpu.py
