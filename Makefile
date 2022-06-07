all:
	g++ -o main -I ./Eigen/ -std=c++11 -O2 main.cpp

run:
	g++ -o main -I ./Eigen/ -std=c++11 -O2 main.cpp && ./main