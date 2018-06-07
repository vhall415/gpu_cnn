#include "cnpy.cpp"
#include <iostream>
#include <complex>
#include <cstdlib>
#include <map>
#include <string>
#include <regex>

int main() {
    try{
	cnpy::NpyArray arr = cnpy::npy_load("var7.npy");
    }
    catch(std::regex_error& e) {
        std::cout << e.what() << std::endl;
    }
        //std::complex<double>* loaded_data = arr.data<std::complex<double>>();
    //std::cout << loaded_data[0];
    return 0;
}
