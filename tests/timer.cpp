#include "../../grad-SGS/src/grad-SGS.hpp"
#include "../../chebyshev_acceleration/src/chebyshev.hpp"
#include "../../simple_iterative_methods/src/iterations.hpp"
#include "../src/conj_grad.hpp"
#include <fstream>
#include <time.h>
#include <chrono>

int main(){
    std::vector<double> x0 = {1., 2., 3.};
    std::vector<double> v = {100., 2., 40., 4., 60.};
    std::vector<std::size_t> c = {0, 1, 1, 1, 2};
    std::vector<std::size_t> r = {0, 2, 3, 5};
    CSR_Matrix a{v, c, r};
    std::vector<double> b = {1., 1., 1.};
    
    std::ofstream out;
    out.open("simple_time.txt");
    if (out.is_open())
    {
        for (double tolerance = 0.00001; tolerance >= 0.0000001; tolerance *= 0.5){
            std::cout << tolerance << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> res = simple(a, x0, b, 0.015, 1000, tolerance);
            auto end = std::chrono::high_resolution_clock::now();
            out << std::to_string(tolerance) << " "  << std::to_string((end-start).count()) << std::endl; 
        }   
    }
    out.close();

    out.open("jacobi_time.txt");
    if (out.is_open())
    {
        for (double tolerance = 0.00001; tolerance >= 0.0000001; tolerance *= 0.5){
            std::cout << tolerance << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> res = jacobi(a, x0, b, 0.25, 1000, tolerance);
            auto end = std::chrono::high_resolution_clock::now();
            out << std::to_string(tolerance) << " "  << std::to_string((end-start).count()) << std::endl; 
        }   
    }
    out.close();

    out.open("gauss_seidel_time.txt");
    if (out.is_open())
    {
        for (double tolerance = 0.00001; tolerance >= 0.0000001; tolerance *= 0.5){
            std::cout << tolerance << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> res = gauss_seidel(a, x0, b, 0.25, 1000, tolerance);
            auto end = std::chrono::high_resolution_clock::now();
            out << std::to_string(tolerance) << " "  << std::to_string((end-start).count()) << std::endl; 
        }   
    }
    out.close();

    out.open("simple_chebyshev_time.txt");
    if (out.is_open())
    {
        for (double tolerance = 0.00001; tolerance >= 0.0000001; tolerance *= 0.5){
            std::cout << tolerance << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> res = chebyshev(a, x0, b, 10, tolerance, 40, 100);
            auto end = std::chrono::high_resolution_clock::now();
            out << std::to_string(tolerance) << " "  << std::to_string((end-start).count()) << std::endl; 
        }   
    }
    out.close();

    out.open("conj_grad_time.txt");
    if (out.is_open())
    {
        for (double tolerance = 0.00001; tolerance >= 0.0000001; tolerance *= 0.5){
            std::cout << tolerance << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> res = conj_grad(a, x0, b, 1000, tolerance);
            auto end = std::chrono::high_resolution_clock::now();
            out << std::to_string(tolerance) << " "  << std::to_string((end-start).count()) << std::endl; 
        }   
    }
    out.close();

    out.open("sym_gauss_seidel_time.txt");
    if (out.is_open())
    {
        for (double tolerance = 0.00001; tolerance >= 0.0000001; tolerance *= 0.5){
            std::cout << tolerance << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> res = sym_gauss_seidel(a, x0, b, 1000, tolerance);
            auto end = std::chrono::high_resolution_clock::now();
            out << std::to_string(tolerance) << " "  << std::to_string((end-start).count()) << std::endl; 
        }   
    }
    out.close();

    out.open("grad_desc_time.txt");
    if (out.is_open())
    {
        for (double tolerance = 0.00001; tolerance >= 0.0000001; tolerance *= 0.5){
            std::cout << tolerance << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> res = grad_desc(a, x0, b, 1000, tolerance);
            auto end = std::chrono::high_resolution_clock::now();
            out << std::to_string(tolerance) << " "  << std::to_string((end-start).count()) << std::endl; 
        }   
    }
    out.close();
}