#include "../src/conj_grad.hpp"

int main(){
    std::vector<double> x0 = {1., 2., 3.};
    std::vector<double> v = {100., 2., 40., 4., 60.};
    std::vector<std::size_t> c = {0, 1, 1, 1, 2};
    std::vector<std::size_t> r = {0, 2, 3, 5};
    CSR_Matrix a{v, c, r};
    std::vector<double> b = {1., 1., 1.};
    //std::vector<double> res = simple(a, x0, b, 0.25, 100, 0.001);
    //std::vector<double> res = jacobi(a, x0, b, 0.25, 100, 0.001);
    std::vector<double> res = conj_grad(a, x0, b, 100, 0.01);
    std::cout<<res[0]<<" "<<res[1]<<" "<<res[2];
}