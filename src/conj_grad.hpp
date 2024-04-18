#include "../../Dense-CSR/csr_matrix.h"
#include "../../Dense-CSR/vector_operations.h"
#include <cmath>

std::vector<double> conj_grad(const CSR_Matrix &A, std::vector<double> &x0, const std::vector<double> &b, const size_t &N, const double &tolerance) {
    std::vector<double> x = x0;
    std::vector<double> r = A*x - b;
    std::vector<double> d = r;
    double alpha;
    double beta;
    for(size_t i = 0; i < N && !(std::sqrt(r*r) < tolerance); i++){
        alpha = (r * r) / (d * (A * d));
        x = x - (alpha * d);
        std::vector<double> tmp = r;
        r = (A * x) - b;
        // if (r * r == 0){
        //     return x;
        // }
        beta = (r * r) / (tmp * tmp);
        d = r + (beta * d);
        // for (int i = 0; i < 3; i++) {
        //     std::cout << x[i] << ' ';
        // }
        // std::cout << std::endl;
        //std::cout<<r*r<<std::endl;
    }
    return x;
}