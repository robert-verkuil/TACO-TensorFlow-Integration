#include <iostream>
#include "include/taco.h"

using namespace taco;

int main(int argc, char* argv[]) {
    Format csr({Dense,Sparse});
    Tensor<double> B({2,2}, csr);
    B.insert({0,0}, 1.0);
    B.pack();
    std::cout << B << std::endl;
    return 0;
}
