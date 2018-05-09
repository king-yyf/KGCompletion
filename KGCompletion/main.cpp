//
//  main.cpp
//  KGCompletion
//
//  Created by Yang Yunfei on 2018/4/4.
//  Copyright © 2018年 Yang Yunfei. All rights reserved.
//

#include <iostream>
#include <string>
#include "kgcrl.hpp"

int main(int argc, const char * argv[]) {
    srand((unsigned) time(NULL));
    int method = 1;
    int dim = 100;
    double rate = 0.001;
    double margin = 1;
    std::string version;
    version = method ? "bern" : "unif";
    KGCRL *pkgc = new KGCRL(version,dim);
//    pkgc->init();
//    pkgc->run_train(dim, rate, margin, method);
    
    pkgc->pre_test();
    pkgc->run_test();
    return 0;
}
