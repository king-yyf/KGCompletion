//
//  main.cpp
//  KGCompletion
//
//  Created by Yang Yunfei on 2018/4/4.
//  Copyright © 2018年 Yang Yunfei. All rights reserved.
//

#include <iostream>
#include <string>
#include <cstring>
#include "kgcrl.hpp"

void help()
{
	printf("error argument\n");
        printf("-------------help-------------\n");
        printf("Usage:  1: train: ./kgc train\n");
        printf("        2: test: ./kgc test\n");

}

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
    if(argc < 2)
    {
//	printf("not enough argument\n");
//	printf("-------------help-------------\n");
//	printf("Usage:  1: train: ./kgc train\n");
//	printf("        2: test: ./kgc test\n");
	help();
	return 0;
    }	
    else{
	printf("%s, %s\n", argv[0], argv[1]);
	if(strcmp(argv[1], "train") == 0)
	{
	    pkgc->init();   
	    pkgc->run_train(dim, rate, margin);
	}
	else if(strcmp(argv[1], "test") == 0)
	{
	    pkgc->pre_test();
	    pkgc->run_test();
	}
	else
	{
	    help();
	}
	
    }
    return 0;
}
