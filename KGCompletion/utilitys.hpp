//
//  utilitys.hpp
//  KGCompletion
//
//  Created by Yang Yunfei on 2018/4/4.
//  Copyright © 2018年 Yang Yunfei. All rights reserved.
//

#ifndef utilitys_hpp
#define utilitys_hpp

#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>
#define PI 3.1415926535897932384626433832795

using std::vector;

/*************************************************************************
 *  类名称：Utility ：模型工具类
 *  类说明：知识图谱表示学习模型使用到的一些工具函数
 *  创建时间：2018-4-4
 *  修改时间：2018-4-4
 *************************************************************************/

class Utility
{
public:
    //sigmod激活函数
    double sigmod(double x)
    {
        return 1.0 / (1 + exp(-x));
    }
    
    //vector输出,用于测试
    void output_vec(vector<double> vec)
    {
        for(size_t i = 0; i < vec.size(); ++i)
        {
            printf("%lf\t", vec[i]);
            if(i % 10 == 9)
            {
                printf("\n");
            }
        }
    }
    
    //平方
    double square(double var)
    {
        double res = var * var;
        return res;
    }
    
    //随机生成[dMin, dMax]之间的实数
    double rand(double dMin, double dMax)
    {
        double dRand = std::rand() / (RAND_MAX + 1.0);
        return (dMax - dMin) * dRand + dMin;
    }
    
    //返回一个以miu为均值，sigma为方差的高斯分布（Gaussian distribution），取值为x时的y值
    double Gaussian(double x, double miu, double sigma)
    {
        double expon = -1 * square(x - miu) / (2 * square(sigma));
        double coef = 1.0 / (sqrt(2 * PI) * sigma);
        return coef * exp(expon);
    }
    
    double randn(double miu, double sigma, double dMin, double dMax)
    {
        double x, y, yMax = Gaussian(miu, miu, sigma), randy;
        do{
            x = rand(dMin, dMax);
            y = Gaussian(x, miu, sigma);
            randy = rand(0.0, yMax);
        }while (randy > y);
        return x;
    }
    
    //求向量的模，用于正则化
    double vec_len(vector<double> & vec)
    {
        double res = 0;
        for(size_t i = 0; i < vec.size(); ++i)
            res += square(vec[i]);
        return sqrt(res);
    }
    
    //向量的正则化
    void norm(vector<double> & vec)
    {
        double len = vec_len(vec);
        if(len > 1)
        {
            for(size_t i = 0; i < vec.size(); ++i)
            {
                vec[i] /= len;
            }
        }
    }
    
    //大于0，小于n的随机整数
    int rand_max(int n)
    {
        int res = (std::rand() * std::rand()) % n;
        if (res < 0)
        {
            res += n;
        }
        return res;
    }
    
    // 并行状态下的随机取余
    int rand_r_max(int n, unsigned int * seed)
    {
        int res = (rand_r(seed)*rand_r(seed)) % n;
        if(res < 0)
        {
            res += n;
        }
        return res;
    }
    
};
#endif /* utilitys_hpp */
