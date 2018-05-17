//
//  kgcrl.hpp
//  KGCompletion
//
//  Created by Yang Yunfei on 2018/4/4.
//  Copyright © 2018年 Yang Yunfei. All rights reserved.
//

#ifndef kgcrl_hpp
#define kgcrl_hpp

#include <stdio.h>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <omp.h>
#include "trie_map.hpp"
#include "utilitys.hpp"


using std::string;
using std::vector;
using std::map;

typedef std::pair<int, int> P_INT;
typedef std::pair<int, double> P_DOUBLE;

const char * const ENTITY_FILE = "../data/entity2id.txt";
const char * const RELATION_FILE = "../data/relation2id.txt";
const char * const TRAIN_FILE = "../data/train.txt";
const char * const TEST_FILE = "../data/test.txt";
const char * const VALID_FILE = "../data/valid.txt";

/*************************************************************************
 *  类名称：KGCRL ： Knowledge Graph Completion by Representation Learning
 *  类说明：知识图谱补全功能实现类
 *  创建时间：2018-4-4
 *  最后修改时间：2018-4-16
 *************************************************************************/
class KGCRL
{
public:
    KGCRL()
    :entity_num(0), relation_num(0)
    {
        rel2vec = "relation2vec";
        entity2vec = "entity2vec";
    }
    
    KGCRL(string v, int n)
    :entity_num(0), relation_num(0), dim(n)
    {
        rel2vec = "relation2vec." + v;
        entity2vec = "entity2vec." + v;
    }

    ~KGCRL()
    {}
    
    bool init()
    {
        // 处理实体和id的映射
        FILE * fp = fopen(ENTITY_FILE, "r");
        char buff[1000], left[100], right[100];
        string temp, head, tail;
        int id;
        while (fscanf(fp, "%s %d", buff, &id) == 2)
        {
            temp = buff;
            entity_trie.insert(temp, id);
            id2entity.emplace(id, temp);
            entity_num++;
        }
        fclose(fp);
        
        // 处理关系和id的映射
        fp = fopen(RELATION_FILE, "r");
        while (fscanf(fp, "%s %d", buff, &id) == 2)
        {
            temp = buff;
            relation_trie.insert(temp, id);
            id2relation.emplace(id, temp);
            relation_num++;
        }
        fclose(fp);
        
        // 读入训练文件（h，r，t）三元组
        fp = fopen(TRAIN_FILE, "r");
        while (fscanf(fp, "%s %s %s", left, right, buff) == 3)
        {
            head = left;
            tail = right;
            temp = buff;
            if(!entity_trie.exists(head))
            {
                printf("can not found entity %s\n", left);
            }
            if(!entity_trie.exists(tail))
            {
                printf("can not found entity %s\n", right);
            }
            if(!relation_trie.exists(buff))
            {
                relation_trie.insert(temp, relation_num);
                relation_num++;
            }
            int val1, val2, val3;
            entity_trie.value_at(head, val1);
            entity_trie.value_at(tail, val2);
            relation_trie.value_at(temp, val3);
            left_entity[val3][val1]++;
            right_entity[val3][val2]++;
            append(val1, val3, val2);
        }
        for(int i = 0; i < relation_num; i++)
        {
            double sum1 = 0, sum2 = 0;
            for(auto it = left_entity[i].begin(); it != left_entity[i].end(); it++)
            {
                sum1++;
                sum2 += it->second;
            }
            left_num[i] = sum2 / sum1;
        }
        for(int i = 0; i < relation_num; i++)
        {
            double sum1 = 0, sum2 = 0;
            for(auto it = right_entity[i].begin(); it != right_entity[i].end(); ++it)
            {
                sum1++;
                sum2 += it->second;
            }
            right_num[i] = sum2 / sum1;
        }
//        cout << "relation"
        return true;
    }
    
    //添加三元组
    void append(int h, int r, int t, bool add = true)
    {
        if(add)
        {
            head_vec.push_back(h);
            rel_vec.push_back(r);
            tail_vec.push_back(t);
        }
        P_INT p(h, r);
        tuples[p][t] = 1;
    }
    
    //求和，损失函数
    double cal_sum(int e1, int e2, int rel)
    {
//        printf("e1: %d, e2: %d, rel %d", e1, e2, rel);
        double sum = 0;
        if(L1_flag)
        {
            for (int i = 0; i < dim; ++i)
            {
                sum += fabs(entity_vec[e2][i] - entity_vec[e1][i] - relation_vec[rel][i]);
            }
        }
        else
        {
            for(int i = 0; i < dim; ++i)
            {
                sum += utils.square(entity_vec[e2][i] - entity_vec[e1][i] - relation_vec[rel][i]);
            }
        }
        return sum;
    }
    
    //计算导数
    void gradient(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b)
    {
        for(int i = 0; i < dim; ++i)
        {
            double x = 2 * (entity_vec[e2_a][i] - entity_vec[e1_a][i] - relation_vec[rel_a][i]);
            if(L1_flag)
            {
                x = x > 0 ? 1 : -1;
            }
            relation_tmp[rel_a][i] -= -1 * rate * x;
            entity_tmp[e1_a][i] -= -1 * rate * x;
            entity_tmp[e2_a][i] += -1 * rate * x;
            x = 2 * (entity_vec[e2_b][i] - entity_vec[e1_b][i] - relation_vec[rel_b][i]);
            if(L1_flag)
            {
                x = x > 0 ? 1 : -1;
            }
            relation_tmp[rel_b][i] -= rate * x;
            entity_tmp[e1_b][i] -= rate * x;
            entity_tmp[e2_b][i] += rate * x;
        }
    }
    
    //bfgs 拟牛顿优化算法
    bool bfgs()
    {
        int nbatches = 100, nepoch = 1000;
        size_t batchsize = head_vec.size() / nbatches;
        nthreads = 20;
        //开始并行化执行
#pragma omp parallel for firstprivate(tuples)
        for(int t = 0; t < nthreads; t++)
        {
            time_t start, stop;
            double private_res;
            unsigned int seed = (unsigned) time(NULL) + t;
            for(int epoch = 0; epoch < nepoch/nthreads; epoch++)
            {
                start = time(NULL);
                loss_value = 0;
                private_res = 0;
                for(int batch = 0; batch < nbatches; batch++)
                {
//                    relation_tmp = relation_vec;
//                    entity_tmp = entity_vec;
                    for(int k = 0; k < batchsize; k++)
                    {
                        int i = utils.rand_r_max((int)head_vec.size(), &seed);
                        int j = utils.rand_r_max(entity_num, &seed);
                        double pr = 1000 * right_num[rel_vec[i]] / (right_num[rel_vec[i]] + left_num[rel_vec[i]]);
                        if(std::rand() % 1000 < pr)
                        {
                            while (tuples[std::make_pair(head_vec[i], head_vec[i])].count(j) > 0)
                            {
                                j = utils.rand_r_max(entity_num, &seed);
                            }
                            train_kb(head_vec[i], tail_vec[i], rel_vec[i], head_vec[i], j, rel_vec[i], private_res);
                        }
                        else
                        {
                            while (tuples[std::make_pair(j, head_vec[i])].count(tail_vec[i]) > 0)
                            {
                                j = utils.rand_max(entity_num);
                            }
                            train_kb(head_vec[i], tail_vec[i], rel_vec[i], j, tail_vec[i], rel_vec[i], private_res);
                        }
                        utils.norm(relation_tmp[rel_vec[i]]);
                        utils.norm(entity_tmp[head_vec[i]]);
                        utils.norm(entity_tmp[tail_vec[i]]);
                        utils.norm(entity_tmp[j]);
                    }
//                    relation_vec = relation_tmp;
//                    entity_vec = entity_tmp;
                }
                stop = time(NULL);
                printf("epoch: %d %lf\n", epoch, loss_value);
            }
        }
       
        
        FILE * fp_rel = fopen(rel2vec.c_str(), "w");
        FILE * fp_ent = fopen(entity2vec.c_str(), "w");
        for(int i = 0; i < relation_num; ++i)
        {
            for(int j = 0; j < dim; j++)
            {
                fprintf(fp_rel, "%.6lf\t", relation_vec[i][j]);
            }
            fprintf(fp_rel, "\n");
        }
        for(int i = 0; i < entity_num; ++i)
        {
            for(int j = 0; j < dim; ++j)
            {
                fprintf(fp_ent, "%.6lf\t", entity_vec[i][j]);
            }
            fprintf(fp_ent, "\n");
        }
        fclose(fp_rel);
        fclose(fp_ent);
        return true;
    }
    
    void run_train(double _rate, double _margin, int _method)
    {
        rate = _rate;
        margin = _margin;
        method = _method;
        
        relation_vec.resize(relation_num);
        relation_tmp.resize(relation_num);
        for(int i = 0; i < relation_num; i++)
        {
            relation_vec[i].resize(dim);
            relation_tmp[i].resize(dim);
        }
        entity_vec.resize(entity_num);
        entity_tmp.resize(entity_num);
        for(size_t i = 0; i < entity_vec.size(); i++)
        {
            entity_vec[i].resize(dim);
            entity_tmp[i].resize(dim);
        }
        for(int i = 0; i < relation_num; i++)
        {
            for(int j = 0; j < dim; j++)
                relation_vec[i][j] = utils.randn(0, 1.0/dim, -6/sqrt(dim), 6/sqrt(dim));
        }
        for(int i = 0; i < entity_num; i++)
        {
            for(int j = 0; j < dim; j++)
                entity_vec[i][j] = utils.randn(0, 1.0/dim, -6/sqrt(dim), 6/sqrt(dim));
            utils.norm(entity_vec[i]);
        }
        printf("KGCompletion train running...\n");
        bfgs();
    }
    
    void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b, double &private_res)
    {
        double sum1 = cal_sum(e1_a, e2_a, rel_a);
        double sum2 = cal_sum(e1_b, e2_b, rel_b);
        if(sum1 + margin > sum2)
        {
            private_res += margin + sum1 - sum2;
            gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
        }
    }
    
    //从训练模式切换到测试模式
    void set_test()
    {
        head_vec.clear();
        tail_vec.clear();
        rel_vec.clear();
        tuples.clear();
    }
    
    bool pre_test()
    {
        set_test();
        // 处理实体和id的映射
        FILE * fp = fopen(ENTITY_FILE, "r");
        char rel[500], head[50], tail[50];
        string left, right, relation;
        int id;
        while (fscanf(fp, "%s %d", head, &id) == 2)
        {
            left = head;
            entity_trie.insert(left, id);
            id2entity.emplace(id, left);
            entity_num++;
        }
        fclose(fp);
        
        // 处理关系和id的映射
        fp = fopen(RELATION_FILE, "r");
        while (fscanf(fp, "%s %d", rel, &id) == 2)
        {
            relation = rel;
            relation_trie.insert(relation, id);
            id2relation.emplace(id, relation);
            relation_num++;
        }
        fclose(fp);
        
        //读入测试数据
        fp = fopen(TEST_FILE, "r");
        while (fscanf(fp, "%s %s %s", head, tail, rel) == 3)
        {
            left = head;
            right = tail;
            relation = rel;
            if(!entity_trie.exists(left))
            {
                printf("miss entity : %s\n", head);
            }
            if(!entity_trie.exists(right))
            {
                  printf("miss entity : %s\n", tail);
            }
            if(!relation_trie.exists(relation))
            {
                printf("miss relation : %s\n", rel);
                relation_trie.insert(relation, relation_num);
                relation_num++;
            }
            int val1, val2, val3;
            entity_trie.value_at(left, val1);
            entity_trie.value_at(right, val2);
            relation_trie.value_at(relation, val3);
            append(val1, val3, val2);
        }
        fclose(fp);
        
        // 读入训练数据中的三元组
        fp = fopen(TRAIN_FILE, "r");
        while (fscanf(fp, "%s %s %s", head, tail, rel) == 3)
        {
            left = head;
            right = tail;
            relation = rel;
            if(!entity_trie.exists(left))
            {
                printf("miss entity : %s\n", head);
            }
            if(!entity_trie.exists(right))
            {
                printf("miss entity : %s\n", tail);
            }
            if(!relation_trie.exists(relation))
            {
                relation_trie.insert(relation, relation_num);
                relation_num++;
            }
            int val1, val2, val3;
            entity_trie.value_at(left, val1);
            entity_trie.value_at(right, val2);
            relation_trie.value_at(relation, val3);
            append(val1, val3, val2, false);
        }
        fclose(fp);
        
        // 读入验证数据集
        fp = fopen(VALID_FILE, "r");
        while (fscanf(fp, "%s %s %s", head, tail, rel) == 3)
        {
            left = head;
            right = tail;
            relation = rel;
            if(!entity_trie.exists(left))
            {
                printf("miss entity : %s\n", head);
            }
            if(!entity_trie.exists(right))
            {
                printf("miss entity : %s\n", tail);
            }
            if(!relation_trie.exists(relation))
            {
                relation_trie.insert(relation, relation_num);
                relation_num++;
            }
            int val1, val2, val3;
            entity_trie.value_at(left, val1);
            entity_trie.value_at(right, val2);
            relation_trie.value_at(relation, val3);
            append(val1, val3, val2, false);
        }
        fclose(fp);
        return true;
    }
    
    bool read_vec()
    {
        FILE * fp_r = fopen(rel2vec.c_str(), "r");
        FILE * fp_e = fopen(entity2vec.c_str(), "r");
        if(fp_r == NULL || fp_e == NULL)
        {
            printf("open file error !\n");
            return false;
        }
        printf("relation_num: %d, entity_num: %d\n",relation_num, entity_num);
        relation_vec.resize(relation_num);
        for(int i = 0; i < relation_num; ++i)
        {
            relation_vec[i].resize(dim);
            for(int j = 0; j < dim; ++j)
            {
                fscanf(fp_r, "%lf", &relation_vec[i][j]);
            }
        }
        entity_vec.resize(entity_num);
        for(int i = 0; i < entity_num; ++i)
        {
            entity_vec[i].resize(dim);
            for(int j = 0; j < dim; ++j)
            {
                fscanf(fp_e, "%lf", &entity_vec[i][j]);
            }
            int len = utils.vec_len(entity_vec[i])   ;
            if(len - 1 > 1e-3)
            {
                printf("error entity %d : %d\n", i, len);
            }
        }
        fclose(fp_e);
        fclose(fp_r);
        return true;
    }
    
    void run_test()
    {
        if(read_vec())
        {
            printf("read data ok !!!\n");
        }
        else
        {
            perror("read data error \n");
            return;
        }
        
        printf("KGCompletion test running...\n");
        
        int sum_rank = 0, sum_rank_f = 0;
        int n_hit = 0, n_hit_f = 0;
        
        for(size_t id = 0; id < tail_vec.size(); id++)
        {
            if(id % 100 == 0)
                printf("id:%lu\n", id);
            int h = head_vec[id];
            int t = tail_vec[id];
            int r = rel_vec[id];
            
            double energy = cal_sum(h, t, r);
            int hrank = 0, trank = 0;
            int hrank_f = 0, trank_f = 0;
            for(int i = 0; i < entity_num; ++i)
            {
                // 替换头实体
                if(i != h)
                {
                    double tmp = cal_sum(i, t, r);
                    if(tmp < energy)
                    {
                        hrank ++;
                        if(tuples[P_INT(i, r)].count(t) == 0)
                        {
                            hrank_f ++;
                        }
                    }
                }
                //替换尾实体
                if(i != t)
                {
                    double tmp = cal_sum(h, i, r);
                    if(tmp < energy)
                    {
                        trank ++;
                        if(tuples[P_INT(h,r)].count(i) == 0)
                        {
                            trank_f ++;
                        }
                    }
                }
            }
            if(hrank <= 10) n_hit++;
            if(hrank_f <= 10) n_hit_f++;
            if(hrank <= 10) n_hit++;
            if(hrank_f <= 10) n_hit_f++;
            sum_rank += hrank;
            sum_rank_f += hrank_f;
            sum_rank += trank;
            sum_rank_f += trank_f;
        } 

        double len = (int)tail_vec.size() * 2.0;
        printf("MeanRank(Raw): %lf\tMeanRank(Filter): %lf\n", sum_rank / len, sum_rank_f / len);
        printf("Hit@10(Row): %lf\tHit@10(Filter): %lf\n", n_hit / len, n_hit_f / len);
    }
private:
    
    //实体和实体id的映射
    trie<int> entity_trie;
    //关系和关系id的映射
    trie<int> relation_trie;
    map<int, string> id2entity;
    map<int, string> id2relation;
    map<int, map<int,int> > left_entity, right_entity;
    map<int, double> left_num, right_num;
    map<P_INT, map<int, int> > tuples;
    Utility utils;
    vector<int> head_vec, rel_vec, tail_vec;
    vector<vector<double> > relation_vec, entity_vec;
    vector<vector<double> >relation_tmp, entity_tmp;
    int entity_num;
    int relation_num;
    int dim;
    int method;
    bool L1_flag;
    double margin;
    double rate, belta;
    double loss_value;
    int nthreads;
    string rel2vec, entity2vec;
};
#endif /* kgcrl_hpp */
