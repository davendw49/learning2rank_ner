import pymysql
import ujson
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import torch
np.set_printoptions(threshold = np.inf, suppress = True)

db = pymysql.connect(host="10.10.10.10", port=3306, user="readonly", password="readonly", db="acekg", charset="utf8")

f_pqid = open('data/pqid.txt', 'r', encoding='utf8')
qid = {}
idq = {}
flag = 0
for line in f_pqid:
    c = line[:-1].replace("\"","").split('\t')
    qid[str(c[0]) + "@" + str(c[1])] = flag
    idq[flag] = str(c[0]) + "@" + str(c[1])
    flag += 1

f_benchmark_paper_id = open('data/benchmark_paper.txt', 'r', encoding='utf8')
f_benchmark_res_top3 = open('data/benchmark_result.txt', 'r', encoding='utf8')

bm_paper = []
for line in f_benchmark_paper_id:
    bm_paper.append(line[:-1])

bm_answer = []
for line in f_benchmark_res_top3:
    bm_answer.append(line[:-1])

def get_benchmark(paper_id):
    cur = db.cursor()
    sql='SELECT paper_id, question, abs_score, title_score, qa_score, word_len, letter_len, complex_len, dpaqn_id FROM dde_paper_abstract_QA_ner WHERE paper_id = ' + str(paper_id)
    cur.execute(sql)
    one_paper_bm = []
    try:
        result = cur.fetchall()
        for res in result:
            para = []
            for i in range(2, 8):
                para.append(round(float(res[i]),5))
            one_line_bm = "qid:"+str(qid[str(res[0])+"@"+str(res[1])])+" 1:"+str(round(para[0], 5))+" 2:"+str(round(para[1], 5))+" 3:"+str(round(para[2], 5))+" 4:"+str(round(para[3], 5))+" 5:"+str(round(para[4], 5))+" 6:"+str(round(para[5], 5))
            one_paper_bm.append((one_line_bm, res[8]))
        return one_paper_bm
    except:
        print(paper_id)
        return one_paper_bm