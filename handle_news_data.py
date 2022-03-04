import json
# from preprocessing import *
from typing import List
import string
import os
import random
import re

label_dict = {'O': 0,'PERIOD':1, 'COMMA':2, 'COLON':3, 'QMARK':4}
PUNCTUATIONS = ['O','PERIOD', 'COMMA', 'QMARK', 'PERIOD', 'COLON','COMMA']
PUNCTUATION_SYM = ['','.',',','?','!',':',';']
# EOS_TOKENS = ['PERIOD','QMARK','EXCLAM']
EOS_TOKENS = ['PERIOD','QMARK']


def word_map(line:str):
    line  = line.split()
    return line[0] + PUNCTUATION_SYM[PUNCTUATIONS.index(line[1])]
def show(path):
    text_ls = []
    seq_length = 100
    with open(path) as f:
        ls = f.read().splitlines()
        print(len(ls))
    for i in range(0,len(ls),seq_length):
        temp_text = ' '.join(list(map(word_map,ls[i:i+seq_length])))
        text_ls.append(temp_text)
        # print(temp_text)
    random.shuffle(text_ls)
    with open('check_text.json','w') as f:
        json.dump(text_ls[:5000],f,ensure_ascii=False) 
# show('dataset/icomm_news_dataset/Cleansed_data/train.txt')
# input()
category_list = ['Chính trị','Đối ngoại','Giáo dục','Pháp luật','Quân sự','Văn hóa','Xã hội']

def random_infer(path):
    with open(path) as f:
        ls = json.load(f)
    with open('check.json','w') as f:
        json.dump(ls[:100],f,ensure_ascii=False)
# random_infer('/home/huydang/project/post_asr_normalize/data/news/Xã hội/content.json')
# print(string.punctuation)

def de_punc(text:str)->List:
    # Example: Chào, tôi là developer. -> ['Chào O', 'tôi O', ....]
    

    words = text.split()
    # print(words)
    result_ls = []
    for ind,word in enumerate(words):
        
        if word[-1] not in PUNCTUATION_SYM[1:]:
            result_ls.append(word + ' O')
        else:
            if (len(word[:-1] ) <= 0):
                # print(word)
                # print(text)
                # print(words[ind+1])
                # print(words[ind-1])
                continue
            punctuation_label = PUNCTUATIONS[PUNCTUATION_SYM.index(word[-1])]
            result_ls.append(word[:-1] + ' ' + punctuation_label)
    return result_ls
# print(de_punc('Tại hội_nghị giao_ban trực_tuyến của UBND thành_phố Hà_Nội diễn ra hôm_nay ( 6/4 ) , Giám_đốc Sở Nội_vụ Vũ_Thu_Hà cho biết các đoàn kiểm_tra công_vụ đã kiểm_tra đột_xuất 34 lượt các cơ_quan , đơn_vị và kiểm_tra xác_minh 2 vụ_việc theo chỉ_đạo của thành_phố .'))
def preprocess(text:str):
    r = text
    try:
        text = text.strip()
        
        text = text.replace('_',' ')
        text = re.sub('\s+',' ',text)
        for punc in PUNCTUATION_SYM[1:]:
            text = text.lstrip(punc)
            text = re.sub(f'\s+\{punc}',f'{punc}',text)
            text = re.sub(f'\{punc}+',f'{punc}',text)
        if text[-1] not in string.punctuation:
            text += '.'
    except:
        print(r)
        print(text)
        return ''
    return text
def get_data():
    train_ratio = 0.7
    dev_ratio = 0.15
    test_ratio = 0.15
    train_origin_text_ls = []
    dev_origin_text_ls  = []
    test_origin_text_ls = []
    for category in category_list:
        with open(f'data/news/{category}/content.json') as f:
            temp_ls = json.load(f)
            temp_ls = temp_ls[:int(len(temp_ls)/3)]
            random.shuffle(temp_ls)
        temp_ls = [i['message'] for i in temp_ls if 'message' in i]
        temp_ls = list(map(preprocess,temp_ls))
        temp_ls = [i for i in temp_ls if i != '']
        with open('show_sen.json','w') as f:
            
            json.dump(temp_ls[:100],f,ensure_ascii=False)
     
        split_index1 = int(len(temp_ls)*train_ratio)
        split_index2 = split_index1 + int(len(temp_ls)*dev_ratio)
        print(split_index1,split_index2,len(temp_ls))
        train_origin_text_ls.extend(temp_ls[:split_index1])
        dev_origin_text_ls.extend(temp_ls[split_index1:split_index2])
        test_origin_text_ls.extend(temp_ls[split_index2:])
    print('train doc:',len(train_origin_text_ls))
    print('test doc:',len(test_origin_text_ls))
    train_set  = []
    test_set  = []
    dev_set  = []
    for doc in train_origin_text_ls:
        train_set.extend(de_punc(doc))
    for doc in dev_origin_text_ls:
        dev_set.extend(de_punc(doc))
    for doc in test_origin_text_ls:
        test_set.extend(de_punc(doc))
  
    with open('dataset/icomm_news_dataset/Cleansed_data/train.txt','w') as f:
        f.write('\n'.join(train_set))
    with open('dataset/icomm_news_dataset/Cleansed_data/test.txt','w') as f:
        f.write('\n'.join(test_set))
    with open('dataset/icomm_news_dataset/Cleansed_data/valid.txt','w') as f:
        f.write('\n'.join(dev_set))

get_data()
# with open('/home/huydang/project/post_asr_normalize/vietnamese-punctuation-prediction/Vietnamese_newspapers/dataset/icomm_news_dataset/Cleansed_data/train.txt') as f:
#     a = f.read().splitlines()
#     print(len(a))
#     print(a[0])