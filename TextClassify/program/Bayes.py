#coding=gbk
import os
import nltk
from nltk.corpus import stopwords
#贝叶斯分类

def getTokens(text):
    tokens=[];
    stop=stopwords.words("english");
    sents=nltk.sent_tokenize(text)
    for sentence in sents:
        temp=[i for i in sentence.split() if i not in stop]
        tokens.extend(temp)
    return tokens;


VECTOR_MAX_NUMBER=100
VOCABULARY=[];
STAT_TYPE_DOCS_P={};#类别概率


types = os.listdir("../result/");
for t in types:
    content=open("../result/"+t).read()
    tokens=(content.split("|"))

    tokens=tokens[0:VECTOR_MAX_NUMBER];
    for w in tokens:
        if w not in VOCABULARY:
            VOCABULARY.append(w.upper());
#p(v)
sum=0;
for t in types:
    files=os.listdir("../data/training/"+t+"/");
    STAT_TYPE_DOCS_P[t]=len(files);
    sum=sum+len(files);
for t in types:
    STAT_TYPE_DOCS_P[t]=STAT_TYPE_DOCS_P[t]/sum;
#p(w/v)
result={}
for t in types:
    temp={};
    N=0;#不同单词位置的总数
    files=os.listdir("../data/training/"+t+"/");
    text="";
    for f in files:
        fname="../data/training/"+t+"/"+f;
        text=text+open(fname).read();   
    tokens=getTokens(text)
    N_word={}#不同单词出现的次数
    for w in VOCABULARY:
        N_word[w]=0;
    for word in tokens:
        if(word in VOCABULARY):
            N_word[word]=N_word[word]+1;
    for word in VOCABULARY:
        temp[word]=(N_word[word]+1)/(N+len(VOCABULARY))
    result[t]=temp;

#分类
file_type={};
file_list=[];
for t in types:
    files=os.listdir("../data/test/"+t+"/");
    for f in files:
        fname="../data/test/"+t+"/"+f;
        file_type[fname]=t;
        file_list.append(fname);
        
t_num=0;
num=0;        
for fname in file_list:
    print(fname);
    content=open(fname,'r').read()
    tokens=getTokens(content);
    #只保留特征词
    words=[w for w in tokens if w.upper() in VOCABULARY]
    
    max=0;
    max_t='';
    for t in types:
        p=STAT_TYPE_DOCS_P[t];
        for w in words:
            p=p*result[t][w.upper()];
        if p>max:
            max=p;
            max_t=t;
    num=num+1;
    if(max_t==file_type[fname]):
        t_num=t_num+1;
print(t_num);
print(num);




