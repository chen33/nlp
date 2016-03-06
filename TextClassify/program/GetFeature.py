#coding=gbk
import os
import nltk

#description：特征提取

def getMaxK(arr,k):
    arr_k=[];
    for i in range(k):
        max=0;
        max_i=[];
        for item in arr:
            if(item[1]>max):
                max=item[1];
                max_i=item;
        arr_k.append(max_i);
        arr.remove(max_i);
    return arr_k;


def delStopwords(words):
    stopwords=open("../data/stopwords.txt").read().split('\n');
    new_token=[];
    for w in words:
        if not(w.lower() in stopwords):
            new_token.append(w.lower());
    words=new_token;
    return words;


VOCABULARY=[];
types = os.listdir("../data/training/");
typesnum={};
vec_allfiles={};
files_sum=0;
for t in types:
    files=os.listdir("../data/training/"+t);
    files_sum=files_sum+len(files);
    typesnum[t]=len(files);
    for f in files:
        temp=[];
        vec={}
        content=open("../data/training/"+t+"/"+f).read();
        sents=nltk.sent_tokenize(content)
        for sentence in sents:
            tokens=nltk.word_tokenize(sentence)
            tokens_delsw=delStopwords(tokens);
            for w in tokens_delsw:
                if(w not in VOCABULARY):
                    VOCABULARY.append(w)
                if(w not in temp):
                    temp.append(w);
                    vec[w]=1;
                else:
                    vec[w]=vec[w]+1;
        tempp={"vector":vec,"t":t};
        vec_allfiles[f]=tempp;

#计算词汇表的每个单词与各个类别的卡方值
final={};
for t in types:
    #该类别的文档数目
    record=[];
    type_num=typesnum[t];
    for w in VOCABULARY:
        YY=0;#既包含这个单词又属于该类别的文档
        YN=0;
        NN=0;
        NY=0;
        result=0;
        #包含这个单词的文档数
        doc_num=0;
        for k in vec_allfiles:
            flag=0;
            for item in vec_allfiles[k]['vector']:
                if(item==w):
                    doc_num=doc_num+1;
                    flag=1;
                    break;
            if(flag==1 and vec_allfiles[k]['t']==t):
                YY=YY+1;
        YN=doc_num-YY;
        NY=type_num-YY;
        NN=files_sum-doc_num-type_num+YY;
        #print('------词汇：',w,' 类别：',t,' NY=',NY,' NN=',NN);
        v=(YY*NN-YN*NY)*(YY*NN-YN*NY)
        if (YY+YN)*(NY+NN)==0:
            v=0;
        else:
            v=v/((YY+YN)*(NY+NN))
        record.append([w,v])
    maxk=getMaxK(record,len(record)-1);
    final[t]=maxk
    content=""
    for item in maxk:
        content=content+item[0]+"|"
    file_object = open('../result/'+t, 'w');
    file_object.write(content);
    file_object.close( ) 

        

                    
        
        
    
    

