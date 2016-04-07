from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import *
def getData():
	train_data= load_files('dataset/train')    
	test_data=load_files("dataset/test")
	count_Vec=TfidfVectorizer(min_df=1,decode_error="replace")
	doc_train=count_Vec.fit_transform(train_data.data)
	doc_test=count_Vec.transform(test_data.data)
	return doc_train.toarray(),train_data.target,doc_test.toarray(),test_data.target

def sigmoid(x):
	return 1.0/(1+exp(-x))
def classify(train_data,train_label,test_data,test_label):
	data_matrix=mat(train_data)
	train_y=mat(train_label).transpose()
	m,n=shape(data_matrix)
	weights=ones((n,1))
	alpha=0.3
	for k in xrange(200):
		output=sigmoid(data_matrix*weights)
		error=train_y-output
		weights=weights+alpha*data_matrix.transpose()*error
	test_matrix=mat(test_data)
	result=sigmoid(test_matrix*weights)
	right_num=0
	num=0;
	for i in xrange(len(result)):
		flag=2
		if result[i]>0.5:
			flag=1
		elif result[i]<0.5:
			flag=0
		if(flag==test_label[i]):
			right_num+=1
		num+=1;
	print right_num,num

def main():
	T_SUM=0;
	SUM=0;
	train_data,train_label,test_data,test_label=getData()
	classify(train_data,train_label,test_data,test_label)
if __name__ == '__main__':
    main();