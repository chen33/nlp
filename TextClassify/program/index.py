from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import *

def getData():
	train_data= load_files('training')    
	test_data=load_files("test")
	count_Vec=TfidfVectorizer(min_df=1,decode_error="replace")
	doc_train=count_Vec.fit_transform(train_data.data)
	doc_test=count_Vec.transform(test_data.data)# ! there is transform not fit_transform
	return doc_train.toarray(),train_data.target,doc_test.toarray()

def classify(train_data,train_label,test_data):
	l=shape(train_data)
	diff_mat=tile(test_data,(l[0],1))-mat(train_data)
	print("diff_mat")
        #exit()
	diff_mat=diff_mat**2
	print("diff_nat_2")
	temp=diff.toarray()
	print(temp[0])



def main():
	T_SUM=0;
	SUM=0;
	train_data,train_label,test_data=getData()
	for i in xrange(len(test_data)):
		target=train_label[i]
		x=test_data[i]
		y=classify(train_data,train_label,x)
		exit()
		if(y==target):
			T_SUM=T_SUM+1;
		SUM=SUM+1;
	print(T_SUM/SUM)




if __name__ == '__main__':
    main();
        



