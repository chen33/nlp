import os
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def getData(filename):
	result_data=[];
	reader=csv.reader(file(filename,'rb'))
	temp=0;
	for line in reader:
		if(temp!=0):
			result_data.append([line[1],line[2],line[4],line[5],line[6],line[7]])
		temp +=1
	return result_data
def preProcess(data):
	sum_age=0
	sum_p=0
	for item in data:
		if(item[3]!=""):
			sum_age+=float(item[3])
			sum_p+=1
	average_age=(sum_age/sum_p)
	train_x=[]
	train_y=[]
	for item in data:
		age=item[3]
		if(age==""):
			age=average_age
		gender=( 0 if item[2]=="male" else 1)
		train_x.append([int(item[1]),gender,float(age),int(item[4]),int(item[5])])
		train_y.append(int(item[0]))

	#



	return train_x,train_y
def getTestData(filename):
	result_data=[];
	reader=csv.reader(file(filename,'rb'))
	temp=0;
	for line in reader:
		if(temp!=0):
			result_data.append(["-1",line[1],line[3],line[4],line[5],line[6],line[0]])
		temp +=1
	return result_data;
		
def main():
	train_data=getData("train.csv")
	train_x,train_y=preProcess(train_data)
	test_data=getTestData("test.csv")
	ids=[item[6] for item in test_data]
	test_x,temp=preProcess(test_data)
	# regressionFunc=LogisticRegression(C=10,penalty='l2',tol=0.0001)
	# train_sco=regressionFunc.fit(train_x,train_y).score(train_x,train_y)
	# result=regressionFunc.predict(test_x)
	rf=RandomForestClassifier(max_depth=3)
	train_sco=rf.fit(train_x,train_y).score(train_x,train_y)
	result=rf.predict(test_x)
	print(train_sco)


	predictions_file = open("resultlr.csv", "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(ids, result))
	predictions_file.close()

if __name__ == '__main__':
    main();