import os
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy import *
def getData(filename):
	result_data=[];
	reader=csv.reader(file(filename,'rb'))
	temp=0;
	for line in reader:
		if(temp!=0):
			result_data.append([int(line[1]),line[2],line[4],(line[5]),(line[6]),line[7],line[0]])
		temp +=1
	return result_data
def getAgeLevel(age):
	if age<=18 or age>=70:
		return 1;
	elif(age>10 and age<=18 )or(age>=50 and age<70):
		return 2
	else:
		return 3
def getGender(a):
	return (0 if a=="male" else 1 )
def SorDataByItem(data,pos):
	n=len(data)
	for i in xrange(n):
		for j in xrange(1,n-i):
			if(int(data[j-1][pos])>int(data[j][pos])):
				temp=data[j-1]
				data[j-1]=data[j]
				data[j]=temp
	result=[]
	for item in data:
		result.append(item[0:len(item)-1])
	return result;
def preProcess(data):
	train_y=[]
	for item in data:
		train_y.append(int(item[0]))
	temp_train_x=[]
	temp_train_y=[]
	temp_test_x=[]
	temp_x=[];
	temp_test=[]
	for item in data:
		if(item[3]!=""):
			age=getAgeLevel(float(item[3]))
			temp_train_x.append([int (item[1]),getGender(item[2]),int(item[4]),int(item[5])])
			temp_train_y.append(age)
			temp_x.append([int (item[1]),getGender(item[2]),age,int(item[4]),int(item[5]),item[6]])
		else:
			temp_test_x.append([int (item[1]),getGender(item[2]),int(item[4]),int(item[5])])
			temp_test.append([int (item[1]),getGender(item[2]),int(item[4]),int(item[5]),item[6]])
	dtc=DecisionTreeClassifier()
	dtc.fit(temp_train_x,temp_train_y)
	result=dtc.predict(temp_test_x)
	for i in xrange(len(result)):
		item=temp_test[i]
		age=result[i]
		temp_x.append([int (item[0]),item[1],age,item[2],int(item[3]),int(item[4])])
	train_x=SorDataByItem(temp_x,5);
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


def classifyByGender(train_x,train_y,test_data):
	data_male_train=[];
	data_male_train_y=[]
	data_female_train=[];
	data_female_train_y=[]
	for i in xrange(len(train_x)):
		item=train_x[i]
		itemy=train_y[i]
		if(item[1]==0):
			data_male_train.append([item[0],item[2],item[3],item[4]])
			data_male_train_y.append(itemy)
		else:
			data_female_train.append([item[0],item[2],item[3],item[4]])
			data_female_train_y.append(itemy)
	ids_male=[]
	ids_female=[]
	regressionFunc=LogisticRegression(C=10,penalty='l2',tol=0.0001)
	train_sco=regressionFunc.fit(data_male_train,data_male_train_y).score(data_male_train,data_male_train_y)
	test_male=[];
	for i in xrange(len(test_data)):
		item=test_data[i]
		if(item[1]==0):
			test_male.append([item[0],item[2],item[3],item[4]])
			ids_male.append(i)
	result_male=regressionFunc.predict(test_male)

	regressionFunc2=LogisticRegression(C=10,penalty='l2',tol=0.0001)
	train_sco=regressionFunc2.fit(data_female_train,data_female_train_y).score(data_female_train,data_female_train_y)
	test_female=[];
	for i in xrange(len(test_data)):
		item=test_data[i]
		if(item[1]==1):
			test_female.append([item[0],item[2],item[3],item[4]])
			ids_female.append(i)
	result_female=regressionFunc2.predict(test_female)
	#combile
	ids=ids_male+ids_female
	result_all=result_male.tolist()+result_female.tolist()

	result=[0 for item in ids]
	
	for i in xrange(len(ids)):
		pid=ids[i]
		result[pid]=result_all[i]
	return result;
def main():
	train_data=getData("train.csv")
	train_x,train_y=preProcess(train_data)
	test_data=getTestData("test.csv")
	ids=[item[6] for item in test_data]
	test_x,temp=preProcess(test_data)
	regressionFunc=LogisticRegression(C=10,penalty='l2',tol=0.0001)
	train_sco=regressionFunc.fit(train_x,train_y).score(train_x,train_y)
	result=regressionFunc.predict(test_x)
	#diffirent gender have diffirent model
	# result=classifyByGender(train_x,train_y,test_x)
	# rf=RandomForestClassifier(max_depth=3)
	# train_sco=rf.fit(train_x,train_y).score(train_x,train_y)
	# result=rf.predict(test_x)
	predictions_file = open("result.csv", "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(ids, result))
	predictions_file.close()

if __name__ == '__main__':
    main();