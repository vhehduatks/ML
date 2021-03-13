# Machine Learning readme
- [Week_2.iris](#2주차iris-데이터셋-기계학습)
- [Week 2.wisconsin](#2주차wisconsin-breast-cancer-기계학습)
---
## (2주차)Iris 데이터셋 기계학습


순서
1.  iris 데이터셋의 구조를 파악
2.  파이썬 사이킷런 라이브러리를 사용하여 기계학습
3.  훈련/테스트데이터 구성 변경 결과 확인하기
4.  knn알고리즘의 n_neighbors=k 수 변경해보기
5.  random_state의 변경

### 1.iris 데이터셋의 구조를 파악
```python
iris_dataset=load_iris()
print("dataset의 키",iris_dataset.keys())
```
![KakaoTalk_20210311_052256465](https://user-images.githubusercontent.com/64114699/110692991-4dc62280-822a-11eb-87ec-58f22037049d.png)
<br><br/>

```python
print(iris_dataset['DESCR'])
```
![KakaoTalk_20210311_052310226](https://user-images.githubusercontent.com/64114699/110693225-8f56cd80-822a-11eb-8ebe-fae846e1e390.png)
<br><br/>

```python
print(type(iris_dataset))
print(type(iris_dataset.data))
print(iris_dataset.data.shape)
print(iris_dataset['target_names'])
print(iris_dataset['feature_names'])
print(iris_dataset['target'])
print(iris_dataset['data'][:5])
```
![KakaoTalk_20210311_052342298](https://user-images.githubusercontent.com/64114699/110693528-e197ee80-822a-11eb-8d38-95d43b140563.png)
<br><br/>

### 2.파이썬 사이킷런 라이브러리를 사용하여 기계학습
```python
x_train,x_test,y_train,y_test=train_test_split(iris_dataset.data,iris_dataset.target,random_state=0)

print('x_tarin의 크기',x_train.shape)
print('y_tarin의 크기',y_train.shape)
print('x_test의 크기',x_test.shape)
print('y_test의 크기',y_test.shape)
```
![KakaoTalk_20210311_164447865](https://user-images.githubusercontent.com/64114699/110752861-294d6300-8289-11eb-9725-751290a14652.png)
<br><br/>

```python
#iris데이터프레임(x_train을 이용한) 만들기
iris_df=pd.DataFrame(x_train,columns=iris_dataset.feature_names)

#넘파이로 데이터의 산전도행렬 표현
pd.plotting.scatter_matrix(iris_df,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=0.8,cmap=mglearn.cm3)
plt.show()
```
![KakaoTalk_20210311_052505906](https://user-images.githubusercontent.com/64114699/110752020-066e7f00-8288-11eb-80f5-0a3cfd0eb53f.png)
<br><br/>

```python
#가장 가까운 1개의 이웃의 영향을 받아 분류하도록 함
knn=KNeighborsClassifier(n_neighbors=1)

#x데이터와 y라벨로 fit 시켜준다=학습
knn.fit(x_train,y_train)

#새로운 데이터를 예측하기
unknown_data=[[5,2.9,1,0.2]]
guesses=knn.predict(unknown_data)
print('predict:',guesses)
print('predict name:',iris_dataset['target_names'][guesses])

#테스트 데이터의 정확도 평가하기
print('test_x accuracy :',knn.score(x_test,y_test))
```
![KakaoTalk_20210311_164413523](https://user-images.githubusercontent.com/64114699/110752942-3ff3ba00-8289-11eb-83d5-89ef8d9c07b7.png)
<br><br/>

### 4.knn알고리즘의 n_neighbors=k 수 변경해보기
 ```python
 #k값의 변동에 따른 모델의 정확도 시각화
k_range=range(1,101)
accuracy=list()

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    accuracy.append(knn.score(x_test,y_test))

plt.plot(k_range,accuracy)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
```
![KakaoTalk_20210311_052416132](https://user-images.githubusercontent.com/64114699/110753672-33bc2c80-828a-11eb-8d1b-d3c1787418dd.png)
<br><br/>

### 5.random_state의 변경
```python
#데이터셋 분할(훈련,테스트)
# x_train,x_test,y_train,y_test=train_test_split(iris_dataset.data,iris_dataset.target,random_state=0)
x_train,x_test,y_train,y_test=train_test_split(iris_dataset.data,iris_dataset.target,test_size=0.33,random_state=42)

print('x_tarin의 크기',x_train.shape)
print('y_tarin의 크기',y_train.shape)
print('x_test의 크기',x_test.shape)
print('y_test의 크기',y_test.shape)
```
before
![KakaoTalk_20210311_164447865](https://user-images.githubusercontent.com/64114699/110752861-294d6300-8289-11eb-9725-751290a14652.png)
after
![KakaoTalk_20210311_165633236](https://user-images.githubusercontent.com/64114699/110754125-ca88e900-828a-11eb-8723-fe617d5beca6.png)
<br><br/>
```python
#iris데이터프레임(x_train을 이용한) 만들기
iris_df=pd.DataFrame(x_train,columns=iris_dataset.feature_names)

#넘파이로 데이터의 산전도행렬 표현
pd.plotting.scatter_matrix(iris_df,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=0.8,cmap=mglearn.cm3)
plt.show()
```
before
![KakaoTalk_20210311_052505906](https://user-images.githubusercontent.com/64114699/110752020-066e7f00-8288-11eb-80f5-0a3cfd0eb53f.png)
after
![KakaoTalk_20210311_165033245](https://user-images.githubusercontent.com/64114699/110756009-fb6a1d80-828c-11eb-87b0-e453ac7c8182.png)
<br><br/>

```python
print('predict:',guesses)
print('predict name:',iris_dataset['target_names'][guesses])

#테스트 데이터의 정확도 평가하기
print('test_x accuracy :',knn.score(x_test,y_test))
```
before
![KakaoTalk_20210311_164413523](https://user-images.githubusercontent.com/64114699/110752942-3ff3ba00-8289-11eb-83d5-89ef8d9c07b7.png)
after
![KakaoTalk_20210311_171035816](https://user-images.githubusercontent.com/64114699/110755822-bb0a9f80-828c-11eb-8d95-65e6ab329039.png)
<br><br/>

 ```python
 #k값의 변동에 따른 모델의 정확도 시각화
k_range=range(1,101)
accuracy=list()

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    accuracy.append(knn.score(x_test,y_test))

plt.plot(k_range,accuracy)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
```

before

![KakaoTalk_20210311_052416132](https://user-images.githubusercontent.com/64114699/110753672-33bc2c80-828a-11eb-8d1b-d3c1787418dd.png)

after

![KakaoTalk_20210311_165043171](https://user-images.githubusercontent.com/64114699/110756125-248aae00-828d-11eb-96e7-40102f471400.png)
<br><br/>
<br><br/>

---
## (2주차)Wisconsin Breast Cancer 기계학습

순서
1.  make_merge()를 사용한 데이터 생성 및 기계학습
2.  make_wave()를 사용한 데이터 생성 및 도표생성
3.  Wisconsin Breast Cancer Data를 사용한 기계학습 

### 1.make_merge()를 사용한 데이터 생성 및 기계학습

```python
#임의의 점 데이터를 생성하는 make_forge()
data,label=mglearn.datasets.make_forge()

#train, test 데이터를 나눔
data_train,data_test,label_train,label_test=train_test_split(data,label,random_state=0)

#data모양을 확인
print(data[:5])
data_frame=pd.DataFrame(data,label)
print(type(data))
print(data_frame.head())
print(label[0:5])
```
![KakaoTalk_20210313_205211444](https://user-images.githubusercontent.com/64114699/111029115-36f81980-843e-11eb-9d58-7626735dd930.png)
<br><br/>
```python
#데이터프레임화 하여 산점도 표현
data_class_0=data_frame.loc[0]
data_class_1=data_frame.loc[1]
label_c=['red' if y==1 else 'blue' for y in label]
class_0=plt.scatter(data_class_0[0],data_class_0[1],c='red',marker='o',alpha=0.5)
class_1=plt.scatter(data_class_1[0],data_class_1[1],c='blue',marker='o',alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(handles=(class_0,class_1),labels=('class0','class1'),loc='best')
plt.show()
plt.clf()
```
![KakaoTalk_20210313_205440968](https://user-images.githubusercontent.com/64114699/111029147-5d1db980-843e-11eb-87fc-a89f1e022d3f.png)
<br><br/>

```python
# knn으로 분류 후 정확도 확인
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(data_train,label_train)
# print(data[:5]) #float형 임의의 데이터
# print(label[:5]) #1,0으로 된 라벨
print('predict ans:',label_test)
print('predict test:',knn.predict(data_test))
print('predict score:',knn.score(data_test,label_test))
```
![KakaoTalk_20210313_205649135](https://user-images.githubusercontent.com/64114699/111029198-aa9a2680-843e-11eb-9f84-af601f7bdbbe.png)
<br><br/>
### 2.make_wave()를 사용한 데이터 생성 및 도표생성
```python
#임의의 점 데이터를 생성하는 make_wave()
data_x,data_y=mglearn.datasets.make_wave(n_samples=40)

# 데이터의 분포를 산점도로 확인
# data_frame=pd.DataFrame(data_x)
# print(data_frame.shape)
plt.scatter(data_x,data_y,marker='o',alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
![KakaoTalk_20210313_205600095](https://user-images.githubusercontent.com/64114699/111029660-2ac18b80-8441-11eb-8138-719b9f1b7b17.png)
<br><br/>

### 3.Wisconsin Breast Cancer Data를 사용한 기계학습 

```python
#Wisconsin Breast Cancer Data
cancer=load_breast_cancer()

C_data=cancer.data
C_target=cancer.target

# #데이터의 모양을 확인 569x30
print(C_data.shape,C_target.shape)

#데이터프레임화
cancer_df=pd.DataFrame(C_data,C_target,columns=cancer.feature_names)
cancer_df_0=cancer_df.loc[0]
cancer_df_1=cancer_df.loc[1]

#인덱스 초기화
cancer_df_0=cancer_df_0.reset_index(drop=True)
cancer_df_1=cancer_df_1.reset_index(drop=True)

#행열 바꾸기
cancer_df_0=cancer_df_0.transpose()
cancer_df_1=cancer_df_1.transpose()

#데이터의 분포를 plot으로 확인
# print(cancer_df_0)
# print(cancer_df_1.shape)
plot_0=plt.plot(cancer_df_0,c='blue',alpha=0.2)
plot_1=plt.plot(cancer_df_1,c='red',alpha=0.2)
plt.show()
```
![KakaoTalk_20210313_212624626](https://user-images.githubusercontent.com/64114699/111029935-dd461e00-8442-11eb-88cd-b76617c01962.png)
<br><br/>

```python
#train,test 로 나눔
c_data_train,c_data_test,c_target_train,c_target_test=train_test_split(C_data,C_target,random_state=1)

#knn fit
knn_c=KNeighborsClassifier(n_neighbors=3)
knn_c.fit(c_data_train,c_target_train)

#accuracy(k=3)
print('accuracy:',knn_c.score(c_data_test,c_target_test))
```
![KakaoTalk_20210313_212807873](https://user-images.githubusercontent.com/64114699/111029964-0f578000-8443-11eb-82e8-31b2f30fbcd7.png)
<br><br/>
```python
#신뢰도가 높은 k값 찾기
k_range=range(1,101)
accuracy_list_train=list()
accuracy_list_test=list()
for k in k_range:
    knn_c=KNeighborsClassifier(n_neighbors=k)
    knn_c.fit(c_data_train,c_target_train)
    accuracy_list_train.append(knn_c.score(c_data_train,c_target_train))
    accuracy_list_test.append(knn_c.score(c_data_test,c_target_test))

c_test_plot,=plt.plot(k_range,accuracy_list_test,c='red')    
c_train_plot,=plt.plot(k_range,accuracy_list_train,c='blue')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.legend(handles=(c_test_plot,c_train_plot),labels=('test_ac','train_ac'),loc='best')
plt.show()
```
![KakaoTalk_20210313_212906363](https://user-images.githubusercontent.com/64114699/111029983-2c8c4e80-8443-11eb-9211-0f139db8fc11.png)
<br><br/>
2주차 총평: 내가 기계학습을 공부하는건지 pandas를 공부하는건지
