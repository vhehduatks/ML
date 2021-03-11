# (2주차)Iris 품종분류 기계학습


순서
1.  iris 데이터셋의 구조를 파악
2.  파이썬 사이킷런 라이브러리를 사용하여 기계학습
3.  훈련/테스트데이터 구성 변경 결과 확인하기
4.  knn알고리즘의 n_neighbors=k 수 변경해보기
5.  random_state의 변경
---
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

