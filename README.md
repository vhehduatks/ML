# Machine Learning readme
- [Week_2.iris](#2주차iris-데이터셋-기계학습)
- [Week 2.wisconsin](#2주차wisconsin-breast-cancer-기계학습)
- [Week 3.Regression](#3주차knn회귀분석선형회귀분석)
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

---
## (3주차)KNN회귀분석,선형회귀분석
이전처럼 readme를 작성하는건 py파일을 올렸을경우에 어디서 어디까지 코드가 실행된건지 모르는 상황이 자주 발생하므로 유용했지만,
ipynb파일을 git에 업로드 하면 해당 문제점이 해결되는 부분이므로 굳이 코드를 찍어 올리는 고생을 할 필요가 없어졌다.
따라서 3주차부터는 코드를 작성하면서 막혔던 부분만 리뷰하는 형식으로 진행하려고 한다.

```python
#훈련용,테스트용 데이터셋으로 분류
Input_train,Input_test,Output_train,Output_test=train_test_split(Input,Output,random_state=0)

#K의 수를 3으로 하고 기계학습
knn_R=KNeighborsRegressor(n_neighbors=3)
knn_R.fit(Input_train,Output_train)

#테스트셋 예측 및 도표화
print('test_1 Output:',Output_test)
Predict=knn_R.predict(Input_test)
print('test_1 predict:',Predict)
print('test_1 score:',knn_R.score(Input_test,Output_test))
zero_y=[-3]*len(Input_test) #테스트셋의 인풋을 표현하기 위해 만든 y label
data_p=plt.scatter(Input,Output,marker='o',alpha=0.5)
predict_p=plt.scatter(Input_test,Predict,c='red',marker="*",alpha=0.5)
input_p=plt.scatter(Input_test,zero_y,c='green',marker="*",alpha=0.5)
plt.legend(handles=(data_p,predict_p,input_p),labels=('data','predict','test input'))
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()
```
zero_y라는 리스트는 np.linspace(시작,끝,갯수)를 사용하면 더 간편하게 구현이 가능
plt.scatter함수에서 c='red'라고 할 필요없이 c='r'로 간편하게 구현이 가능
plt.plot함수에는 plot의 색,마커의 모양,선의 모양을 한번에 설정할 수 있는데 scatter에는 해당 기능이 없는것같다.
<br><br/>

```python
def make_kn_p(K,N=40):
    Dataset=mglearn.datasets.make_wave(n_samples=N)
    Input,Output=Dataset
    Input_train,Input_test,Output_train,Output_test=train_test_split(Input,Output,random_state=42)
    knn_R=KNeighborsRegressor(n_neighbors=K)
    knn_R.fit(Input_train,Output_train)
    knn_test_score=knn_R.score(Input_test,Output_test)
    knn_train_score=knn_R.score(Input_train,Output_train)
    title='KNeighborsRegressor/'+'train score :'+'%.2f'%knn_train_score+', test score :'+'%.2f'%knn_test_score
    plt.title(title)
    #선으로 도표그리기
    line=np.linspace(-3,3,1000).reshape(-1,1)
    A=plt.scatter(Input_train,Output_train,c='b',marker='^',alpha=0.5)
    B=plt.scatter(Input_test,Output_test,c='r',marker='v',alpha=0.5)
    C,=plt.plot(line,knn_R.predict(line),c='g')
    #plot은 ,을 붙어야 범례가 붙음.
    plt.legend(handles=(A,B,C),labels=(['tarin','test','predict']))
    plt.show()
 ```
plt.legend 함수는 항상 handles와 labels를 구별하여 붙일것, plot은 해당 plot 객체에','을 붙어야 범례가 붙음.
<br><br/>

```python
#---------------------------------
#선형회귀

#데이터셋을 만듬
Dataset=mglearn.datasets.make_wave(n_samples=60)
Input,Output=Dataset

#훈련용,테스트용으로 분류
Input_train,Input_test,Output_train,Output_test=train_test_split(Input,Output,random_state=42)

#선형회귀로 훈련
lr=LinearRegression().fit(Input_train,Output_train)

#score확인
print('train score:',lr.score(Input_train,Output_train))
print('test score:',lr.score(Input_test,Output_test))

#선형회귀 일차방정식의 기울기와 절편
print('lr.coef_:',lr.coef_)#기울기
print('lr.intercept_:',lr.intercept_)#절편
```
선형회귀의 기울기 피라미터는 lr 객체의 coef_ 속성에 저장되어있다.+절편 피라미터는 intercept_속성에 저장되어있다.
plt으로 도표를 만들때 scatter은 x값에 (N,)꼴이 들어가도 상관없지만 plot은 오류를 일으키므로 둘다 항상 (N,1)형태로 전환해주도록 하자
(N,1)형태로 전환하는 방법은 numpy의 reshape 함수를 이용하면 간단하다. 
ex.코드
```python
#보스턴 데이터에서 방의 개수로 산점도 그래프 작성
dot1=plt.scatter(boston_df['RM'],Output,c='r',alpha=0.5)


print(boston_df['RM'].shape)
#pandas series 타입임
print(type(boston_df['RM']))

#RM 데이터가 1행에 다 들어있으므로 열로 바꿔줘야됨
#reshape 함수를 사용하기 위해 넘파이 형식으로 바꿔줌
new_x=np.reshape(boston_df['RM'].to_numpy(),(-1,1))
print(new_x.shape)

#훈련용, 테스트용 분류
Input_train,Input_test,Output_train,Output_test=train_test_split(new_x,Output,random_state=0)
```
reshape 함수에서 (-1,1) -1은 본래 기준이 되었던 리스트의 길이에 따라 1개짜리 (N,1)리스트가 생성된다는 뜻
<br><br/>
데이터를 가공할때 우선 pd.DataFrame()형태로 가공하면 보기 편하다. 
+df는 'columns='를 사용해 열의 이름을 쉽게 바꿀수 있다.
pd.DataFrame(data,label,colums='feature')형식으로 만들면 좋음
이렇게 가공된 df는 라벨에 따라 분류하기도 편해짐 df=df.loc[0] 이란 소리는 해당 df에서 0인 라벨만 골라서 만들어진 df라는 뜻이다.
이렇게 각각 뽑아버리면 나중에 도표로 만들때도 편해짐
ex.
```python
#붙인라벨에 따라서 df 하나씩 생성
temp_df_0=temp_df.loc[0]
temp_df_1=temp_df.loc[1]
```
+이렇게 만들어진 df는 원하는 column만 따로 뺄수도 있음 
ex.
```python
iris_data_pl=iris_df['petal length (cm)']
```
<br><br/>
로지스틱 회귀분석에서 OUTPUT을 0,1로 만들어야 할때 해당 리스트가 numpy array이면 OUTPUT=npArray==value 형식으로 표현하면 된다.
ex.
```python
#---------------------------------
#로지스틱 회귀분석(분류)
iris=datasets.load_iris()

print(iris.keys())
print(iris.feature_names)
print(iris.target_names)
#1가지 특성만 가져옴 (petal length를 가져오기로 함)
iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(iris_df.head())
iris_data_pl=iris_df['petal length (cm)']
print(iris_data_pl.head())
#로지스틱 회귀분석을 위해 1개의 output=virginica를 재외하고 0으로 만듬
print(type(iris.target))
OutPut=iris.target==2
print(OutPut)
```
<br><br/>
솔레노이드 곡선을 만들기 위해 로지스틱 회귀분류의 .predict_proba(x) 함수가 필요하다.
이때 해당 함수로 만들어진 예측값은 (N,2)의 형태를 띄고 있는데
col=0은 1-p값이고(예측이 틀릴 확률)
col=1은 p값이다(예측이 맞을 확률)
ex.
```python
#꽃잎의 넓이에 대해 로지스틱 회귀모델을 이용해 추정

log_reg=LogisticRegression().fit(iris_data_pl,OutPut)
Input_pl=np.linspace(min(iris_data_pl),max(iris_data_pl),1000).reshape(-1,1)
#각input에 대한 확률을 예측
p=log_reg.predict_proba(Input_pl)
print(p.shape)
#0열은 False, 1열은 True 라고 생각
print(pd.DataFrame(p).head())
p=pd.DataFrame(p)
#도표화

plt.scatter(iris_data_pl,OutPut,c='g',alpha=0.5,label='Yes or No')
plt.plot(Input_pl,p[0],label='Non virginica')
plt.plot(Input_pl,p[1],label='virginica')
plt.ylabel('probability')
plt.xlabel('petal width')
plt.legend()
plt.show()
```
<br><br/>
도표를 분리하기 위한 함수는 
mglearn.plots.plot_2d_separator(사용한 모델,모델의 train data ,fill=True,alpha=0.2)#분류모델,train_input,채움,투명도 이다
이다.



