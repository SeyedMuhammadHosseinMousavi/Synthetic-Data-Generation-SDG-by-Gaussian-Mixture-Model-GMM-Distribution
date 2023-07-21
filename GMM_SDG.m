%% Synthetic Data Generation (SDG) by Gaussian Mixture Model (GMM) Distribution 
% Developed by Seyed Muhammad Hossein Mousavi (July 2023)
% The dataset is "Iris" dataset
% Number of desired synthetic samples can be defined in "NoofSynthetic"
% Gaussian Mixture Model (GMM) distribution is used to generate the synthetic data
% K-means clustering is used to extract labels for classification task
% SVM is used as the classifier 
clear;
clc;
close all;
load fisheriris
Data=meas;
Input=Data;
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target';
DataSize=size(Data);DataSize=DataSize(1,1);
Classes=3; % Number of categories (for this data, it is 3)

%% Number of desire synthetic samples to be generated
NoofSynthetic=800; 

%% Fit Gaussian Mixture Model (GMM) to the original data
GMModel1 = fitgmdist(Input,Classes,SharedCovariance=true);

%% Synthetic Data Generation (SDG)
SDG = random(GMModel1,NoofSynthetic);

%% Getting labels of synthetic generated data by K-means clustering
[Lbl,C,sumd,D] = kmeans(SDG,Classes,'MaxIter',10000,...
    'Display','final','Replicates',10);
% Generated data plus labels 
AugAll=[SDG Lbl];

%% Plot original and gnerated dataset in two dimensions
F1=3; % Feature one
F2=2; % Feature two
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
gscatter(Input(:,F1),Input(:,F2),Target,'rgm'); title('Original Data');
subplot(2,2,2)
gscatter(SDG(:,F1),SDG(:,F2),Lbl,'rgm'); title('Augmented Data');
subplot(2,2,3)
plot(Input, 'linewidth',1); xlim([0 size(Data, 1)]);
subplot(2,2,4)
plot(SDG(:,1:end), 'linewidth',1); xlim([0 size(SDG, 1)]);

%% Train and Test
% Training augmented dataset by SVM
% Training multiple times for getting average of them
TrainNumbers=5; % Number of trains
for i = 1:TrainNumbers
Mdlsvm = fitcecoc(SDG,Lbl);
CVMdlsvm = crossval(Mdlsvm);
SVMError(i) = kfoldLoss(CVMdlsvm);
SVMAccAugTrainAvg(i) = (1 - SVMError(i))*100;
disp ([' Training SVM No "',num2str(i)]);
end
SVMAccAugTrain=sum(SVMAccAugTrainAvg)/TrainNumbers; % Train accuracy

% Predict new data by augmented model (SVM) on the original dataset
[label5,score5,cost5] = predict(Mdlsvm,Input);

% Test error and accuracy calculations
a=0;b=0;c=0;
for i=1:DataSize
if label5(i)== 1
a=a+1;
elseif label5(i)==2
b=b+1;
else
label5(i)==3
c=c+1;
end;end;
erra=abs(a-50);errb=abs(b-50);errc=abs(c-50);
err=erra+errb+errc;TestErr=err*100/DataSize;SVMAccAugTest=100-TestErr; % Test Accuracy

% Train and Test Accuracy Results
AugRessvm = [' Augmented Train SVM "',num2str(SVMAccAugTrain),'" Augmented Test SVM"', num2str(SVMAccAugTest),'"'];
disp(AugRessvm);
