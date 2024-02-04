import time
import math
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats.mstats import winsorize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.base import BaseEstimator
# ------------------------------------------------------------------

def PreProcess_Initial(in_X):
    # 1-of-n encoding
    in_X = pd.get_dummies(in_X, columns=in_X.columns[(in_X.dtypes == object) | (in_X.dtypes == "category")],drop_first=True)
    # zero imputation for missing values
    in_X.fillna(0,inplace=True)
    return in_X

def PreProcess(X_1, X_2): # learn from X_1 and then apply to both X_1 and X_2
    for c in X_1.columns[(X_1.dtypes != object) & (X_1.dtypes != "category")]:
        # winsorization
        if X_1[c].nunique()>60: 
            winsorize(X_1[c],limits=[0.05, 0.05],inplace=True)
            MyMin = X_1[c].min()
            MyMax = X_1[c].max()
            X_2[c] = np.where(X_2[c]>MyMax,MyMax,X_2[c])
            X_2[c] = np.where(X_2[c]<MyMin,MyMin,X_2[c])
        # Min-Max Scaling
        temp = np.array(X_1[c]).reshape(-1, 1)
        MyScaler = preprocessing.MinMaxScaler()
        MyScaler.fit(temp)
        X_1[c] = MyScaler.transform(temp)
        temp = np.array(X_2[c]).reshape(-1, 1)
        X_2[c] = MyScaler.transform(temp)
    return X_1, X_2

def PrepareData_Train(My_FileName): 
    # the file should have a column named "y"      
    df_train = pd.read_csv('csv\\'+My_FileName+'-TRAIN.csv',sep=",")
    y_train = df_train['y']
    X_train = df_train.drop('y',axis=1)
    X_train = PreProcess_Initial(X_train)
    return X_train, y_train

def PrepareData_Test(My_FileName,X_train, y_train): 
    # the file should have a column named "y"      
    df_test = pd.read_csv('csv\\'+My_FileName+'-TEST.csv',sep=",")
    y_test = df_test['y']
    X_test = df_test.drop('y',axis=1)
    X_test = PreProcess_Initial(X_test)
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    X_test = X_test[X_train.columns]
    X_train, X_test = PreProcess(X_train, X_test)
    return X_test, y_test

def Centroid0(X_train, y_train):

    X0 = X_train.loc[y_train==0]  
    
    X0_Center = pd.DataFrame(X0.mean()).T

    X0_Std = X0.std().fillna(0)
    X0_Std = np.where(X0_Std<=X0_Center*0.01,X0_Center*0.01,X0_Std)
    X0_Std = pd.DataFrame(np.where(X0_Std<=0.01,0.01,X0_Std))
    X0_Std.columns = X_train.columns.to_list()
    X0_Std.index = X0_Center.index

    X0_Mad = X0.mad().fillna(0)
    X0_Mad = np.where(X0_Mad<=X0_Center*0.01,X0_Center*0.01,X0_Mad)
    X0_Mad = pd.DataFrame(np.where(X0_Mad==0,0.01,X0_Mad))
    X0_Mad.columns = X_train.columns.to_list()
    X0_Mad.index = X0_Center.index

    X0_CoV = X0.std().fillna(0) / X0.mean().fillna(0)
    X0_CoV = np.where(np.isnan(X0_CoV),0.01,X0_CoV)
    X0_CoV = np.where(np.isinf(X0_CoV),3,X0_CoV)
    X0_CoV = np.where(X0_CoV<=0.01,0.01,X0_CoV)
    X0_CoV = pd.DataFrame(X0_CoV).T
    X0_CoV.columns = X_train.columns.to_list()
    X0_CoV.index = X0_Center.index
    
    return X0_Center, X0_Std, X0_Mad, X0_CoV

# ------------------------------------------------------------------

def Centroid1(X_train, y_train):

    X1 = X_train.loc[y_train==1]  
    
    X1_Center = pd.DataFrame(X1.mean()).T

    X1_Std = X1.std().fillna(0)
    X1_Std = np.where(X1_Std<=X1_Center*0.01,X1_Center*0.01,X1_Std)
    X1_Std = pd.DataFrame(np.where(X1_Std<=0.01,0.01,X1_Std))
    X1_Std.columns = X_train.columns.to_list()
    X1_Std.index = X1_Center.index

    X1_Mad = X1.mad().fillna(0)
    X1_Mad = np.where(X1_Mad<=X1_Center*0.01,X1_Center*0.01,X1_Mad)
    X1_Mad = pd.DataFrame(np.where(X1_Mad==0,0.01,X1_Mad))
    X1_Mad.columns = X_train.columns.to_list()
    X1_Mad.index = X1_Center.index

    X1_CoV = X1.std().fillna(0) / X1.mean().fillna(0)
    X1_CoV = np.where(np.isnan(X1_CoV),0.01,X1_CoV)
    X1_CoV = np.where(np.isinf(X1_CoV),3,X1_CoV)
    X1_CoV = np.where(X1_CoV<=0.01,0.01,X1_CoV)
    X1_CoV = pd.DataFrame(X1_CoV).T
    X1_CoV.columns = X_train.columns.to_list()
    X1_CoV.index = X1_Center.index
    
    return X1_Center, X1_Std, X1_Mad, X1_CoV

# ------------------------------------------------------------------

def Calculate_Relevance(data, target, bins=10):
    Rel_main_df = pd.DataFrame()
    Rel_detail_df = pd.DataFrame()
    cols = data.columns
    for ivars in cols[~cols.isin([target])]:
        My_y = data[target]
        if (data[ivars].dtype.kind in 'biufc') and (len(np.unique(data[ivars]))>bins):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': My_y})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': My_y})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['%'] = d['Events'] / d['N']
        d['Lift'] = d['%'] / My_y.mean()
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['Importance'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        # print("Information value of " + ivars + " is " + str(round(d['IV Part'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "Importance" : [d['Importance'].sum()]}, columns = ["Variable", "Importance"])
        Rel_main_df=pd.concat([Rel_main_df,temp], axis=0)
        Rel_detail_df=pd.concat([Rel_detail_df,d], axis=0)
            
    Rel_main_df = Rel_main_df.set_index('Variable')
    Rel_main_df["Importance"]=np.where(Rel_main_df["Importance"]>1000,1000,Rel_main_df["Importance"])

    return Rel_main_df, Rel_detail_df

# ------------------------------------------------------------------

def Calc_Attraction (df, X_Std, X_Mad, X_CoV, X_Center, Hom_type, Importance):
    
    Attraction = pd.DataFrame(np.nan, index=df.index, columns=range(0,1))
    num_cols = df.shape[1]
    for c in range(0,1):
        # homogeneity as the inverse of sparsity
        if Hom_type == '1':
            Hom = 1. / X_Std.iloc[c,:num_cols].copy()
            Hom = Hom * X_Std.iloc[c,:num_cols].copy()
        elif Hom_type == 'mad':
            Hom = 1. / X_Mad.iloc[c,:num_cols].copy()
        elif Hom_type == 'std':
            Hom = 1. / X_Std.iloc[c,:num_cols].copy()
        elif Hom_type == 'var':
            Hom = 1. / X_Std.iloc[c,:num_cols].copy()
            Hom = Hom / X_Std.iloc[c,:num_cols].copy()
        elif Hom_type == 'cov':
            Hom = 1. / X_CoV.iloc[c,:num_cols].copy()
        else:
            Hom = 1. / X_Std.iloc[c,:num_cols].copy()
            Hom = Hom * X_Std.iloc[c,:num_cols].copy()

        # multiply homogeneity with importance for each column j
        #ImpHom = Hom.multiply((1+Importance['Importance'])**2,axis='index')
        ImpHom = Hom.multiply(Importance['Importance'],axis='index')
        
        # L2: Euclidean style
        Dif = (df.iloc[:,:num_cols] - X_Center.iloc[c,:num_cols]).copy()**2
        
        # the dot product of difference*(importance*homogeneity) gives distance for each row i
        Dist = Dif.dot(ImpHom)

        Dist = np.where(Dist<0.000001,0.000001,Dist)

        Attraction[c] = 1 / (Dist)

    Attraction.replace(np.nan, 0, inplace=True)
    Attraction.replace([-np.inf], 0, inplace=True)
    Attraction.replace([np.inf], 0, inplace=True)
    return Attraction

# ------------------------------------------------------------------

def Calc_Scaled(x,colname,pred_train_original):
    # Scaling to 0-1
    scaler = preprocessing.MinMaxScaler()
    temp = np.array(pred_train_original).reshape(-1, 1)  # pred_train_original must be used.. otherwise leakage!
    scaler.fit(temp)
    temp = np.array(x).reshape(-1, 1)
    x2 = scaler.transform(temp)
    x2 = pd.DataFrame(x2)
    x2.columns = [colname]
    return x2

# ------------------------------------------------------------------

def Calc_Performance( pred, pred_class, y):
    #print('Perf:',len(pred),len(pred_class),len(y))
    TN, FP, FN, TP = confusion_matrix(y, pred_class, labels=[0, 1]).ravel()
    if TP+FN>0:
        accuracy    = (TP+TN)/(TP+TN+FP+FN)
        sensitivity = TP / (TP+FN)
        specifity   = TN / (TN+FP)
        fpr, tpr, thresholds = roc_curve(y, pred)
        AUC = auc(fpr, tpr)
        GINI = 2 * AUC - 1
    else:
        accuracy    = 0
        sensitivity = 0
        specifity   = 0        
        fpr=0
        tpr=0
        thresholds = 0
        AUC = 0
        GINI = 0
    return TP, FP, TN, FN, accuracy, sensitivity, specifity, AUC, GINI

# ------------------------------------------------------------------

def NCIVH(X, y, Hom_type, 
         X0_Center, X0_Std, X0_Mad, X0_CoV,
         X1_Center, X1_Std, X1_Mad, X1_CoV,
         Importance, pred_train_original, train_mode, cut_off):
    Attraction0 = Calc_Attraction( X, X0_Std, X0_Mad, X0_CoV, X0_Center, Hom_type, Importance )
    Attraction1 = Calc_Attraction( X, X1_Std, X1_Mad, X1_CoV, X1_Center, Hom_type, Importance )
    # conversion to scores
    Attraction0Sum = Attraction0.sum(axis=1)
    Attraction1Sum = Attraction1.sum(axis=1)
    AttractionSum = Attraction0Sum + Attraction1Sum
    Attraction0 = Attraction0.div(AttractionSum,axis=0)
    Attraction1 = Attraction1.div(AttractionSum,axis=0)

    pred_class_info = pd.concat([Attraction0.sum(axis=1),Attraction1.sum(axis=1)], axis=1)

    # NET Similarity to class ONE
    pred = pred_class_info.iloc[:,1] - pred_class_info.iloc[:,0]
    pred = Calc_Scaled(x=pred,colname='score',pred_train_original=pred_train_original)
    pred_class = np.full((len(pred_class_info),1), False, dtype=bool)

    if train_mode==True:
        pred_class[pred.nlargest(y.sum(), ['score']).index] = True
        cut_off = float(pred[pred_class==True].min())
    else:
        pred_class[pred>=cut_off] = True
    pred_class = np.array(pred_class[:,0],dtype=bool)
    
    TP, FP, TN, FN, accuracy, sensitivity, specifity, AUC, GINI = Calc_Performance(pred, pred_class, y)
    exp_results = pd.DataFrame(columns = ['dataset', 'Hom_type', 'TP', 'FP', 'TN', 'FN', 'accuracy', 'sensitivity', 'specifity', 'AUC', 'GINI', 'cut_off', 'Top10_Flag'])
    exp_results.loc[0,'TP'] = TP
    exp_results['FP'] = FP
    exp_results['TN'] = TN
    exp_results['FN'] = FN
    exp_results['accuracy'] = accuracy
    exp_results['sensitivity'] = sensitivity
    exp_results['specifity'] = specifity
    exp_results['AUC'] = AUC
    exp_results['GINI'] = GINI
    exp_results['cut_off'] = 0
    exp_results['Top10_Flag'] = 0
    exp_results['Hom_type'] = Hom_type
    
    return pred, pred_class, exp_results

# ------------------------------------------------------------------

def NCIVH_Tuning(X_train, y_train, My_results, pred_train_original, Hom_type):
    # LEARNING PHASE STEP 1 - Importance
    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train,name='y')
    My_data = pd.concat([X_train,y_train],axis=1)
    Rel_main, Rel_detail = Calculate_Relevance(data=My_data, target='y')
    Importance = Rel_main.copy()
    tmp = Importance[Importance.Importance>0.0]
    Most_important_columns = tmp.nlargest(1000,Importance).index
    # LEARNING PHASE STEP 2 - CENTROIDS
    X0_Center, X0_Std, X0_Mad, X0_CoV = Centroid0(X_train, y_train)
    X1_Center, X1_Std, X1_Mad, X1_CoV = Centroid1(X_train, y_train)

    # LEARNING PHASE STEP 3: NCIVH
    pred_train_original = np.full((len(X_train),1), False, dtype=float)
    # Find BEST Hom_type
    for Hom_type_counter in ['1','mad','std','var','cov']:
        pred, pred_class, exp_results = NCIVH(X_train,y_train, Hom_type_counter,  \
                                            X0_Center, X0_Std, X0_Mad, X0_CoV, \
                                            X1_Center, X1_Std, X1_Mad, X1_CoV, \
                                            Importance, pred_train_original, True, 0)
        My_results = pd.concat([My_results,exp_results], ignore_index = True)
        
    return My_results, pred_train_original, Importance, \
           X0_Center, X0_Std, X0_Mad, X0_CoV,  \
           X1_Center, X1_Std, X1_Mad, X1_CoV,  \
           Rel_main, Rel_detail, Most_important_columns

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

class Performance:
    def __init__(self, pred, pred_class, y):
        TP, FP, TN, FN, accuracy, sensitivity, specifity, AUC, GINI = Calc_Performance(pred, pred_class, y)
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN
        self.accuracy = accuracy
        self.sensitivity = sensitivity
        self.specifity = specifity
        self.AUC = AUC
        self.GINI = GINI

# --------------

class NCIVHClassifier(BaseEstimator):
    def __init__(self, Relevance = 'IV', Hom_type="1"):
        self.Relevance = Relevance
        self.version = 1.0
        self.Hom_type=Hom_type

    def fit(self, X, y):
        self.X_train = pd.DataFrame(X)
        self.y_train = pd.Series(y)                      
        self.My_results = pd.DataFrame(columns = ['dataset', 'Hom_type', 'TP', 'FP', 'TN', 'FN', 'accuracy', 'sensitivity', 'specifity', 'AUC', 'GINI', 'cut_off', 'Top10_Flag'])
        self.pred_train_original, self.Importance                           = pd.DataFrame(), pd.DataFrame()
        self.X0_Center, self.X0_Std, self.X0_Mad, self.X0_CoV = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.X1_Center, self.X1_Std, self.X1_Mad, self.X1_CoV = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.Rel_main, self.Rel_detail, self.Most_important_columns         = pd.DataFrame(), pd.DataFrame(), []
        self.My_results, self.pred_train_original, self.Importance, \
        self.X0_Center, self.X0_Std, self.X0_Mad, self.X0_CoV, \
        self.X1_Center, self.X1_Std, self.X1_Mad, self.X1_CoV, \
        self.Rel_main, self.Rel_detail, self.Most_important_columns          \
        =                                                                    \
        NCIVH_Tuning(self.X_train, self.y_train, self.My_results, self.pred_train_original, self.Hom_type)
        self.My_results = self.My_results.sort_values(by='AUC', ascending=False)
        self.My_results.AUC = self.My_results.AUC.astype(float)
        self.My_results.loc[self.My_results.nlargest(10,['AUC']).index,'Top10_Flag'] = 1
        self.best_params = self.My_results.nlargest(1,['AUC']).loc[:,['Hom_type', 'cut_off']]
        self.best_AUC     = self.My_results.nlargest(1,['AUC']).loc[:,['AUC']]
        self.best_cut_off = self.My_results.nlargest(1,['AUC']).loc[:,['cut_off']]
        self.best_cut_off = float(self.best_cut_off.iloc[0,0])
        #print(self.best_params)
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.accuracy = 0
        self.sensitivity = 0
        self.specifity = 0
        self.AUC = 0
        return
    
# --------------

    def __predictions(self, X):
        temp_res_test0 = NCIVH(X, y = np.zeros(X.shape[0]), 
                             Hom_type = self.best_params.Hom_type.iloc[0], 
                             X0_Center=self.X0_Center, X0_Std=self.X0_Std, X0_Mad=self.X0_Mad, X0_CoV=self.X0_CoV,
                             X1_Center=self.X1_Center, X1_Std=self.X1_Std, X1_Mad=self.X1_Mad, X1_CoV=self.X1_CoV,
                             Importance=self.Importance, pred_train_original=self.pred_train_original,
                             train_mode=False,
                             cut_off=self.best_cut_off
                            )
        return temp_res_test0

    def predict(self, X):
        X = pd.DataFrame(X)
        MyPreds = pd.DataFrame(self.__predictions(X)[1])
        return MyPreds

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        MyScores = self.__predictions(X)[0]
        return MyScores

