import pandas as pd
import numpy as np
import time
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor
from chinese_holiday import chinese_calendar
from sklearn.model_selection import KFold

shuoming = pd.read_csv('data/字段说明.csv',encoding='utf-8',engine='python')
dict = {}
for key,value in zip(shuoming['col_name'],shuoming['column_comments']):
    dict[key] = value

def read_data(path):
	df = pd.read_csv(path)
	df.rename(columns=lambda x:dict[x], inplace=True)
	df['date'] = pd.to_datetime(df['单据结束时间'])
	return df

def create_hot(path):
	df = read_data(path)
	hot_df= df[np.array(df['产品名称'] == '圆钢') + np.array(df['产品名称'] == '热卷')] #筛选热卷
	hot_df = hot_df[['重量','date','数量']]
	hot_df.index = hot_df['date']
	hot_df = hot_df['2015-01-05':]  #2014年的数据存在缺失值较多，弃之
	hot_df = hot_df.resample('D').sum()
	hot_df = hot_df[hot_df>0].resample('D').interpolate('linear')
	hot_df['date'] = hot_df.index
	hot_df['date'] = hot_df['date'].apply(lambda x:str(x)[:10]) #转str
	cal = chinese_calendar()
	holidays = cal.holiday()[1]
	hot_df = hot_df[hot_df['date'].isin(holidays).apply(lambda x: not x)]
	hot_df = hot_df.dropna(axis=0)
	hot_df['date'] = hot_df.index
	hot_df['doy'] = hot_df['date'].apply(lambda x:x.dayofyear)
	hot_df['day'] = hot_df['date'].apply(lambda x:x.day)
	hot_df['month'] = hot_df['date'].apply(lambda x:x.month)
	hot_df['dow'] = hot_df['date'].apply(lambda x:x.dayofweek)
	hot_df['woy'] = hot_df['date'].apply(lambda x:x.weekofyear)
	hot_df['month_start'] = hot_df['date'].apply(lambda x:x.is_month_start)
	hot_df['month_end'] = hot_df['date'].apply(lambda x:x.is_month_end)
	hot_df['quarter_start'] = hot_df['date'].apply(lambda x:x.is_quarter_start)
	hot_df['quarter_end'] = hot_df['date'].apply(lambda x:x.is_quarter_end)
	hot_df['year_start'] = hot_df['date'].apply(lambda x:x.is_year_start)
	hot_df['year_end'] = hot_df['date'].apply(lambda x:x.is_year_end)
	hot_df['diff_1'] = hot_df['重量'].diff(1)
	hot_df['diff_2'] = hot_df['diff_1'].diff(1)
	hot_df_day = hot_df.copy()
	hot_df_week = hot_df.copy()
	hot_df_day['pre_重量'] = hot_df_day['重量'].shift(-7)
	hot_df_week['pre_重量'] = hot_df_week['重量'].shift(-28)
	return hot_df_day, hot_df_week

def create_cold(path):
	df = read_data(path)
	hot_index = df[np.array(df['产品名称'] == '圆钢') + np.array(df['产品名称'] == '热卷')].index
	cold_df = df.drop(hot_index)
	cold_df = cold_df[['重量','date','数量']]
	cold_df.index = cold_df['date']
	cold_df = cold_df['2015-01-05':]
	cold_df = cold_df.resample('D').sum()
	cold_df = cold_df[cold_df>0].resample('D').interpolate('linear')
	cold_df['date'] = cold_df.index
	cold_df['date'] = cold_df['date'].apply(lambda x:str(x)[:10]) #转str
	cal = chinese_calendar()
	holidays = cal.holiday()[1]
	cold_df = cold_df[cold_df['date'].isin(holidays).apply(lambda x: not x)]
	cold_df = cold_df.dropna(axis=0)
	cold_df['date'] = cold_df.index
	cold_df['doy'] = cold_df['date'].apply(lambda x:x.dayofyear)
	cold_df['day'] = cold_df['date'].apply(lambda x:x.day)
	cold_df['month'] = cold_df['date'].apply(lambda x:x.month)
	cold_df['dow'] = cold_df['date'].apply(lambda x:x.dayofweek)
	cold_df['woy'] = cold_df['date'].apply(lambda x:x.weekofyear)
	cold_df['month_start'] = cold_df['date'].apply(lambda x:x.is_month_start)
	cold_df['month_end'] = cold_df['date'].apply(lambda x:x.is_month_end)
	cold_df['quarter_start'] = cold_df['date'].apply(lambda x:x.is_quarter_start)
	cold_df['quarter_end'] = cold_df['date'].apply(lambda x:x.is_quarter_end)
	cold_df['year_start'] = cold_df['date'].apply(lambda x:x.is_year_start)
	cold_df['year_end'] = cold_df['date'].apply(lambda x:x.is_year_end)
	cold_df['diff_1'] = cold_df['重量'].diff(1)
	cold_df['diff_2'] = cold_df['diff_1'].diff(1)
	cold_df_day = cold_df.copy()
	cold_df_week = cold_df.copy()
	cold_df_day['pre_重量'] = cold_df_day['重量'].shift(-7)
	cold_df_week['pre_重量'] = cold_df_week['重量'].shift(-28)
	return cold_df_day, cold_df_week

def lgb_model(y_name,train_data, test_data, params, nflod):
    columns = train_data.columns
    remove_columns = [y_name]
    features_columns = [column for column in columns if column not in remove_columns]
    train_features = train_data[features_columns].values
    train_labels = train_data[y_name].values

    test_features = test_data[features_columns].values
    kfolder = KFold(n_splits=nflod, shuffle=True, random_state=2018)
    kfold = kfolder.split(train_features, train_labels)
    print('start lightgbm train..')
    preds_list = list()
    for train_index, test_index in kfold:
        k_x_train = train_features[train_index]
        k_y_train = train_labels[train_index]
        k_x_test = train_features[test_index]
        k_y_test = train_labels[test_index]

        gbm = lgb.LGBMRegressor(**params)
        gbm = gbm.fit(k_x_train, k_y_train,
                      eval_metric="mse",
                      eval_set=[(k_x_train, k_y_train),
                                (k_x_test, k_y_test)],
                      eval_names=["train", "valid"],
                      early_stopping_rounds=100,
                      verbose=True)

        preds = gbm.predict(test_features, num_iteration=gbm.best_iteration_)

        preds_list.append(preds)

    length = len(preds_list)
    preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_list = list(preds_df.mean(axis=1))

    return preds_list

def xgb_model(y_name,train_data, test_data, xgb_params,num_boost_round):

    columns = train_data.columns
    remove_columns = [y_name]
    features_columns = [column for column in columns if column not in remove_columns]

    X_train = train_data[features_columns].values
    y_train = train_data[y_name].values

    X_test = test_data[features_columns].values

    d_train = xgb.DMatrix(X_train,y_train)
    d_test = xgb.DMatrix(X_test)

    res = xgb.cv(xgb_params, d_train, num_boost_round=num_boost_round, nfold=5, seed=2018, stratified=False,
             early_stopping_rounds=50, verbose_eval=10, show_stdv=True)
    print('start xgboost train..')
    gbdt = xgb.train(xgb_params, d_train, res.shape[0])
    
    return gbdt.predict(d_test)

def catboost_model(y_name,train_data, test_data):
    columns = train_data.columns
    remove_columns = [y_name]
    features_columns = [column for column in columns if column not in remove_columns]

    X_train = train_data[features_columns].values
    y_train = train_data[y_name].values

    X_test = test_data[features_columns].values
    cat_model = CatBoostRegressor(iterations=550, depth=6, learning_rate=0.03, loss_function='RMSE',
                                  random_seed=2018, rsm=0.8)
    print('start catboost train..')
    cat_model.fit(X_train, y_train)

    test_pred = cat_model.predict(X_test)
    return test_pred
	
if __name__ == '__main__':
	print('start loading data...')
	start_date ='2015-01-05'
	end_date = '2018-06-17'
	t1 = time.time()
	cal = chinese_calendar()
	holidays = cal.holiday()[1]

	price_df = pd.read_csv('data/price.csv',sep = '\t')
	price_df.index = pd.to_datetime(price_df['date'])
	price_df = price_df.sort_index()
	price_df = price_df.resample('D').interpolate('linear')
	price_df = price_df[start_date:end_date]
	price_df['date'] = price_df['date'].apply(lambda x:str(x)[:10]) #转str
	price_df = price_df[price_df['date'].isin(holidays).apply(lambda x: not x)]
	price_df = price_df.dropna(axis=0)

	weather_df = pd.read_csv('data/weather.csv')
	weather_df.index = pd.to_datetime(weather_df['date'])
	weather_df = weather_df.sort_index()
	weather_df = weather_df[start_date:end_date]
	weather_df['date'] = weather_df['date'].apply(lambda x:str(x)[:10]) #转str
	weather_df = weather_df[weather_df['date'].isin(holidays).apply(lambda x: not x)]
	weather_df = weather_df.dropna(axis=0)

	temp = weather_df[['weather','wind']]
	temp = pd.get_dummies(temp)
	temp['temperature'] = weather_df['temperature']
	temp['price'] = price_df['price']
	temp['date'] = temp.index
	
	hot_in_day, hot_in_week = create_hot('data/trainData_IN_final.csv')
	cold_in_day, cold_in_week = create_cold('data/trainData_IN_final.csv')
	hot_ext_day, hot_ext_week = create_hot('data/trainData_EXT_final.csv')
	cold_ext_day, cold_ext_week = create_cold('data/trainData_EXT_final.csv')

	lgb_parms = {
		"boosting_type": "gbdt",
		"num_leaves": 127,
		"max_depth": -1,
		"learning_rate": 0.05,
		"n_estimators": 10000,
		"max_bin": 425,
		"subsample_for_bin": 20000,
		"objective": 'regression',
		# "metric": 'l1',
		"min_split_gain": 0,
		"min_child_weight": 0.001,
		"min_child_samples": 20,
		"subsample": 0.8,
		"subsample_freq": 1,
		"colsample_bytree": 0.8,
		"reg_alpha": 3,
		"reg_lambda": 5,
		"seed": 2018,
		"n_jobs": 5,
		"verbose": 1,
		"silent": False
		}
	xgb_params = {
		'seed': 2018,
		'colsample_bytree': 0.7,
		'silent': 1,
		'subsample': 0.7,
		'learning_rate': 0.15,
		'objective': 'reg:linear',
		'max_depth': 6,
		'num_parallel_tree': 1,
		'min_child_weight': 1,
		'eval_metric': 'rmse',
		'nrounds': 500,
		}
	print('start training data...')
	pred_lgb = []
	for data in [hot_ext_week,cold_ext_week,hot_in_week,cold_in_week]:
		train_data = pd.merge(data,temp,on='date').drop('date',axis=1)[:-28]
		test_data = pd.merge(data,temp,on='date').drop('date',axis=1)[-28:]
		lgb_pred = lgb_model('pre_重量',train_data,test_data,lgb_parms,5)
		for i in range(4):
			pred_lgb.append(sum((lgb_pred)[7*i:7*(i+1)]))

	for data in [hot_ext_day,cold_ext_day,hot_in_day,cold_in_day]:
		
		train_data = pd.merge(data,temp,on='date').drop('date',axis=1)[:-7]
		test_data = pd.merge(data,temp,on='date').drop('date',axis=1)[-7:]
		lgb_pred = lgb_model('pre_重量',train_data,test_data,lgb_parms,5)
		pred_lgb.extend(lgb_pred)
	pred_lgb = np.array(pred_lgb)

	pred_xgb = []
	for data in [hot_ext_week,cold_ext_week,hot_in_week,cold_in_week]:
		train_data = pd.merge(data,temp,on='date').drop('date',axis=1)[:-28]
		test_data = pd.merge(data,temp,on='date').drop('date',axis=1)[-28:]
		xgb_pred = xgb_model('pre_重量', train_data, test_data, xgb_params,2000)
		for i in range(4):
			pred_xgb.append(sum(xgb_pred[7*i:7*(i+1)]))
		
	for data in [hot_ext_day,cold_ext_day,hot_in_day,cold_in_day]:
		train_data = pd.merge(data,temp,on='date').drop('date',axis=1)[:-7]
		test_data = pd.merge(data,temp,on='date').drop('date',axis=1)[-7:]
		xgb_pred = xgb_model('pre_重量', train_data, test_data, xgb_params,2000)
		pred_xgb.extend(xgb_pred)
	pred_xgb = np.array(pred_xgb)

	pred_cat = []
	for data in [hot_ext_week,cold_ext_week,hot_in_week,cold_in_week]:
		train_data = pd.merge(data,temp,on='date').drop('date',axis=1)[:-28]
		test_data = pd.merge(data,temp,on='date').drop('date',axis=1)[-28:]
		cat_pred = catboost_model('pre_重量', train_data, test_data)
		for i in range(4):
			pred_cat.append(sum(cat_pred[7*i:7*(i+1)]))
		
	for data in [hot_ext_day,cold_ext_day,hot_in_day,cold_in_day]:
		
		train_data = pd.merge(data,temp,on='date').drop('date',axis=1)[:-7]
		test_data = pd.merge(data,temp,on='date').drop('date',axis=1)[-7:]
		cat_pred = catboost_model('pre_重量', train_data, test_data)
		pred_cat.extend(cat_pred)
	pred_cat = np.array(pred_cat)
	pred = 0.9 * pred_cat + 0.075 * pred_lgb + 0.025 * pred_xgb

	print('save result...')
	result = pd.read_csv('答案模板/result.csv',engine='python')
	df = result
	df['VALUE'] = pred
	df.to_csv('result.csv',index=None)
	t2 = time.time()
	print('spend time:',t2-t1)


