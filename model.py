import pandas as pd
import dill as pickle

from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


pd.options.mode.chained_assignment = None


def filter_data(df_full):
    columns_to_drop = [
        'session_id',
        'hit_date',
        'hit_type',
        'hit_page_path',
        'event_action',
        'event_label',
        'client_id',
        'visit_date',
        'visit_time',
        'device_screen_resolution',
        'hit_referer',
        'device_os',
        'utm_keyword',
        'device_model',
        'hit_time',
        'hit_number',
        'visit_number'
    ]
    return df_full.drop(columns_to_drop, axis=1)


def car_brand(df_full):
    df_full['car_brand'] = df_full['hit_page_path'].str.split('/').str[3]
    df_full.car_brand = df_full.car_brand.fillna('other')
    car_brand_new = []
    for elem in df_full.car_brand.tolist():
        if len(elem) > 2 & len(elem) <= 13:
            car_brand_new.append(elem)
        else:
            car_brand_new.append('other')
    ind = df_full.index
    df_full['car_brand'] = pd.Series(car_brand_new, index=ind)
    return df_full


def car_model(df_full):
    df_full['car_model'] = df_full['hit_page_path'].str.split('/').str[4]
    df_full.car_model = df_full.car_model.fillna('other')
    car_model_new = []
    for elem in df_full.car_model.tolist():
        if len(elem) <= 15:
            car_model_new.append(elem)
        else:
            car_model_new.append('other')
    ind = df_full.index
    df_full['car_model'] = pd.Series(car_model_new, index=ind)
    return df_full

def main():
    df1 = pd.read_csv('data/ga_hits.csv')
    df2 = pd.read_csv('data/ga_sessions.csv', low_memory=False)
    df_full = pd.merge(left=df1, right=df2, on='session_id', how='outer')

    df_full.event_action.fillna('other')
    event_action = ['sub_car_claim_click',
                    'sub_car_claim_submit_click',
                    'sub_open_dialog_click',
                    'sub_custom_question_submit_click',
                    'sub_call_number_click',
                    'sub_callback_submit_click',
                    'sub_submit_success',
                    'sub_car_request_submit_click',
                    0]
    event_action_num = []
    for elem in df_full.event_action.tolist():
        for x in event_action:
            if x == 0:
                event_action_num.append(0)
            elif elem == x:
                event_action_num.append(1)
                break
    df_full['event_value'] = pd.to_numeric(pd.Series(event_action_num))

    df_full = df_full.sample(frac=0.003)

    X = df_full.drop('event_value', axis=1)
    y = df_full['event_value']

    categorical_features = make_column_selector(dtype_include=object)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='float32'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('brand', FunctionTransformer(car_brand)),
        ('model', FunctionTransformer(car_model)),
        ('clean', FunctionTransformer(filter_data)),
        ('column_transformer', column_transformer)
    ])

    models = (
        LogisticRegression(random_state=42, penalty='l1', solver='saga'),
        RandomForestClassifier(random_state=42, n_estimators=50)
    )

    best_score = .0
    best_pipe = None

    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy', error_score='raise')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best_model: {type(best_pipe.named_steps["model"]).__name__}, accuracy: {best_score:.4f}')

    best_pipe.fit(X, y)
    with open('sber_avto.pkl', 'wb') as file:
        pickle.dump({
            'model': best_pipe,
            'metadata': {
                'name': "Prediction sberavto",
                'author': 'Denis Shkaraburov',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["model"]).__name__,
                'accuracy': best_score
            }
        }, file, recurse=True)


if __name__ == '__main__':
    main()