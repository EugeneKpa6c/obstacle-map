import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
import optuna
from optuna.trial import TrialState
from tqdm.auto import tqdm
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# Загрузка данных
df = pd.read_csv('data/masks_data.csv')

# Определение X и y
X = df.drop('Y', axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': trial.suggest_float('learning_rate', 0.000001, 0.001),
        'max_depth': trial.suggest_int('max_depth', 3, 9, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'tree_method': 'gpu_hist'
    }
    
    model = xgb.XGBClassifier(**param)
    
    # Использование SMOTE только на обучающей выборке
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('model', model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)

    return 1 - roc_auc


def tqdm_callback(study, trial):
    if hasattr(study, "_tqdm_bar"):
        completed_trials = len(study.get_trials(states=[TrialState.COMPLETE]))
        study._tqdm_bar.update(completed_trials - study._tqdm_bar.n)
    else:
        total_trials = n_trials
        completed_trials = len(study.get_trials(states=[TrialState.COMPLETE]))
        study._tqdm_bar = tqdm(total=total_trials, leave=False)
        study._tqdm_bar.update(completed_trials)
    if study._stop_flag:
        study._tqdm_bar.close()

study = optuna.create_study(direction='minimize')
n_trials = 10
study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])

print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('Value: ', 1 - trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# Обучение модели на лучших параметрах
best_params = trial.params
best_params['verbosity'] = 0
best_params['objective'] = 'binary:logistic'
best_params['tree_method'] = 'gpu_hist'

# Создание объекта модели с лучшими параметрами
best_model = xgb.XGBClassifier(**best_params)

# Создание конвейера с SMOTE и моделью
pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('model', best_model)])

# Обучение конвейера на обучающих данных
pipeline.fit(X_train, y_train)

# Предсказание вероятностей на тестовых данных
y_pred_proba = pipeline.named_steps['model'].predict_proba(X_test)[:, 1]

# Поиск оптимального порога для максимизации F1-меры
best_threshold = 0.5
best_f1_score = 0

for threshold in np.linspace(0, 1, 101):
    y_pred = (y_pred_proba > threshold).astype(int)
    current_f1_score = f1_score(y_test, y_pred)
    if current_f1_score > best_f1_score:
        best_f1_score = current_f1_score
        best_threshold = threshold

y_pred_best = (y_pred_proba > best_threshold).astype(int)
best_precision = precision_score(y_test, y_pred_best)
best_recall = recall_score(y_test, y_pred_best)

print("Лучший порог:", best_threshold)
print("Лучшая F1-мера:", best_f1_score)
print("Точность:", best_precision)
print("Полнота:", best_recall)

# Сохранение весов модели
pipeline.named_steps['model'].save_model('best_model.xgb')
print('Model weights saved to "best_model.xgb"')