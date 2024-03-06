import pandas as pd
import numpy as np
from deeptables.models.deeptable import DeepTable, ModelConfig
from deeptables.models.deepnets import DeepFM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


filename = 'IDSAI.csv'
df_IDSAI = pd.read_csv(filename)
print(df_IDSAI.shape)

features = df_IDSAI.drop(['label', 'tipo_ataque', 'ip_src', 'ip_dst', 'port_src', 'port_dst', 'protocols'], axis=1)
labels = df_IDSAI['tipo_ataque'].values


encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)


def objective(params):

    conf = ModelConfig(
        nets=DeepFM,
        categorical_columns='auto',
        metrics=['accuracy'],
        auto_categorize=True,
        auto_discrete=False,
        embeddings_output_dim=int(params['embeddings_output_dim']),
        embedding_dropout=params['embedding_dropout'],
        earlystopping_patience=5
    )


    dt = DeepTable(config=conf)


    dt.fit(X_train, y_train, epochs=int(params['epochs']), batch_size=int(params['batch_size']))


    preds_proba = dt.predict_proba(X_test)
    preds_labels = np.argmax(preds_proba, axis=1)
    accuracy = accuracy_score(y_test, preds_labels)


    return {'loss': -accuracy, 'status': STATUS_OK}


space = {
    'embeddings_output_dim': hp.choice('embeddings_output_dim', [10, 20, 30, 40, 50]),
    'embedding_dropout': hp.uniform('embedding_dropout', 0.1, 0.5),
    'epochs': hp.choice('epochs', [10, 50, 100]),
    'batch_size': hp.choice('batch_size', [32, 64, 128])
}


trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=20,
    trials=trials
)

print("Best Hyperparameters:", best)
