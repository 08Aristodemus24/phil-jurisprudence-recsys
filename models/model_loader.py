from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import (BinaryCrossentropy as bce_loss, 
    MeanSquaredError as mse_loss
)

from tensorflow.keras.metrics import (BinaryAccuracy, 
    Precision,
    Recall,
    AUC,
    BinaryCrossentropy as bce_metric, 
    MeanSquaredError as mse_metric
)

# from test_arcs_a import FM, DFM, MKR
from models.model_arcs import FM, DFM, MKR
from metrics.custom_metrics import f1_m

def load_model(model_name: str, protocol: str, n_users: int, n_items: int, n_features: int, layers_dims: list, epoch_to_rec_at: int, rec_alpha: float, rec_lambda: float, rec_keep_prob: float, regularization: str):
    """
    creates, compiles and returns chosen model to train

    args: 
        model_name - 
        protocol - 
        n_users - 
        n_items - 
        n_features - 
        epoch_to_rec_at - 
        rec_alpha - 
        rec_lambda - 
        rec_keep_prob - 
        regularization - 
    """
    
    protocols = {
        'A': {
            'loss': bce_loss(),
            'metrics': [bce_metric(), BinaryAccuracy(), Precision(), Recall(), AUC(), f1_m]
        },
        'B': {
            'loss': mse_loss(),
            'metrics': [mse_metric()]
        }
    }

    models = {
        'FM': FM(
            n_users=n_users, 
            n_items=n_items,
            emb_dim=n_features,
            lambda_=rec_lambda,
            regularization=regularization),
        'DFM': DFM(
            n_users=n_users, 
            n_items=n_items, 
            emb_dim=n_features,
            layers_dims=layers_dims,
            lambda_=rec_lambda, 
            keep_prob=rec_keep_prob, 
            regularization=regularization)
    }

    model = models[model_name]

    model.compile(
        optimizer=Adam(learning_rate=rec_alpha),
        loss=protocols[protocol]['loss'],
        metrics=protocols[protocol]['metrics']
    )
       
    return model