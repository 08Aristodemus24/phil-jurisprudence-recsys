from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import (BinaryCrossentropy as bce_loss, 
    MeanSquaredError as mse_loss
)

from tensorflow.keras.metrics import (BinaryAccuracy, 
    Precision,
    Recall,
    AUC,
    BinaryCrossentropy as bce_metric, 
    MeanSquaredError as mse_metric,
    RootMeanSquaredError
)

from models.test_arcs_a import FM as FM_r, DFM as DFM_r
from models.model_arcs import FM, DFM
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
            'loss': bce_loss(from_logits=True),
            'metrics': [bce_metric(from_logits=True), BinaryAccuracy(), Precision(), Recall(), AUC(), f1_m]
        },
        'B': {
            'loss': mse_loss(),
            'metrics': [mse_metric(), RootMeanSquaredError()]
        }
    }

    models = {
        'FM': {
            "type": FM(
                n_users=n_users, 
                n_items=n_items,
                emb_dim=n_features,
                lambda_=rec_lambda,
                regularization=regularization) if protocol == "A" else FM_r(
                    n_users=n_users,
                    n_items=n_items,
                    emb_dim=n_features,
                    lambda_=rec_alpha,
                    regularization=regularization
                ),
            "name": "factorization machine"
        },
        'DFM': {
            "type": DFM(
                n_users=n_users, 
                n_items=n_items, 
                emb_dim=n_features,
                layers_dims=layers_dims,
                lambda_=rec_lambda, 
                keep_prob=rec_keep_prob, 
                regularization=regularization) if protocol == "A" else DFM_r(
                    n_users=n_users, 
                    n_items=n_items, 
                    emb_dim=n_features,
                    layers_dims=layers_dims,
                    lambda_=rec_lambda, 
                    keep_prob=rec_keep_prob, 
                    regularization=regularization
                ),
            "name": "deep factorization machine"
        }
    }

    model = models[model_name]

    model["type"].compile(
        optimizer=Adam(learning_rate=rec_alpha),
        loss=protocols[protocol]['loss'],
        metrics=protocols[protocol]['metrics']
    )
       
    return model