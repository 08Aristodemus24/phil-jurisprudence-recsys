from models.model_arcs import PhilJurisCollabFilter
from utilities.data_visualizers import view_vars
from utilities.data_loaders import load_ratings_small, load_precalc_params_small, load_Movie_List_pd


if __name__ == "__main__":
    # loads precalculated parameters X, W, B
    X, THETA, BETA = load_precalc_params_small()

    # load user-item rating and interactions matrices
    Y, R = load_ratings_small()

    # view data
    view_vars(Y, R, X, THETA, BETA)
    

