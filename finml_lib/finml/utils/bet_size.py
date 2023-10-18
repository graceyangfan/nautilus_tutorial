from scipy.stats import norm
def get_bet_size(prob,max_size = 5, num_classes =2):
    if prob >= 1.0:
        prob = prob -0.00000001
    bet_sizes = (prob - 1/num_classes) / (prob * (1 - prob))**0.5

    bet_sizes = max_size*( 2.0 * norm.cdf(bet_sizes) - 1)

    return bet_sizes