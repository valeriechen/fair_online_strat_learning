import numpy as np

def compute_regret_exp3(loss_lst, calA, algo_loss, T):
    # size of tmp: T x action_space
    calA_size = len(calA)

    min_loss_hindsight = 0.0
    print ("Algorithm's Loss: %f"%(sum(algo_loss)))
    print ("Best fixed: %f"%min_loss_hindsight)
    print ("Regret:%f"%(sum(algo_loss) - min_loss_hindsight))
    return (sum(algo_loss) - min_loss_hindsight, min_loss_hindsight)


def compute_regret_grind(algo_loss, bf):
    print ("Grinding Algorithm's Loss: %f"%(sum(algo_loss)))
    print ("Grinding Best fixed: %f"%bf)
    print ("Grinding Regret:%f"%(sum(algo_loss) - bf))
    return (sum(algo_loss) - bf)
