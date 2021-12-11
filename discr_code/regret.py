import numpy as np

def regret(loss_lst, calA, algo_loss, T):
    # size of tmp: T x action_space
    calA_size = len(calA)

    loss_per_action = []
    for i in range(calA_size):
        s = 0
        for t in range(T+1):
            s += loss_lst[t][i]
        loss_per_action.append(s)
        

    min_loss_hindsight = np.min(loss_per_action)
    print ("Algorithm's Loss: %f"%(sum(algo_loss)))
    print ("Best fixed: %f"%min_loss_hindsight)
    # for now, we're not comparing with the continuous case
    print ("Regret:%f"%(sum(algo_loss) - min_loss_hindsight))
    return (sum(algo_loss) - min_loss_hindsight)


