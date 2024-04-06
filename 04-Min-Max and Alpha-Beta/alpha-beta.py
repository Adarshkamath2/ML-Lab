MAX , MIN = 1000,-1000

def alphabeta_minmax(depth,nodeIndex,maximizingPlayer,values,alpha,beta):
    
    if depth == 3:
        return values[nodeIndex]
    
    if maximizingPlayer:
        best = MIN
        for i in range(0,2):
            val = alphabeta_minmax(depth+1,nodeIndex*2+i,False,values,alpha,beta)
            
            best = max(best,val)
            alpha = max(alpha,best)
            
            if beta<= alpha:
                break
        return best
    else:
        best = MAX
        for i in range(0,2):
            val = alphabeta_minmax(depth+1,nodeIndex*2+i, True,values,alpha,beta)
            best = min(best,val)
            alpha = min(alpha,best)
            
            if beta<= alpha:
                break
        return best

values = [3,5,6,9,1,2,0,-1]

print("The Optimal Value is ", alphabeta_minmax(0,0,True,values,MIN,MAX))
            