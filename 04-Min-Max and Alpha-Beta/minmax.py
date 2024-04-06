# class TreeNode:
    
#     def __init__(self,value,children = []):
#         self.value = value
#         self.children = children
        
# def minmax(node,depth,maximizing_player):
#     if depth == 0 or not node.children:
#         return node.value,[node.value]
        
#     if maximizing_player:
#         max_value = float("-inf")
#         max_path = []
        
#         for child_node in node.children:
#             child_value,child_path = minmax(child_node,depth-1,False)
#             if child_value > max_value:
#                 max_value = child_value
#                 max_path = [node.value] + child_path
#         return max_value,max_path
#     else:
#         min_value = float("inf")
#         min_path = []
        
#         for child_node in node.children:
#             child_value,child_path = minmax(child_node,depth-1,True)
#             if child_value < min_value:
#                 min_value =child_value
#                 min_path = [node.value] + child_path
        
#         return min_value,min_path
# game_tree=TreeNode(0,[TreeNode(0,[TreeNode(3),TreeNode(12)]),TreeNode(0,[TreeNode(8),TreeNode(2)]),])
        
# val,path=minmax(game_tree,2,False)
# print(val,path)



# A simple Python3 program to find
# maximum score that
# maximizing player can get
import math

def minimax (curDepth, nodeIndex,
			maxTurn, scores, 
			targetDepth):

	# base case : targetDepth reached
	if (curDepth == targetDepth): 
		return scores[nodeIndex]
	
	if (maxTurn):
		return max(minimax(curDepth + 1, nodeIndex * 2, 
					False, scores, targetDepth), 
				minimax(curDepth + 1, nodeIndex * 2 + 1, 
					False, scores, targetDepth))
	
	else:
		return min(minimax(curDepth + 1, nodeIndex * 2, 
					True, scores, targetDepth), 
				minimax(curDepth + 1, nodeIndex * 2 + 1, 
					True, scores, targetDepth))
	
# Driver code
scores = [0,1,3,12,4,8,2]

treeDepth = math.log(len(scores), 2)

print("The optimal value is : ", end = "")
print(minimax(0, 0, True, scores, treeDepth))

# This code is contributed
# by rootshadow
