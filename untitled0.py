# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 00:03:35 2021

@author: abdka
"""


class SimpleBalancedDecisionTree():
    def __init__(self, indices):
        self.indices = indices
        self.test_values = test_values
        return

    def predict(self, data_row):
        i = 0
        # Move to right if
        while i < len(indices):
            if data_row[indices[i]] < test_values[i]:
                i = (2*i)+2
                # If leaf node
                if len(indices) <= i:
                    return 1
            # Move to left
            else:
                i = (2*i)+1
                # If leaf node
                if len(indices) <= i:
                    return 0





class Tree():
    def __init__(self, head):
        self.head = head
    
    # Adding nodes
    # The direction is an array containing zeros and ones
    # It describes how to navigate in the tree when placing a node
    def add(self, node, directions):
        temp = self.head
        placed = False
        
        while not placed:
            for d in directions:
                if d:
                    if temp.right == None:
                        temp.right = node
                        node.parent = temp
                        placed = True
                        break
                    else:
                        temp = temp.right
                else:
                    if temp.left == None:
                        temp.left = node
                        node.parent = temp
                        placed = True
                        break
                    else:
                        temp = temp.left
    
    
    
    
            
    
        


generating nodes

adding those nodes to the tree.
repeating adding algorithm, e.g. [left, left, right] until empty leaf found

each node has adding algorithm

The tree has nodes.


class DecisionNode():
    def __init__(self):
        self.parent
        self.left
        self.right
        self.indices
        self.parameters
        self.test_value

    def test(self, data_row):
        sum_val = sum(data_row[self.indices]*self.parameters)
        if sum_val < test_value:
            i = (2*i)+2
            # If leaf node
            if len(indices) <= i:
                return 1
            # Move to left
        else:
            i = (2*i)+1
            # If leaf node
            if len(indices) <= i:
                return 0


arr1 = [0, 1, 2, 3]

indices = [0, 3, 2, 1, 3]
values = [0, 3, 2, 1, 4]


if indice i in arr is above,
go to left node
if no left. return negative
else:
    go to right node
    if no right, return positive.:


Generating tree
