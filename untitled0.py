# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 00:03:35 2021

@author: abdka
"""

class DecisionTree():
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
        
    
    
class DecisionNode():
    def __init__(self):
        self.indices
        self.parameters
        self.test_value
        
    def test(self, data_row):
        sum_val = sum(data_row[self.indices]*self.parameters)
        if sum_val < test_value:
            



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

