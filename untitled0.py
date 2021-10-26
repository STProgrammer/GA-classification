# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 00:03:35 2021

@author: abdka
"""
import numpy as np

class SimpleBalancedDecisionTree():
    def __init__(self, indices, test_values):
        self.indices = indices
        self.test_values = test_values
        return

    def predict(self, data_row):
        i = 0
        # Move to right if
        while i < len(indices):
            if data_row[indices[i]] < self.test_values[i]:
                i = (2*i)+2
                # If leaf node (also no more nodes)
                if len(indices) <= i:
                    return 1
            # Move to left
            else:
                i = (2*i)+1
                # If leaf node (also no more nodes)
                if len(indices) <= i:
                    return 0





class DecisionTree():
    def __init__(self, head):
        self.head = head
    
    # Adding nodes
    # The direction is an array containing zeros and ones
    # It describes how to navigate in the tree when placing a node
    # 0 means to left, 1 means to right
    def add(self, node, directions):
        temp = self.head
        placed = False
        
        # Repeat directions while not placed
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
    
    
    def predict(self, data_row):
        temp = self.head
        
        direction = temp.test(data_row)
        
        if direction:
            temp = temp.right
            # No nodes on the right
            if temp == None:
                return 1
        else:
            temp = temp.left
            # No nodes on the left
            if temp == None:
                return 0
    
    def predict_all(self, data_set):
        results = np.zeros(len(data_set))
        for i in range(len(data_set)):
            results[i] = self.predict[data_set[i]]
    
    def mutate_tree(self):
        
            






class DecisionNode():
    def __init__(self, indices, parameters, test_value):
        self.parent = None
        self.left = None
        self.right = None
        self.indices = indices
        self.parameters = parameters
        self.test_value = test_value

    def test(self, data_row):
        sum_val = sum(data_row[self.indices]*self.parameters)
        if sum_val < self.test_value:
            # Move to right
            return 1
        else:
            # Move to left
            return 0




def generate_random_tree(vocab_length):
    import random
    nr_of_nodes = np.random.randint(3, 15)
    tree = None
    for i in range(nr_of_nodes):
        formula_length = np.random.randint(1,5)
        indices = np.random.randint(0, vocab_length, formula_length)
        parameters = np.random.uniform(0, 1, formula_length)
        test_val = formula_length*np.random.uniform(0,5)
        node = DecisionNode(indices, parameters, test_val)
        directions = np.random.randint(0,2,6)
        if tree == None:
            tree = DecisionTree(node)
        else:
            tree.add(node, directions)
    
    return tree


def mutate_tree
        
    
    
    

arr1 = [0, 1, 2, 3]

indices = [0, 3, 2, 1, 3]
values = [0, 3, 2, 1, 4]




