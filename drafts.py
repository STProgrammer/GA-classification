# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 01:38:32 2021

@author: abdka
"""

class Test():
    def __init__(self, val):
        self.val = val
        self.arr = list()
    
    def __str__(self):
        return str(self.val)
    
    def __repr__(self):
        return str(self.val)
    
    def __lt__(self, other):
        return self.val < other.val
    
    def add(self, arr = []):
        self.arr.append(1)
        if len(self.arr) > 10:
            return
        self.add(self.arr)
        




t1 = Test(1)

t1.add()

print(t1.arr)