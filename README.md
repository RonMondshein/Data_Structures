# Data_Structures
Second year, first semester, Data Structures course. Contains 2 projects- AVLTree and Fibonacci_Heap. 

*******
AVLTree
*******
In this project I implemented an AVLTree according to the assignment I got. An AVL tree is a self-balancing binary search tree. In an AVL tree, the heights of the two child subtrees of any node differ by at most one, and if at any time they differ by more than one, rebalancing is done to restore this property. Lookup, insertion, and deletion all take O(log n) time in both the average and worst cases, where n is the number of nodes in the tree prior to the operation. Insertions and deletions may require the tree to be rebalanced by one or more tree rotations.

**************
Fibonacci_Heap
**************
In this project I implemented an Fibonacci_Heap according to the assignment I got. Fibonacci Heap is a data structure used for implementing priority queues with efficient insertions, deletions, and extract-min operations. It is a collection of trees with a "lazy" structure that allows for marked nodes to be consolidated later. The key properties of Fibonacci Heap are: the minimum element is stored at the root, each node has an associated degree, each tree satisfies the minimum-heap property, and the degree of any node is at most O(log n). Fibonacci Heap has an amortized time complexity of O(1) for insert, find-min, decrease-key and meld operations and O(log n) for delete-min, where n is the number of nodes in the heap. 
