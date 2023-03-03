# Python program for traversal of a linked list
# Node class

from avl_template_new import AVLTreeList
import time
import array
import random


class Node:

    # Function to initialise the node object
    def __init__(self, data):
        self.data = data  # Assign data
        self.next = None  # Initialize next as null


# Linked List class contains a Node object
class LinkedList:
    def __init__(self):
        self.head = None

    def add_to_head(self, value):
        new_node = Node(value)
        new_node.next = self.head
        self.head = new_node

    def add_to_tail(self, value):
        new_node = Node(value)
        if self.head is None:
            # The linked list is empty, so the new node is the head
            self.head = new_node
        else:
            # Traverse the list to find the last node
            current_node = self.head
            while current_node.next is not None:
                current_node = current_node.next
            # Set the next attribute of the last node to point to the new node
            current_node.next = new_node

    def add_at_index(self, index, value):
        new_node = Node(value)
        if index == 0:
            # Inserting at the beginning of the list
            new_node.next = self.head
            self.head = new_node
        else:
            # Find the node at the index before the specified index
            prev_node = self._get_node(index - 1)
            if prev_node is None:
                # The index is out of range
                return
            # Set the next attribute of the new node to point to the node at the specified index
            new_node.next = prev_node.next
            # Set the next attribute of the node at the index before the specified index to point to the new node
            prev_node.next = new_node

    def _get_node(self, index):
        """Helper function to find the node at a specific index in the list"""
        current_node = self.head
        for _ in range(index):
            if current_node is None:
                # The index is out of range
                return None
            current_node = current_node.next
        return current_node

    def length(self):
        # Traverse the list and count the number of nodes
        count = 0
        current = self.head
        while current is not None:
            count += 1
            current = current.next
        return count

    def print_list(self):
        # Traverse the list and print the data of each node
        current = self.head
        while current is not None:
            print(current.data)
            current = current.next


# Code execution starts here
if __name__ == '__main__':
    numOfCheck = 10
    n = 1500 * numOfCheck

    """
    for i = 0 and i = last
    for i = 0 change 
    llist.add_to_tail(x) -> llist.add_to_head
    array1.append -> array1.insert(0,x)
    AVLtree.insert(AVLtree.length(), x) -> AVLtree.insert(0, x)
 
    """
    set = set(range(0, n))

    start_time_list = time.time()
    llist = LinkedList()

    """
    for x in set:
        llist.add_to_tail(x)
    end_time_list = time.time()
    elapsed_time_list = end_time_list - start_time_list
    print(f"Elapsed time for llist: {elapsed_time_list:.20f} seconds")
    print(f"Average time for llist: {elapsed_time_list / n:.20f} seconds")

    start_time_array = time.time()
    array1 = []

    for x in set:
        array1.append(x)
    end_time_array = time.time()
    elapsed_time_array = end_time_array - start_time_array
    print(f"Elapsed time for array: {elapsed_time_array:.20f} seconds")
    print(f"Average time for array: {elapsed_time_array / n:.20f} seconds")

    start_time_AVLtree = time.time()
    AVLtree = AVLTreeList()

    for x in set:
        AVLtree.insert(AVLtree.length(), x)
    end_time_AVLtree = time.time()
    elapsed_time_AVLtree = end_time_AVLtree - start_time_AVLtree
    print(f"Elapsed time for AVLTree: {elapsed_time_AVLtree:.20f} seconds")
    print(f"Average time for AVLTree: {elapsed_time_AVLtree / n:.20f} seconds")
    

    for the random part
    """

    # Create an array of integers
    my_array = array.array('i')

    # Generate a list of all the possible elements (in this case, integers from 0 to 1500)
    elements = list(range(n))

    # Shuffle the elements using the random.shuffle() function
    random.shuffle(elements)
    llist2 = LinkedList()
    # Add each element to the array
    for i in range(n):
        if llist2.length() == 0:
            llist2.add_to_head(0)
        else:
            j = random.randint(0, llist2.length() - 1)
            llist2.add_at_index(j, j)
    end_time_list = time.time()
    elapsed_time_list = end_time_list - start_time_list
    print(f"Elapsed time for llist: {elapsed_time_list:.20f} seconds")
    print(f"Average time for llist: {elapsed_time_list / n:.20f} seconds")


    start_time_array = time.time()
    array2 = []

    for i in range(n):
        if len(array2) == 0:
            array2.insert(0, 0)
        else:
            j = random.randint(0, len(array2) - 1)
            array2.insert(j, j)
    end_time_array = time.time()
    elapsed_time_array = end_time_array - start_time_array
    print(f"Elapsed time for array: {elapsed_time_array:.20f} seconds")
    print(f"Average time for array: {elapsed_time_array / n:.20f} seconds")


    start_time_AVLtree = time.time()
    AVLtree = AVLTreeList()

    for i in range(n):
        if AVLtree.length() == 0:
            AVLtree.insert(0, 0)
        else:
            j = random.randint(0, AVLtree.length() - 1)
            AVLtree.insert(j, j)
    end_time_AVLtree = time.time()
    elapsed_time_AVLtree = end_time_AVLtree - start_time_AVLtree
    print(f"Elapsed time for AVLTree: {elapsed_time_AVLtree:.20f} seconds")
    print(f"Average time for AVLTree: {elapsed_time_AVLtree / n:.20f} seconds")