# username - pollinger
# id1      - 318688991
# name1    - Eden Pollinger
# id2      - 318895877
# name2    - Ron Mondshein

"""A class representing a node in an AVL tree"""
import random
from typing import Optional


class AVLNode(object):
    """Constructor, you are allowed to add more fields.
    @type value: str
    @param value: data of your node
    when we build a new AVL we will build it with null as left and null as right
    """

    def __init__(self, value, isVirtual: bool = False):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height: int = -1 if isVirtual else 0
        self.size: int = 0 if isVirtual else 1
        self.isVirtual = isVirtual
        if not self.isVirtual:
            self.right = AVLNode(None, True)
            self.right.setParent(self)
            self.left = AVLNode(None, True)
            self.left.setParent(self)

    """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child
    """

    def getLeft(self):
        if self is not None and self.isRealNode():
            return self.left
        return None

    """returns the right child
    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    """

    def getRight(self):
        if self is not None and self.isRealNode():
            return self.right
        return None

    """returns the parent 
    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

    def getParent(self):
        return self.parent

    """return the value
    @rtype: str
    @returns: the value of self, None if the node is virtual
    """

    def getValue(self):
        if self is not None and self.isRealNode():
            return self.value
        return None

    """returns the height
    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

    def getHeight(self):
        if self is not None and self.isRealNode():
            return self.height
        return -1

    """returns the size
    @rtype: int
    @returns: the size of self, -1 if the node is virtual
    """

    def getSize(self):
        if self is not None and self.isRealNode():
            return self.size
        return 0

    """sets left child
    @type node: AVLNode
    @param node: a node
    """

    def setLeft(self, node):
        virtual_child = AVLNode(None, True)
        if node is None:
            # if None put virtualNode
            self.left = virtual_child
            virtual_child.setParent(self)

        else:
            self.left = node
            node.setParent(self)

    """sets right child
    @type node: AVLNode
    @param node: a node
    """

    def setRight(self, node):
        virtual_child = AVLNode(None, True)
        if node is None:
            # if None put virtualNode
            self.right = virtual_child
            virtual_child.setParent(self)
        else:
            self.right = node
            node.setParent(self)

    """sets parent
    @type node: AVLNode
    @param node: a node
    """

    def setParent(self, node):
        self.parent = node

    """sets value
    @type value: str
    @param value: data
    """

    def setValue(self, value):
        self.value = value

    """sets the balance factor of the node
    @type h: int
    @param h: the height
    """

    def setHeight(self, h):
        self.height = h

    """returns whether self is not a virtual node 
    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """

    def isRealNode(self):
        return not self.isVirtual

    """fix the height of a node
    complexity of O(1)
    """

    def newHeight(self):
        self.setHeight(max(self.getLeft().getHeight(), self.getRight().getHeight()) + 1)

    """fix the size of a node
    complexity of O(1)
    """

    def newSize(self):
        self.size = self.getLeft().getSize() + self.getRight().getSize() + 1

    """return the balance factor of a node
    complexity of O(1)
    @rtype: int
    @returns: the difference between the height of the node's children
    """

    def getBalanceFactor(self):
        if self.isVirtual:
            return 0
        return self.getLeft().getHeight() - self.getRight().getHeight()

    """rotate one time to the left
    complexity of O(1)
    @returns: always one, because we rotated one time per call for function
    """

    def leftRotate(self):
        right_child = self.getRight()
        left_child_of_right_child = right_child.getLeft()
        parent = self.getParent()

        if parent:
            if parent.right == self:
                parent.setRight(right_child)
            else:
                parent.setLeft(right_child)
        else:
            right_child.parent = None

        self.parent = right_child
        self.setRight(left_child_of_right_child)
        right_child.setLeft(self)
        self.newHeight()
        self.newSize()
        right_child.newSize()
        right_child.newHeight()
        return 1

    """rotate one time to the right
    complexity of O(1)
    @returns: always one, because we rotated one time per call for function
    """

    def rightRotate(self):
        left_child = self.getLeft()
        right_child_of_left_child = left_child.getRight()
        parent = self.getParent()

        if parent:
            if parent.right == self:
                parent.setRight(left_child)
            else:
                parent.setLeft(left_child)
        else:
            left_child.parent = None

        self.parent = left_child
        self.setLeft(right_child_of_left_child)
        left_child.setRight(self)

        self.newHeight()
        self.newSize()
        left_child.newSize()
        left_child.newHeight()
        return 1

    """make left and then right rotations, return 2 because it did 2 rotations
    @rtype: int
    @returns: number of rotations
    """

    def leftThenRightRotation(self):
        self.getLeft().leftRotate()
        self.rightRotate()
        return 2

    """make left and then right rotations, return 2 because it did 2 rotations
    @rtype: int
    @returns: number of rotations
    """

    def rightThenLeftRotation(self):
        self.getRight().rightRotate()
        self.leftRotate()
        return 2

    """going up until the the root
    complexity of O(logn) as the height of the tree
    @rtype: AVLNode
    """

    def findRoot(self):
        temp = self
        while temp.getParent() is not None:
            temp = temp.getParent()
        return temp

    """return the first node that came before this one (his predecessor)
    complexity o(logn)
    @rtype: AVLNode
    @returns: the predecessor of a node
    """

    def predecessor(self):
        def predecessorRec(node):
            if node.getLeft().isRealNode():
                node = node.getLeft()
                while node.getRight().isRealNode():
                    node = node.getRight()
                return node
            else:
                while node.getParent() is not None and node.getParent().getLeft() == node:
                    node = node.getParent()
                return node.getParent()

        return predecessorRec(self)

    """return the first node that came after this one (his successor)"""
    """complexity o(logn)
    @rtype: AVLNode
    @returns: the successor of a node"""

    def successor(self):
        def successorRec(node):
            if node.getRight().isRealNode():
                node = node.getRight()
                while node.getLeft().isRealNode():
                    node = node.getLeft()
                return node
            else:
                while node.getParent() is not None and node.getParent().getRight() == node:
                    node = node.getParent()
                return node.getParent()

        return successorRec(self)

    """balance the tree (correct the balance factor) 
    complexity of O(logn)
    @rtype: int
    @returns: how many times we rotated the tree to fix it
    """

    def fixTree(self):
        def fixTreeRec(node):
            count_rotates = 0
            if node is None or not node.isRealNode:
                return 0
            node.newHeight()
            node.newSize()
            if node.getBalanceFactor() < -1:
                # right have bigger BF then left
                if node.getRight().getBalanceFactor() == 1:
                    count_rotates += node.rightThenLeftRotation()
                else:
                    count_rotates += node.leftRotate()
            elif node.getBalanceFactor() > 1:
                # left have bigger BF then right
                if node.getLeft().getBalanceFactor() == -1:

                    count_rotates += node.leftThenRightRotation()
                else:
                    count_rotates += node.rightRotate()
            return count_rotates + fixTreeRec(node.getParent())

        return fixTreeRec(self)

    """
    A class implementing the ADT list, using an AVL tree.
    """


class AVLTreeList(object):
    """
        Constructor, you are allowed to add more fields.
        """

    def __init__(self):
        self.height: int = -1
        self.size: int = 0
        self.root: Optional[AVLNode] = None
        self.min: Optional[AVLNode] = None
        self.max: Optional[AVLNode] = None

    """returns whether the list is empty
    @rtype: bool
    @returns: True if the list is empty, False otherwise
    """

    def empty(self):
        return self.root is None or not self.root.isRealNode()

    """Help function to find the value of the node in place k
    complexity of O(logn)
    @rtype: AVLNode
    @returns: the node in place k
    """

    def TreeSelectRec(self, current_node: AVLNode, k: int) -> AVLNode:
        if current_node is not None and current_node.getLeft() is not None:
            left_tree_size = current_node.getLeft().getSize()
        else:
            left_tree_size = 0

        if k == left_tree_size + 1:
            return current_node

        if k <= left_tree_size:
            return self.TreeSelectRec(current_node.getLeft(), k)
        else:
            return self.TreeSelectRec(current_node.getRight(), k - left_tree_size - 1)

    """retrieves the value of the i'th item in the list
    complexity of O(logn)
    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the value of the i'th item in the list
    """

    def retrieve(self, i: int) -> any:
        if not self.empty() and self.root.getSize() > i >= 0 and self.retrieve_node(i) is not None:
            return self.retrieve_node(i).value
        return None

    """retrieves the i'th item in the list
    complexity of O(logn)
    @rtype: AVLNode
    @returns: the i'th item in the list
    """

    def retrieve_node(self, i: int) -> AVLNode:
        if i < 0 or i > self.root.getSize():
            raise Exception("bad arguments")
        return self.TreeSelectRec(self.root, i + 1)

    """inserts val at position i in the list
    complexity o(logn)
    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing

    "get- AVLNode root, index for insert and val, if we can insert return the AVLNode that we inserted"""

    def insert(self, i: int, val: any) -> int:
        new_node: AVLNode = AVLNode(val)
        if i > self.length():
            return -1

        if self.length() == 0:  # the tree was empty before the insert
            self.root = new_node
            self.min = new_node
            self.max = new_node
            return 0

        if i == self.length():  # insert to last
            self.max.setRight(new_node)
            self.max = new_node

        elif i == 0:  # insert to first
            self.min.setLeft(new_node)
            self.min = new_node

        else:  # insert in the middle
            current_parent = self.retrieve_node(i)
            if not current_parent.getLeft().isRealNode():
                current_parent.setLeft(new_node)
            else:
                predecessor = current_parent.predecessor()
                predecessor.setRight(new_node)

        # fix tree's balance, parameters and fields
        count_rotate = new_node.fixTree()
        self.root = new_node.findRoot()
        self.height = self.root.height
        return count_rotate

    """deletes the i'th item in the list
    complexity o(logn)
    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def delete(self, i: int) -> int:
        if self.empty() or self.getRoot().getSize() < (i + 1) or i < 0:
            return -1

        count_rotates = 0

        # there is only one node in the tree
        if self.root.size == 1:
            self.root = None
            self.min = None
            self.max = None
            return 0

        deleted_node = self.retrieve_node(i)

        if deleted_node.getRight().isRealNode() and deleted_node.getLeft().isRealNode():
            successor = deleted_node.successor()
            deleted_node.setValue(successor.getValue())
            deleted_node = successor

        parent = deleted_node.getParent()
        replace_node = deleted_node.getRight()
        if replace_node is None or not replace_node.isRealNode():
            if deleted_node.getLeft().isRealNode():
                replace_node = deleted_node.getLeft()
            else:  # deleting a leaf
                replace_node = None

        if parent is None:  # deleting the root
            pre = self.root.predecessor()
            if replace_node == self.root.getRight() and pre is not None and pre.isRealNode():
                replace_node = pre
                pre.setRight(self.root.getRight())
                if pre != self.root.getLeft():
                    current = pre.parent

                    while self.root.getLeft() is not None and current != self.root.getLeft():
                        current.setSize(current.getSize() - 1)
                    current.setSize(current.getSize() - 1)

                    pre.setLeft(self.root.getLeft())
            self.root = replace_node
            self.getRoot().setParent(None)
            count_rotates += self.root.fixTree()
            self.root.newSize()
        else:
            if deleted_node.getParent().getRight() == deleted_node:
                # is right son
                parent.setRight(replace_node)
            elif deleted_node.getParent().getLeft() == deleted_node:
                # is left son
                parent.setLeft(replace_node)
            count_rotates += parent.fixTree()
            self.root = self.root.findRoot()

        if i == 0:
            self.min = self.retrieve_node(0)  # updating min since we deleted the previous one

        if i == self.length():
            self.max = self.retrieve_node(self.length() - 1)  # updating max since we deleted the previous one

        self.height = self.root.height
        return count_rotates

    """returns the value of the first item in the list
    complexity o(1)
    @rtype: str
    @returns: the value of the first item, None if the list is empty
    """

    def first(self):
        if not self.empty():
            return self.min.value
        return None

    """returns the value of the last item in the list
    complexity o(1)
    @rtype: str
    @returns: the value of the last item, None if the list is empty
    """

    def last(self):
        if not self.empty():
            return self.max.value
        return None

    """returns an array representing list 
    Complexity: O(n)
    @rtype: list
    @returns: a list of strings representing the data structure
    """

    def listToArray(self):
        def listToArrayRec(node, arr):
            if node is None or not node.isRealNode():
                return arr
            else:
                listToArrayRec(node.getLeft(), arr)
                arr.append(node.value)
                listToArrayRec(node.getRight(), arr)

        helper = []
        listToArrayRec(self.root, helper)
        return helper

    """returns the size of the list
    Complexity: O(1)
    @rtype: int
    @returns: the size of the list
    """

    def length(self):
        if self.root:
            return self.root.getSize()
        return 0

    """Convert list to AVL Tree
    complexity of O(n)
    @rtype : AVLNode
    @returns: root node of balanced AVL Tree
    """
    def listToAVL(self, lst):
        if not lst:
            return None

        # Find middle and get its floor value
        mid = int((len(lst)) / 2)
        root = AVLNode(lst[mid])

        # Recursively construct the left and right subtree
        root.left = self.listToAVL(lst[:mid])
        root.right = self.listToAVL(lst[mid + 1:])

        # Return the root of the constructed tree
        return root

    """sort the info values of the list using mergeSort and inserting the values to a new tree
    Complexity: O(nlogn)
    @rtype: list
    @returns: an AVLTreeList where the values are sorted by the info of the original list.
    """

    def sort(self):
        def sortRec(tree_as_list):# sort list using merge sort
            if len(tree_as_list) > 1:
                mid = len(tree_as_list) // 2
                left_tree = tree_as_list[:mid]
                right_tree = tree_as_list[mid:]

                # Sorting the first half
                sortRec(left_tree)
                # Sorting the second half
                sortRec(right_tree)

                i = j = k = 0

                while i < len(left_tree) and j < len(right_tree):
                    if left_tree[i] <= right_tree[j]:
                        tree_as_list[k] = left_tree[i]
                        i += 1
                    else:
                        tree_as_list[k] = right_tree[j]
                        j += 1
                    k += 1

                # Checking if any element was left
                while i < len(left_tree):
                    tree_as_list[k] = left_tree[i]
                    i += 1
                    k += 1

                while j < len(right_tree):
                    tree_as_list[k] = right_tree[j]
                    j += 1
                    k += 1

        new_list = self.listToArray()
        sortRec(new_list)

        # inserting the list's values to a new tree
        new_tree = AVLTreeList()
        new_tree.root = self.listToAVL(new_list)
        return new_tree

    """permute the info values of the list
    complexity of O(n)
    @rtype: list
    @returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
    """

    def permutation(self):
        tree_to_list = self.listToArray()
        index = 0
        for x in tree_to_list:
            ran = random.randint(index, len(tree_to_list) - 1)
            temp = tree_to_list[index]
            tree_to_list[index] = tree_to_list[ran]
            tree_to_list[ran] = temp
            index += 1

        # inserting the list's values to a new tree
        new_tree = AVLTreeList()
        new_tree.root = self.listToAVL(tree_to_list)
        return new_tree

    """join the last node in lst1 to last2
    complexity- o(heights difference) (learned in class) that means
    complexity of O(logm - logn) = O(log(m/n))
    @rtype void
    @type list- AVLTREE
    @type bridge_node- AVLNODE
    @return joining lst2 after lst1 using bridge_node
    """

    def join(self, bridge_node, lst):

        if lst is None or lst.empty():
            self.insert(self.length(), bridge_node.value)
            self.max = self.retrieve_node(self.length() - 1)
            return

        if self is None or self.empty():
            lst.insert(0, bridge_node.value)
            self.root = lst.root
            self.min = bridge_node
            self.max = lst.max
            return

        if self is not None and not self.empty() and self.getRoot().getHeight() <= lst.getRoot().getHeight():
            # lst1's height is less than lst2's height
            connect_node = lst.getRoot()
            while connect_node.getHeight() > self.getRoot().getHeight() and connect_node.getLeft().isRealNode():
                # We want to make connect_node the root of a subtree with the height of lst1
                connect_node = connect_node.getLeft()

            # connect bridge_node"""
            bridge_node.setLeft(self.getRoot())

            # set new root
            if connect_node.getParent() is not None:
                bridge_node.setRight(connect_node.getParent())
            else:
                bridge_node.setRight(connect_node)
            bridge_node.size = bridge_node.getLeft().getSize() + bridge_node.getRight().getSize() + 1

            self.root = bridge_node
            self.root.size = self.root.getLeft().getSize() + self.root.getRight().getSize() + 1
            self.root.newHeight()
            self.max = lst.max
        else:  # lst1's height is bigger than lst2's height

            connect_node = lst.getRoot()
            while connect_node.getHeight() < lst.root.getHeight() and connect_node.getRight().isRealNode():
                # We want to make connect_node the root of a subtree with the height of lst2
                connect_node = connect_node.getRight()

            # # set new root

            if connect_node.getParent() is not None:
                connect_node.getParent().setRight(bridge_node)
            else:
                bridge_node.setRight(connect_node)

            bridge_node.setLeft(self.getRoot())
            self.root = bridge_node
            self.root.newSize()
            self.root.newHeight()
            self.max = lst.max
            self.height = self.root.height

    """concatenates lst to self
    complexity of O(n) 
    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def concat(self, lst):
        if self.empty() and lst.empty():
            return 0
        if lst.empty() and not self.empty():
            return self.getRoot().getHeight()

        if self.empty():
            self.root = lst.getRoot()
            self.min = AVLNode(lst.min)
            self.max = AVLNode(lst.max)
            return lst.getRoot().getHeight()

        height_difference = abs(self.getRoot().getHeight() - lst.getRoot().getHeight())
        last_node = self.max
        self.delete(self.root.getSize()-1)
        self.join(last_node, lst)
        return height_difference

    """searches for a *value* in the list
    complexity of O(n)
    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    """

    def search(self, val):
        index = 0
        tree_as_list = self.listToArray()
        for x in tree_as_list:
            if x == val:
                return index
            index += 1
        return -1

    """returns the root of the tree representing the list
    complexity of O(1)
    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """

    def getRoot(self):
        return self.root
