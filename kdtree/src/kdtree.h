#pragma once
#include <vector>
#include <algorithm>

template<typename T>
struct TreeNode{
    typedef TreeNode* ptr;

    TreeNode(T* data, int axis) : data_(data), axis_(axis){}

    bool is_left(T* sample){
        return (*sample)[axis] < (*data_)[axis];
    }

    bool is_right(T* sample){
        return (*sample)[axis] > (*data_)[axis];
    }

    T* data_;
    int axis_;

    ptr left = nullptr;
    ptr right = nullptr;
    std::vector<T*> leaves_;

    bool has_leaves(){
        return !this->leaves_.empty();
    }
};

template<typename T>
class KDTree{
public:
    typedef typename TreeNode<T>::ptr nodePtr;
    KDTree(T* head, int size, int leaf_size = 1);  
    KDTree() = default;
    ~KDTree();

private:
    void buildTree(nodePtr& curr, T* begin, T* end);
    void nextAxis();

    nodePtr root_ = nullptr;
    int axis_ = 0;
    int size_;
    int leaf_size_;
};

template<typename T>
KDTree<T>::KDTree(T* head, int size, int leaf_size = 1) : size_(size), leaf_size_(leaf_size){
    this->buildTree(root_, head, head + size_);
}

template<typename T>
inline void KDTree<T>::nextAxis(){
    axis_ = (axis_ == T::dim_ - 1) ? 0 : axis_ + 1;
}

template<typename T>
void KDTree<T>::buildTree(nodePtr& curr, T* begin, T* end){
    int dist = std::distance(begin, end);
    T* mid = begin + dist / 2;

    std::nth_element(begin, mid, end, [=](const T* lhs, const T* rhs)){return lhs[axis_] < rhs[axis_];};
    curr = new TreeNode(mid, axis_);

    nextAxis();

    if(dist <= leaf_size_){
        curr->leaves_.reserve(leaf_size_);
        std::for_each(begin, end, [&](T& data) {curr->push_back(&data);});
    }
    else{
        buildTree(curr->left, begin, mid);
        buildTree(curr->right, mid, end);
    }
}
