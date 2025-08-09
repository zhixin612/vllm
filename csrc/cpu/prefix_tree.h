#ifndef PREFIX_TREE_H
#define PREFIX_TREE_H

#include <vector>
#include <unordered_map>
#include <memory> // For std::unique_ptr

// Represents a node in the Radix Tree
struct Node {
    int id;
    int parent_id;
    int S = 1; // Number of sequences passing through or ending at this node
    int len = 0; // Length in base pairs (not blocks)
    int num_v = 0; // Number of child vertices
    std::vector<int> seq_indices;
    std::vector<int> block_ids;

    // Constructor for convenience
    Node(int node_id = -1, int p_id = -1) : id(node_id), parent_id(p_id) {}
};

class PrefixTree {
public:
    // Constructor
    PrefixTree(int block_size);

    // Main function to build the tree from a set of sequences
    void build_radix_tree(
        const std::vector<int>& seq_lens,
        const std::vector<std::vector<int>>& block_table);

private:
    // Internal state
    std::unordered_map<int, std::unordered_map<int, std::unique_ptr<Node>>> v;
    Node root;
    int num_nodes;
    int block_size;
    std::vector<int> split_per_seq;

    // Helper function to insert a single sequence into the tree
    void insert(int sId, int seq_len, const std::vector<int>& blocks);

    // Helper function to find the length of the common prefix in terms of blocks
    size_t get_common_block_count(
        const std::vector<int>& blocks1,
        const std::vector<int>& blocks2);
};

#endif // PREFIX_TREE_H