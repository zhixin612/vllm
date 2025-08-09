#include "prefix_tree.h"
#include <algorithm> // For std::min

// Constructor
PrefixTree::PrefixTree(int b_size)
    : root(-1, -1), num_nodes(0), block_size(b_size) {
    // The root node is virtual and doesn't count towards num_nodes
}

/**
 * @brief Finds the number of common blocks at the start of two block vectors.
 * @note This uses a direct linear scan, which is more efficient than the
 * original binary search for typical prefix comparisons.
 * @return The number of matching blocks from the beginning.
 */
size_t PrefixTree::get_common_block_count(
    const std::vector<int>& blocks1,
    const std::vector<int>& blocks2) {

    size_t common_count = 0;
    size_t len_to_check = std::min(blocks1.size(), blocks2.size());
    while (common_count < len_to_check && blocks1[common_count] == blocks2[common_count]) {
        common_count++;
    }
    return common_count;
}

// Inserts a single sequence into the Radix Tree
void PrefixTree::insert(int sId, int seq_len, const std::vector<int>& blocks) {
    Node* current_node = &root;
    size_t blocks_processed = 0;

    while (blocks_processed < blocks.size()) {
        bool no_common_path = true;

        // Use a temporary variable to allow iterating while modifying the map
        int child_to_split = -1;

        for (auto const& [child_id, child_node_ptr] : v[current_node->id]) {
            size_t common_block_count = get_common_block_count(
                std::vector<int>(blocks.begin() + blocks_processed, blocks.end()),
                child_node_ptr->block_ids
            );

            if (common_block_count > 0) {
                no_common_path = false;

                // Case 1: The incoming sequence perfectly matches a child's prefix
                if (common_block_count == child_node_ptr->block_ids.size()) {
                    child_node_ptr->S++;
                    child_node_ptr->seq_indices.push_back(sId);
                    current_node = child_node_ptr.get();
                    blocks_processed += common_block_count;

                }
                // Case 2: Split is required
                else {
                    // This child needs to be split. We'll handle it outside the loop.
                    child_to_split = child_id;
                }

                // Break from the for-loop to handle the found path or the split
                break;
            }
        }

        // Handle the split outside the iteration loop to avoid invalidating iterators
        if (child_to_split != -1) {
            auto& parent_children_map = v[current_node->id];

            // 1. Take ownership of the child that will be split
            std::unique_ptr<Node> original_child_node = std::move(parent_children_map.at(child_to_split));
            parent_children_map.erase(child_to_split);

            // 2. Create the new intermediate node for the common part
            size_t common_block_count = get_common_block_count(
                 std::vector<int>(blocks.begin() + blocks_processed, blocks.end()),
                 original_child_node->block_ids
            );

            auto mid_node = std::make_unique<Node>(num_nodes++, current_node->id);
            mid_node->S = original_child_node->S + 1;
            mid_node->len = common_block_count * block_size;
            mid_node->num_v = 2;
            mid_node->seq_indices = original_child_node->seq_indices;
            mid_node->seq_indices.push_back(sId);
            mid_node->block_ids.assign(
                original_child_node->block_ids.begin(),
                original_child_node->block_ids.begin() + common_block_count
            );

            // 3. Update the original child and make it a child of `mid_node`
            original_child_node->parent_id = mid_node->id;
            original_child_node->block_ids.erase(
                original_child_node->block_ids.begin(),
                original_child_node->block_ids.begin() + common_block_count
            );
            original_child_node->len -= mid_node->len;

            // 4. Create the new node for the remainder of the inserted sequence
            auto new_leaf_node = std::make_unique<Node>(num_nodes++, mid_node->id);
            new_leaf_node->seq_indices.push_back(sId);
            new_leaf_node->len = seq_len - (blocks_processed + common_block_count) * block_size;
            new_leaf_node->block_ids.assign(
                blocks.begin() + blocks_processed + common_block_count,
                blocks.end()
            );

            // 5. Wire everything up
            auto& mid_children_map = v[mid_node->id];
            mid_children_map[original_child_node->id] = std::move(original_child_node);
            mid_children_map[new_leaf_node->id] = std::move(new_leaf_node);

            parent_children_map[mid_node->id] = std::move(mid_node);

            blocks_processed = blocks.size(); // Mark as fully processed
        }
        else if (no_common_path) {
            // Case 3: No common path found, create a new child for the remainder
            auto new_node = std::make_unique<Node>(num_nodes++, current_node->id);
            new_node->len = seq_len - blocks_processed * block_size;
            new_node->seq_indices.push_back(sId);
            new_node->block_ids.assign(blocks.begin() + blocks_processed, blocks.end());

            current_node->num_v++;
            v[current_node->id][new_node->id] = std::move(new_node);

            blocks_processed = blocks.size(); // Mark as fully processed
        }
    }
}

// Main function to build the tree
void PrefixTree::build_radix_tree(
    const std::vector<int>& seq_lens,
    const std::vector<std::vector<int>>& block_table) {

    for (size_t i = 0; i < block_table.size(); ++i) {
        if (i < seq_lens.size()) {
             insert(i, seq_lens[i], block_table[i]);
        }
    }
    this->split_per_seq.assign(block_table.size(), 0);
}