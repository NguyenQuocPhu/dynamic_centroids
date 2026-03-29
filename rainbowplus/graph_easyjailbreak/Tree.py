import math

class MCTSNode:
    def __init__(self, instance, parent=None):
        self.instance = instance # Chứa prompt, điểm số, cell_key...
        self.parent = parent
        self.children = []
        self.visits = 0
        self.max_score = (instance.eval_results[-1] - 1) / 9.0 if instance.eval_results else 0.0
        self.is_terminal = self.max_score >= 1.0

class Forest:
    def __init__(self):
        # Dummy root không chứa prompt thật, chỉ làm quản lý
        self.dummy_root = MCTSNode(instance=None) 
        self.dummy_root.visits = 1 

    def add_migrant_tree(self, new_instance):
        """Khi có một nhánh từ cell khác bay sang, gắn nó vào dummy_root"""
        new_node = MCTSNode(instance=new_instance, parent=self.dummy_root)
        self.dummy_root.children.append(new_node)

        self.dummy_root.visits += 1

    def select_leaf(self):
        """Thuật toán UCT đi từ dummy_root xuống tìm lá tiềm năng nhất"""
        current = self.dummy_root
        while len(current.children) > 0:
            valid_children = [c for c in current.children if not c.is_terminal]
            if not valid_children:
                current.is_terminal = True
                return current # Trả về node hiện tại nếu ngõ cụt
            
            # Chọn con có điểm UCT cao nhất
            current = max(valid_children, key=lambda c: self._calculate_uct(current, c))
        return current

    def _calculate_uct(self, parent, child, C=1.414):
        if child.visits == 0: return float('inf')
        return child.max_score + C * math.sqrt(math.log(parent.visits) / child.visits)

    def backpropagate(self, leaf_node, new_children_nodes):
        """Truyền Max-Backup và Lượt thăm lên tận dummy_root"""
        K = len(new_children_nodes)
        max_new_score = max([c.max_score for c in new_children_nodes]) if new_children_nodes else 0
        
        current = leaf_node
        while current is not None:
            current.visits += K
            if max_new_score > current.max_score:
                current.max_score = max_new_score
            current = current.parent