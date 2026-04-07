import math


class MCTSNode:
    def __init__(self, instance, parent=None):
        self.instance = instance # Chứa prompt, điểm số, cell_key...
        self.parent = parent
        self.children = []
        self.visits = 0
        self.max_score = (
            (instance.eval_results[-1] - 1) / 9.0
            if instance is not None and instance.eval_results
            else 0.0
        )
        self.is_terminal = self.max_score >= ((9 - 1) / 9.0)

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
            if current.children:
                current.is_terminal = all(c.is_terminal for c in current.children)
            current = current.parent
    def visualize(self, C: float = 1.414):
        """In cây dạng text ra terminal để debug nhanh."""
    
        def _uct_str(parent, child):
            if child.visits == 0:
                return "uct=∞"
            return f"uct={self._calculate_uct(parent, child, C):.3f}"

        def _node_tag(node):
            if node.is_terminal:
                return "[TERM]"
            if node.max_score >= 0.6:
                return "[HIGH]"
            return ""

        def _print_node(node, prefix="", is_last=True, parent=None):
            connector = "└── " if is_last else "├── "
            
            if node is self.dummy_root:
                label = f"[ROOT] visits={node.visits} children={len(node.children)}"
            else:
                uct = _uct_str(parent, node) if parent else ""
                tag = _node_tag(node)
                label = (
                    f"score={node.max_score:.3f} "
                    f"visits={node.visits} "
                    f"{uct} {tag}"
                )
                # Thêm cell_key nếu instance có
                if hasattr(node.instance, 'cell_key'):
                    label = f"[{node.instance.cell_key}] " + label

            if node is self.dummy_root:
                print(label)
            else:
                print(prefix + connector + label)

            children = node.children
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(children):
                _print_node(
                    child,
                    prefix=child_prefix,
                    is_last=(i == len(children) - 1),
                    parent=node,
                )

        print("\n" + "─" * 60)
        
        # Stats tổng quan
        all_nodes = []
        from collections import deque
        q = deque([self.dummy_root])
        while q:
            n = q.popleft()
            all_nodes.append(n)
            q.extend(n.children)
        
        total = len(all_nodes) - 1
        terminal = sum(1 for n in all_nodes if n is not self.dummy_root and n.is_terminal)
        best = max((n.max_score for n in all_nodes if n is not self.dummy_root), default=0)
        print(f"nodes={total} | terminal={terminal} | best_score={best:.3f}")
        print("─" * 60)
        
        _print_node(self.dummy_root)
        print()