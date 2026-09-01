import math

# Raw judge-score threshold (1-10 scale, same as eval_results) at/above which
# a node is marked terminal by default -- see MCTSNode.__init__. A
# terminal_score <= 0 (the CLI's --terminal_score 0) disables terminal
# marking entirely, since 0 is never a valid raw judge score.
DEFAULT_TERMINAL_SCORE = 9.0


class MCTSNode:
    def __init__(self, instance, parent=None, terminal_score: float = DEFAULT_TERMINAL_SCORE):
        self.instance = instance # Chứa prompt, điểm số, cell_key...
        self.parent = parent
        self.children = []
        self.visits = 0
        self.max_score = (
            (instance.eval_results[-1] - 1) / 9.0
            if instance is not None and instance.eval_results
            else 0.0
        )
        # terminal_score <= 0 (or None) disables terminal-marking entirely --
        # see --terminal_score / DEFAULT_TERMINAL_SCORE above.
        if terminal_score is None or terminal_score <= 0:
            self.is_terminal = False
        else:
            self.is_terminal = self.max_score * 9.0 + 1 >= terminal_score

class Forest:
    def __init__(self, terminal_score: float = DEFAULT_TERMINAL_SCORE, visit_increment: str = "k"):
        # terminal_score/visit_increment are stashed on the Forest so every
        # MCTSNode created for it (add_migrant_tree here, and the driver
        # loop's per-attack child nodes) shares one consistent ablation
        # setting instead of re-reading CLI args/globals from inside Tree.py.
        self.terminal_score = terminal_score
        self.visit_increment = visit_increment
        # Dummy root không chứa prompt thật, chỉ làm quản lý
        self.root = MCTSNode(instance=None, terminal_score=terminal_score)
        self.root.visits = 1

    def add_migrant_tree(self, new_instance):
        """Khi có một nhánh từ cell khác bay sang, gắn nó vào root"""
        new_node = MCTSNode(instance=new_instance, parent=self.root, terminal_score=self.terminal_score)
        self.root.children.append(new_node)

        self.root.visits += 1

    def select_leaf(self, check = True, C: float = 1.414):
        """Thuật toán UCT đi từ root xuống tìm lá tiềm năng nhất"""
        current = self.root
        while len(current.children) > 0:
            if check:
                valid_children = [c for c in current.children if not c.is_terminal]
                if not valid_children:
                    current.is_terminal = True
                    return current # Trả về node hiện tại nếu ngõ cụt
            else:
                valid_children = [c for c in current.children]
                if not valid_children:
                    return current # Trả về node hiện tại nếu ngõ cụt
            
            # Chọn con có điểm UCT cao nhất
            current = max(valid_children, key=lambda c: self._calculate_uct(current, c, C))
        return current

    def _calculate_uct(self, parent, child, C=1.414):
        if child.visits == 0: return float('inf')
        return child.max_score + C * math.sqrt(math.log(parent.visits) / child.visits)

    def backpropagate(self, leaf_node, new_children_nodes):
        """Truyền Max-Backup và Lượt thăm lên tận root"""
        K = len(new_children_nodes)
        # visit_increment='k' (default) reproduces today's += K; '1' is
        # textbook UCT visit-counting -- see --visit_increment.
        increment = K if self.visit_increment == "k" else 1
        max_new_score = max([c.max_score for c in new_children_nodes]) if new_children_nodes else 0

        current = leaf_node
        while current is not None:
            current.visits += increment
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
            
            if node is self.root:
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

            if node is self.root:
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
        q = deque([self.root])
        while q:
            n = q.popleft()
            all_nodes.append(n)
            q.extend(n.children)
        
        total = len(all_nodes) - 1
        terminal = sum(1 for n in all_nodes if n is not self.root and n.is_terminal)
        best = max((n.max_score for n in all_nodes if n is not self.root), default=0)
        print(f"nodes={total} | terminal={terminal} | best_score={best:.3f}")
        print("─" * 60)
        
        _print_node(self.root)
        print()