# Save this code as diagnose_app.py in the same folder as app.py

import ast
import os

class FunctionScanner(ast.NodeVisitor):
    """
    An AST visitor to find function definitions and calls.
    """
    def __init__(self):
        self.defined_functions = set()
        self.called_functions = {}  # {caller: [callee1, callee2]}

    def visit_FunctionDef(self, node):
        self.defined_functions.add(node.name)
        # Store calls made *within* this function
        self.current_caller = node.name
        self.called_functions[self.current_caller] = []
        self.generic_visit(node)
        self.current_caller = None

    def visit_Call(self, node):
        if hasattr(self, 'current_caller') and self.current_caller:
            if isinstance(node.func, ast.Name):
                self.called_functions[self.current_caller].append(node.func.id)
        self.generic_visit(node)

def analyze_app_script(filepath="app.py"):
    """
    Analyzes the app.py script for NameErrors related to function calls.
    """
    print(" V&V Sentinel Code Integrity Check ".center(80, "="))
    
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: Could not find '{filepath}'. Make sure this script is in the same directory.")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        source_code = f.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"\n❌ CRITICAL ERROR: The script has a SyntaxError and cannot be parsed.")
        print(f"   Error: {e}")
        return

    scanner = FunctionScanner()
    scanner.visit(tree)

    defined_funcs = scanner.defined_functions
    called_funcs = scanner.called_functions
    
    errors_found = 0
    
    print("\n[1] Checking for calls to undefined helper/plotting functions...")
    for caller, callees in called_funcs.items():
        if caller.startswith("render_"):
            for callee in callees:
                # We are interested in plotting functions, not Streamlit functions like st.markdown
                if callee.startswith("plot_") and callee not in defined_funcs:
                    print(f"  - 🐞 BUG FOUND: The UI function `{caller}` calls a plotting function named `{callee}`, but this function is NOT DEFINED in the script.")
                    errors_found += 1
    
    if errors_found == 0:
        print("  - ✅ PASSED: All plotting functions called by UI functions are defined.")

    # --- Analysis for PAGE_DISPATCHER ---
    print("\n[2] Checking the PAGE_DISPATCHER dictionary...")
    dispatcher_errors = 0
    dispatcher_keys = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and node.targets[0].id == 'PAGE_DISPATCHER':
            if isinstance(node.value, ast.Dict):
                for key_node, value_node in zip(node.value.keys, node.value.values):
                    if isinstance(value_node, ast.Name):
                        dispatcher_keys.add(value_node.id)

    for func_name in sorted(list(dispatcher_keys)):
        if func_name not in defined_funcs:
            print(f"  - 🐞 BUG FOUND: The PAGE_DISPATCHER references `{func_name}`, but this function is NOT DEFINED.")
            dispatcher_errors += 1

    if dispatcher_errors == 0:
        print("  - ✅ PASSED: All functions in PAGE_DISPATCHER are defined.")
        
    total_errors = errors_found + dispatcher_errors
    
    print("\n" + " Integrity Check Summary ".center(80, "-"))
    if total_errors > 0:
        print(f"\n🔴 Total Issues Found: {total_errors}")
        print("   Please review the log above to find and fix the missing function definitions.")
    else:
        print("\n🟢 SUCCESS: No `NameError` issues related to function definitions were found.")
    print("=" * 80)


if __name__ == "__main__":
    analyze_app_script()
