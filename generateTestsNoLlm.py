import ast
import astor
import astroid
import os
import argparse
from typing import List, Dict, Any

class FunctionAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        self.functions.append({
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'returns': node.returns.id if node.returns else None,
            'body': astor.to_source(node)
        })

def analyze_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())
    analyzer = FunctionAnalyzer()
    analyzer.visit(tree)
    return analyzer.functions

def generate_test_case(function: Dict[str, Any]) -> str:
    func_name = function['name']
    args = function['args']
    returns = function['returns']

    test_case = f"""
def test_{func_name}():
    # Arrange
    {''.join([f'{arg} = None  # TODO: Replace with appropriate test value\n    ' for arg in args])}
    
    # Act
    result = {func_name}({', '.join(args)})
    
    # Assert
    {'assert result is not None  # TODO: Replace with appropriate assertion' if returns else 'assert True  # TODO: Replace with appropriate assertion'}
"""
    return test_case

def generate_tests(file_path: str) -> str:
    functions = analyze_file(file_path)
    tests = [generate_test_case(func) for func in functions]
    
    imports = f"from {os.path.splitext(os.path.basename(file_path))[0]} import *\n\n"
    return imports + '\n'.join(tests)

def main():
    parser = argparse.ArgumentParser(description="Static Analysis Unit Test Generator")
    parser.add_argument("file", type=str, help="Path to the Python file to generate tests for")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: The file {args.file} does not exist.")
        return

    tests = generate_tests(args.file)
    
    test_file_path = f"test_{os.path.basename(args.file)}"
    with open(test_file_path, "w") as test_file:
        test_file.write(tests)
    
    print(f"Tests generated and written to {test_file_path}")

if __name__ == "__main__":
    main()