#!/usr/bin/env python3

import os
import argparse
import importlib
import ast
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import git
import radon.complexity as radon
from coverage import Coverage

# Constants
LARGE_FILE_SIZE_THRESHOLD = 50000  # Adjust the threshold as needed
CHUNK_SIZE = 2000

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        import openai
        openai.api_key = api_key
        self.client = openai.ChatCompletion

    def generate(self, prompt: str, max_tokens: int) -> str:
        response = self.client.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

class GeminiClient(LLMClient):
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel('gemini-pro')

    def generate(self, prompt: str, max_tokens: int) -> str:
        response = self.client.generate_content(prompt)
        return response.text.strip()

class ClaudeClient(LLMClient):
    def __init__(self, api_key: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int) -> str:
        response = self.client.completions.create(
            model="claude-2.1",
            prompt=prompt,
            max_tokens_to_sample=max_tokens
        )
        return response.completion.strip()

def initialize_client(model: str, api_keys: Dict[str, str]) -> LLMClient:
    if model == 'openai':
        return OpenAIClient(api_keys['openai'])
    elif model == 'gemini':
        return GeminiClient(api_keys['gemini'])
    elif model == 'claude':
        return ClaudeClient(api_keys['claude'])
    else:
        raise ValueError("Unsupported model")

def read_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def extract_imports(code_content: str) -> List[str]:
    tree = ast.parse(code_content)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    return imports

def summarize_large_file(file_path: str, client: LLMClient) -> str:
    code_content = read_file(file_path)
    prompt = f"""
    Please summarize the following code. Include only the key parts relevant for understanding the functionality and generating unit tests.

    {code_content}

    Provide the summary in a concise format.
    """
    return client.generate(prompt, 1000)

def chunk_large_file(file_path: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    code_content = read_file(file_path)
    return [code_content[i:i + chunk_size] for i in range(0, len(code_content), chunk_size)]

def process_chunks(chunks: List[str], client: LLMClient) -> str:
    summaries = []
    for chunk in chunks:
        prompt = f"""
        Please summarize the following code. Include only the key parts relevant for understanding the functionality and generating unit tests.

        {chunk}

        Provide the summary in a concise format.
        """
        summaries.append(client.generate(prompt, 1000))
    return "\n\n".join(summaries)

def create_unified_code_file(main_file_path: str, imported_files: List[str], client: LLMClient) -> str:
    unified_code = read_file(main_file_path) + "\n\n"
    
    for file_path in imported_files:
        if os.path.exists(file_path):
            if os.path.getsize(file_path) > LARGE_FILE_SIZE_THRESHOLD:
                chunks = chunk_large_file(file_path)
                summarized_code = process_chunks(chunks, client)
            else:
                summarized_code = summarize_large_file(file_path, client)
            
            unified_code += f"# Content from {file_path}\n"
            unified_code += summarized_code + "\n\n"
    
    unified_file_path = f"unified_{os.path.basename(main_file_path)}"
    with open(unified_file_path, "w") as unified_file:
        unified_file.write(unified_code)
    
    return unified_file_path

class TestGenerator:
    def __init__(self, client: LLMClient, config: Dict[str, Any]):
        self.client = client
        self.config = config

    def generate_tests(self, file_path: str) -> str:
        code_content = read_file(file_path)
        complexity = self.analyze_complexity(code_content)
        coverage = self.analyze_coverage(file_path)
        
        prompt = f"""
        Generate unit tests for the following code. The code has a cyclomatic complexity of {complexity}.
        Current test coverage: {coverage}%.
        Focus on complex functions and areas with low coverage.
        Use the {self.config['test_framework']} testing framework.
        
        Code:
        {code_content}
        
        Include natural language descriptions for each test.
        """
        
        return self.client.generate(prompt, 2000)

    def analyze_complexity(self, code_content: str) -> float:
        return radon.cc_visit(code_content)

    def analyze_coverage(self, file_path: str) -> float:
        cov = Coverage()
        cov.start()
        importlib.import_module(os.path.splitext(os.path.basename(file_path))[0])
        cov.stop()
        return cov.report()

class GitIntegration:
    def __init__(self, repo_path: str):
        self.repo = git.Repo(repo_path)

    def get_modified_files(self) -> List[str]:
        return [item.a_path for item in self.repo.index.diff(None)]

class IncrementalTestGenerator:
    def __init__(self, test_generator: TestGenerator, git_integration: GitIntegration):
        self.test_generator = test_generator
        self.git_integration = git_integration

    def generate_incremental_tests(self) -> Dict[str, str]:
        modified_files = self.git_integration.get_modified_files()
        return {file: self.test_generator.generate_tests(file) for file in modified_files if file.endswith('.py')}

class InteractiveTestGenerator:
    def __init__(self, test_generator: TestGenerator):
        self.test_generator = test_generator

    def generate_interactive_tests(self, file_path: str) -> str:
        initial_tests = self.test_generator.generate_tests(file_path)
        print("Initial tests generated. Please review and provide any additional context or requirements.")
        
        while True:
            user_input = input("Enter additional context (or 'done' to finish): ")
            if user_input.lower() == 'done':
                break
            
            prompt = f"""
            Given the following additional context, improve or extend the previously generated tests:
            
            Additional context: {user_input}
            
            Previous tests:
            {initial_tests}
            """
            
            initial_tests = self.client.generate(prompt, 2000)
        
        return initial_tests

def main():
    parser = argparse.ArgumentParser(description="Advanced Unit Test Generator")
    parser.add_argument("file", type=str, help="Path to the main code file or repository.")
    parser.add_argument("--model", type=str, choices=['openai', 'gemini', 'claude'], help="LLM model to use (optional).")
    parser.add_argument("--incremental", action="store_true", help="Use incremental test generation.")
    parser.add_argument("--interactive", action="store_true", help="Use interactive test generation.")
    parser.add_argument("--test-framework", type=str, default="pytest", help="Specify the test framework to use.")
    
    args = parser.parse_args()

    # Load API keys from environment variables
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'gemini': os.getenv('GEMINI_API_KEY'),
        'claude': os.getenv('CLAUDE_API_KEY')
    }

    model = args.model if args.model else select_model(api_keys)
    client = initialize_client(model, api_keys)

    if not os.path.exists(args.file):
        print(f"Error: The file or repository {args.file} does not exist.")
        return
    
    config = {
        "test_framework": args.test_framework,
        # Add more configuration options as needed
    }

    test_generator = TestGenerator(client, config)

    if args.incremental:
        git_integration = GitIntegration(args.file)
        incremental_generator = IncrementalTestGenerator(test_generator, git_integration)
        test_results = incremental_generator.generate_incremental_tests()
        
        for file, tests in test_results.items():
            test_file_path = f"test_{os.path.basename(file)}"
            with open(test_file_path, "w") as test_file:
                test_file.write(tests)
            print(f"Incremental tests generated for {file} and written to {test_file_path}")
    
    elif args.interactive:
        interactive_generator = InteractiveTestGenerator(test_generator)
        tests = interactive_generator.generate_interactive_tests(args.file)
        
        test_file_path = f"test_{os.path.basename(args.file)}"
        with open(test_file_path, "w") as test_file:
            test_file.write(tests)
        print(f"Interactive tests generated and written to {test_file_path}")
    
    else:
        tests = test_generator.generate_tests(args.file)
        
        test_file_path = f"test_{os.path.basename(args.file)}"
        with open(test_file_path, "w") as test_file:
            test_file.write(tests)
        print(f"Tests generated and written to {test_file_path}")

def select_model(api_keys: Dict[str, str]) -> str:
    available_models = [model for model, key in api_keys.items() if key]
    if not available_models:
        raise ValueError("No API keys provided. Please set at least one API key.")
    return available_models[0]

if __name__ == "__main__":
    main()