import argparse
import os
import ast
from typing import List, Dict
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI, Anthropic, GooglePalm
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

# Environment variables for API keys (you should set these in your environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class CodeAnalysisTool(BaseTool):
    name = "Code Analysis"
    description = "Analyzes the given code and provides insights about its structure, imports, and dependencies."

    def _run(self, file_path: str) -> str:
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            tree = ast.parse(content)
            
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            analysis = f"File: {file_path}\n"
            analysis += f"Number of lines: {len(content.splitlines())}\n"
            analysis += f"Number of imports: {len(imports)}\n"
            analysis += f"Number of functions: {len(functions)}\n"
            analysis += f"Number of classes: {len(classes)}\n\n"
            
            analysis += "Imports:\n"
            for imp in imports:
                if isinstance(imp, ast.Import):
                    for alias in imp.names:
                        analysis += f"- {alias.name}\n"
                elif isinstance(imp, ast.ImportFrom):
                    for alias in imp.names:
                        analysis += f"- from {imp.module} import {alias.name}\n"
            
            analysis += "\nFunctions:\n"
            for func in functions:
                analysis += f"- {func.name}\n"
            
            analysis += "\nClasses:\n"
            for cls in classes:
                analysis += f"- {cls.name}\n"
            
            return analysis
        except Exception as e:
            return f"Error analyzing file: {str(e)}"

    def _arun(self, file_path: str):
        raise NotImplementedError("This tool does not support async")

class TestGenerationTool(BaseTool):
    name = "Test Generation"
    description = "Generates unit tests for the given code based on the analysis and LLM output."

    def _run(self, code: str, analysis: str) -> str:
        prompt = f"""
        Based on the following code analysis, generate unit tests for the most critical parts of the code.
        Focus on testing the main functions and methods of classes.
        
        Code Analysis:
        {analysis}
        
        Generate pytest-style unit tests for this code. Include appropriate imports and use best practices for unit testing.
        """
        
        # In a real implementation, you would use an LLM here to generate the tests
        # For this example, we'll provide a template-based generation
        tests = f"""
        import pytest
        from {code.split('/')[-1].split('.')[0]} import *

        def test_placeholder():
            assert True

        # Add more specific tests based on the functions and classes in the analysis
        """
        
        return tests

    def _arun(self, code: str, analysis: str):
        raise NotImplementedError("This tool does not support async")

class UnitTestGeneratorInput(BaseModel):
    file_path: str = Field(description="Path to the file for which to generate unit tests")

class UnitTestGenerator:
    def __init__(self):
        self.code_analysis_tool = CodeAnalysisTool()
        self.test_generation_tool = TestGenerationTool()
        
        self.llms = {
            "openai": ChatOpenAI(temperature=0, model_name="gpt-4"),
            "anthropic": ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229"),
            "google": GooglePalm(temperature=0)
        }

        self.tools = [
            Tool(
                name="Code Analysis",
                func=self.code_analysis_tool._run,
                description=self.code_analysis_tool.description
            ),
            Tool(
                name="Test Generation",
                func=self.test_generation_tool._run,
                description=self.test_generation_tool.description
            )
        ]

        self.template = """
        You are an AI assistant tasked with generating unit tests for a given code file.
        The file path is: {file_path}
        
        To generate appropriate unit tests, follow these steps:
        1. Analyze the code using the Code Analysis tool.
        2. Based on the analysis, generate unit tests using the Test Generation tool.
        3. Ensure the tests cover different scenarios and edge cases.
        4. If the file is large or complex, consider generating tests for the most critical parts first.
        
        Human: Generate unit tests for the given file.
        AI: Certainly! I'll analyze the code and generate unit tests for the file at {file_path}. Let's begin with the analysis.

        Action: Code Analysis
        Action Input: {file_path}
        
        Human: Here's the result of the code analysis. Please generate the unit tests based on this information.
        AI: Thank you for providing the code analysis. I'll now use this information to generate appropriate unit tests.

        Action: Test Generation
        Action Input: {{
            "code": "{file_path}",
            "analysis": "<result_of_code_analysis>"
        }}
        
        Human: Great! Can you please provide the generated unit tests?
        AI: Certainly! Here are the generated unit tests based on the code analysis and the file content:

        {test_generation_result}

        These unit tests cover the main functionalities of the code in {file_path}. You may need to adjust them slightly based on your specific testing framework and any additional requirements. Let me know if you need any further assistance or explanations about the generated tests.

        Human: Thank you. That's all I needed.
        AI: You're welcome! I'm glad I could help you generate unit tests for your code file. If you have any more questions or need assistance with anything else, please don't hesitate to ask. Good luck with your testing!

        Human: Thanks for your help. Goodbye!
        AI: Goodbye! It was a pleasure assisting you. If you need help with unit testing or any other programming tasks in the future, feel free to ask. Have a great day!
        """

        self.prompt = StringPromptTemplate(
            input_variables=["file_path"],
            template=self.template
        )

        self.llm_chain = LLMChain(llm=self.llms["anthropic"], prompt=self.prompt)
        
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=None,
            stop=["\nHuman:"],
            allowed_tools=[tool.name for tool in self.tools]
        )

        self.memory = ConversationBufferMemory(memory_key="chat_history")

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory
        )

    def generate_tests(self, file_path: str) -> str:
        try:
            return self.agent_executor.run(file_path=file_path)
        except Exception as e:
            return f"Error generating tests: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Generate unit tests for a given file.")
    parser.add_argument("file_path", type=str, help="Path to the file for which to generate unit tests")
    args = parser.parse_args()

    generator = UnitTestGenerator()
    result = generator.generate_tests(args.file_path)
    print(result)

if __name__ == "__main__":
    main()