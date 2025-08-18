from langchain_google_vertex import ChatVertexAI
llm = ChatVertexAI(model_name="gemini-2.0-flash-001")

#Define our toy Pydantic model and let's assume we got an invalid output that can't parsed
from langchain.output_parsers import RetryWithErrorOutputParser,PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

class SearchAction(BaseModel):
  query: str = Field(description="A query to search for if a search action is taken")

#PydanticOutputParser enforces type safety for structured outputs using Pydantic schemas
parser = PydanticOutputParser(pydantic_object=SearchAction)
completion_with_error ="{'action': 'what is the weather likein Munich tomorrow}"

try:
  parser.parse(completion_with_error)
  except Exception as e:
    print(e)#Invalid json output: {'action': 'what is the weather likein Munich tomorrow}



"""Solution: 
1. Define our output parser with retry.
2. RetryWithErrorOutputParser: Wrap a parser and try to fix parsing errors. Does this by passing the original prompt, the completion, AND the error that was raised to another language model.
"""
fix_parser = RetryWithErrorOutputParser.from_llm(
  llm=llm,
  parser=parser
)

#Inspect the default retry prompt it's using:
from langchain.output_parsers.retry import NAIVE_RETRY_PROMPT
print(NAIVE_RETRY_PROMPT)

#Define a template for the prompt value to be substituted in the default promp
retry_template =(
  "Your previous response doesn't follow the required schema and failed parsing.Fix the response so that it follows the expected schema."
  "Do not change the nature of response, only adjust the schema."
  "\n\nEXPECTED SCHEMA:{schema}\n\n"
)

retry_prompt = PromptTemplate.from_template(retry_template)

# Run parser and validate if parsing error is fixed
fixed_output= fix_parser.parse_with_prompt(
  completion=completion_with_error,
  prompt_value=retry_prompt.format_prompt(schema=parser.get_format_instructions())
)

print(fixed_output) #SearchAction(query='what is the weather like in Munich tomorrow')


