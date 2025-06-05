from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

prompt = PromptTemplate.from_template("""
Extract key claim information from the following text:
{text}

Return:
- Incident Date
- Damaged Items/Property
- Claimed Amounts (if stated)
- Cause of Damage
- Supporting Documents (if mentioned)
""")

def extract_key_info(text):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(text)
    return result.strip()
