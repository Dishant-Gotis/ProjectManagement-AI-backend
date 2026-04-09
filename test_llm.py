import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from services.llm_service import llm_service
from langchain_core.messages import HumanMessage

print("Testing process_query...")
try:
    card = llm_service.process_query("What is Agile?")
    print("CARD GENERATED:", card.concept)
except Exception as e:
    print("Error during process_query:", e)
