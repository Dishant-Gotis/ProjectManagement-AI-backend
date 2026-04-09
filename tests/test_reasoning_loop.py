import unittest
import sys
import os
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
from services.llm_service import LLMService
from models.card import CaseStudyCard

class TestReasoningLoop(unittest.TestCase):
    @patch('services.llm_service.ChatOpenAI')
    @patch('services.llm_service.get_tool_schemas')
    def test_process_query_direct(self, mock_get_schemas, mock_chat_openai):
        """
        Test that process_query returns a valid card when the LLM responds directly (no tools).
        """
        # Mocking the LLM
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        mock_get_schemas.return_value = []
        
        # Mocking the bound LLM response
        mock_bound_llm = MagicMock()
        mock_llm_instance.bind_tools.return_value = mock_bound_llm
        
        # Mocking a direct JSON response
        mock_card_json = {
            "concept": "Scope Management",
            "story": "A project manager at a software company had to define what was in and out of scope.",
            "problem": "Uncontrolled feature creep.",
            "decision_point": "Saying no to the CEO.",
            "concept_mapping": "Demonstrates scope baseline.",
            "key_lessons": ["Lock early", "Document everything"],
            "think_about_this": "How would you handle a CEO?"
        }
        
        # Configure a plain mock object (avoiding spec issues with complex types)
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_card_json)
        mock_response.tool_calls = []
        mock_bound_llm.invoke.return_value = mock_response

        # Run the service
        service = LLMService()
        card = service.process_query("Tell me about scope")

        # Assertions
        self.assertIsInstance(card, CaseStudyCard)
        self.assertEqual(card.concept, "Scope Management")
        self.assertEqual(len(card.key_lessons), 2)

    @patch('services.llm_service.ChatOpenAI')
    @patch('services.llm_service.execute_tool')
    @patch('services.llm_service.get_tool_schemas')
    def test_process_query_with_tool_loop(self, mock_get_schemas, mock_execute_tool, mock_chat_openai):
        """
        Test that process_query handles tool calls and then synthesizes a card.
        """
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        mock_get_schemas.return_value = [{"name": "retrieve_kb"}]
        
        mock_bound_llm = MagicMock()
        mock_llm_instance.bind_tools.return_value = mock_bound_llm
        
        # Turn 1: LLM calls a tool
        mock_resp_1 = MagicMock()
        mock_resp_1.tool_calls = [{"name": "retrieve_kb", "args": {"query": "EVM"}, "id": "call_1"}]
        mock_resp_1.content = ""
        
        # Turn 2: LLM provides final synthesis
        mock_card_json = {
            "concept": "EVM",
            "story": "EVM was used.",
            "problem": "Cost overruns.",
            "decision_point": "Stop or go?",
            "concept_mapping": "CPI < 1.",
            "key_lessons": ["Watch costs"],
            "think_about_this": "Why CPI matters?"
        }
        mock_resp_2 = MagicMock()
        mock_resp_2.tool_calls = []
        mock_resp_2.content = json.dumps(mock_card_json)
        
        # Set sequence of responses
        mock_bound_llm.invoke.side_effect = [mock_resp_1, mock_resp_2]
        mock_execute_tool.return_value = {"content": "EVM results"}

        # Run the service
        service = LLMService()
        card = service.process_query("Explain EVM")

        # Assertions
        self.assertEqual(mock_execute_tool.call_count, 1)
        self.assertEqual(card.concept, "EVM")

if __name__ == '__main__':
    unittest.main()
