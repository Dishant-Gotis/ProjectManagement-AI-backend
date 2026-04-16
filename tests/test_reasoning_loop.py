import unittest
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
from models.card import CaseStudyCard
from services.llm_service import LLMService


def build_case_study_card() -> CaseStudyCard:
    return CaseStudyCard(
        concept="Earned Value Management",
        story="A delivery program used EVM metrics to spot schedule and cost drift in week three.",
        problem="The team was unsure whether to continue the current scope or re-plan.",
        decision_point="The PM had to decide between recovery actions and timeline renegotiation.",
        concept_mapping="SPI and CPI below 1.0 signaled underperformance.",
        key_lessons=["Track trends early", "Act on variance quickly", "Communicate trade-offs"],
        think_about_this="What thresholds should trigger corrective action in your project?",
    )

class TestReasoningLoop(unittest.TestCase):
    @patch("services.supabase_service.supabase_service.save_query_history")
    @patch("services.llm_service.execute_tool")
    @patch("services.llm_service.get_tool_schemas")
    @patch("services.llm_service.ChatOpenAI")
    def test_case_study_route_returns_structured_card(
        self,
        mock_chat_openai,
        mock_get_schemas,
        mock_execute_tool,
        mock_save_query_history,
    ):
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        mock_get_schemas.return_value = [{"name": "retrieve_kb"}, {"name": "web_search"}]
        mock_llm_instance.bind_tools.return_value = MagicMock()

        mock_chat_llm = MagicMock()
        mock_chat_llm.invoke.return_value = MagicMock(content="fallback text")
        mock_llm_instance.bind.return_value = mock_chat_llm

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = build_case_study_card()
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm

        mock_execute_tool.side_effect = [
            {"source": "knowledge_base", "results": [{"title": "EVM Basics"}]},
            {"source": "web_search", "results": [{"title": "NASA EVM"}]},
        ]

        service = LLMService()
        service._load_history = MagicMock(return_value=[])
        card = service.process_query("Explain EVM in project management", user_id="test-user")

        self.assertIsInstance(card, CaseStudyCard)
        self.assertEqual(card.concept, "Earned Value Management")
        self.assertEqual(mock_execute_tool.call_count, 2)
        mock_save_query_history.assert_called_once()

    @patch("services.supabase_service.supabase_service.save_query_history")
    @patch("services.llm_service.execute_tool")
    @patch("services.llm_service.get_tool_schemas")
    @patch("services.llm_service.ChatOpenAI")
    def test_greeting_fast_path_skips_tool_calls(
        self,
        mock_chat_openai,
        mock_get_schemas,
        mock_execute_tool,
        mock_save_query_history,
    ):
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        mock_get_schemas.return_value = []
        mock_llm_instance.bind_tools.return_value = MagicMock()
        mock_llm_instance.bind.return_value = MagicMock()

        service = LLMService()
        service._load_history = MagicMock(return_value=[])
        response = service.process_query("hii", user_id="test-user")

        self.assertIsInstance(response, str)
        self.assertIn("project management", response.lower())
        mock_execute_tool.assert_not_called()
        self.assertFalse(mock_llm_instance.with_structured_output.called)
        mock_save_query_history.assert_called_once()

if __name__ == '__main__':
    unittest.main()
