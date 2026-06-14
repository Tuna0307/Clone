import followup.intent as fi
from followup.context import FollowupIntent


def test_fallback_intent_from_query_summary():
    intent = fi._fallback_intent_from_query("what other issues?", None)
    assert intent is not None
    assert intent.ask_type == "summary"
    assert intent.confidence >= 0.45


def test_fallback_intent_from_query_timeline():
    intent = fi._fallback_intent_from_query("show me the timeline", None)
    assert intent.ask_type == "timeline"


def test_format_chat_history():
    history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    text = fi._format_chat_history(history)
    assert "User:" in text
    assert "hello" in text
