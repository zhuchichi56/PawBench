from pawbench.utils.anomalies import detect_anomalies


def _event(role, text="x"):
    return {
        "type": "message",
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }


def _detect(transcript, status="success"):
    return detect_anomalies(
        {
            "exit_code": 0,
            "timed_out": False,
            "status": status,
            "execution_time": 30,
            "transcript_length": len(transcript),
            "transcript": transcript,
            "stderr": "",
        },
        "",
    )


def test_complete_one_action_tool_exchange_is_not_short_transcript_error():
    transcript = [
        _event("user"),
        _event("assistant", "tool call"),
        _event("toolResult", "tool output"),
        _event("assistant", "done"),
    ]
    result = _detect(transcript)
    assert "SHORT_TRANSCRIPT" not in {item["id"] for item in result["items"]}
    assert not result["has_error"]


def test_complete_direct_exchange_is_not_short_transcript_error():
    result = _detect([_event("user"), _event("assistant", "done")])
    assert "SHORT_TRANSCRIPT" not in {item["id"] for item in result["items"]}


def test_incomplete_short_transcript_remains_an_error():
    result = _detect([_event("assistant", "partial")])
    items = {item["id"]: item for item in result["items"]}
    assert items["SHORT_TRANSCRIPT"]["severity"] == "error"
    assert result["has_error"]
